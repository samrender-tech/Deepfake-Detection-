"""F14 - the training engine.

Config-driven, device-agnostic, resumable. Every run writes the provenance
artefacts (contract 5.4) so a number in the paper can be traced to the commit,
config and seed that produced it.

Two things here exist specifically to protect the headline result:

* ``--eval-splits`` defaults to ``val`` only. Writing test-set predictions
  requires an explicit flag, because the moment training routinely emits test
  scores, someone starts watching them and the cross-dataset number stops
  being honest.
* Early stopping, calibration and the threshold all read the SOURCE validation
  split and nothing else.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddetect.calibrate import fit_calibration
from ddetect.data.augment import AudioAugConfig, AugConfig
from ddetect.data.dataset import (
    AudioStratifiedSampler,
    BalancedMethodSampler,
    DataConfig,
    DeepfakeClipDataset,
    collate,
)
from ddetect.data.sbi import SBIConfig
from ddetect.losses import CompositeLoss
from ddetect.models.registry import build_model, describe, make_config
from ddetect.ood import MahalanobisOOD, energy_score
from ddetect.utils.device import amp_dtype, pick_device, supports_amp
from ddetect.utils.log import JsonlWriter, get_logger, setup_logging, stamp_run
from ddetect.utils.seed import seed_everything, worker_init_fn

log = get_logger(__name__)


# ==========================================================================
@dataclass
class TrainConfig:
    exp: str = "smoke"
    model: str = "smoke"
    seed: int = 0
    run_root: str = "runs"

    # data
    manifest: str = "data/manifests/fixture.parquet"
    cache_root: str = "processed"
    n_frames: int = 32
    image_size: int = 299
    batch_size: int = 8
    num_workers: int = 2
    audio_seconds: float = 4.0
    max_sync_windows: int = 6
    sampler: str = "balanced"  # balanced | audio_stratified | none

    # optimisation
    epochs: int = 10
    lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    grad_clip: float = 1.0
    accum_steps: int = 1
    amp: bool = True

    # staged schedule
    freeze_epochs: int = 2  # epochs in stage I before unfreezing

    # loss weights
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    lambda_mask: float = 0.0
    lambda_supcon: float = 0.0
    lambda_sync: float = 0.0

    # domain generalisation (ablation grid). All default OFF: DomainBed
    # found almost none of these reliably beat plain ERM under a fair protocol,
    # so they are reported as ablations against an honest baseline rather than
    # adopted. SWAD below is the one that survived that scrutiny.
    dann: bool = False
    dann_weight: float = 0.1
    groupdro: bool = False
    groupdro_eta: float = 0.01
    irm: bool = False
    irm_weight: float = 1.0
    domain_key: str = "forgery_method"

    # averaging
    ema_decay: float = 0.999
    use_ema: bool = True
    swad: bool = False  # flat-minima averaging (Cha et al. 2021)
    swad_start_epoch: int = 3

    # early stopping -- on SOURCE VAL only
    patience: int = 5
    monitor: str = "val_auc"

    # tracking. The run directory stays the source of truth; a tracker is a
    # nicer way to look at runs, never where a paper number comes from.
    tracker: str = "none"

    # evaluation
    eval_splits: tuple[str, ...] = ("val",)
    calibration: str = "temperature"
    conformal_alpha: float = 0.1

    # augmentation / SBI
    aug: dict[str, Any] = field(default_factory=dict)
    audio_aug: dict[str, Any] = field(default_factory=dict)
    sbi: dict[str, Any] = field(default_factory=dict)

    limit_train: int | None = None
    limit_val: int | None = None
    resume: bool = True

    @property
    def run_dir(self) -> Path:
        return Path(self.run_root) / self.exp / f"seed{self.seed}"


# ==========================================================================
class EMA:
    """Exponential moving average of weights.

    Cheap variance reduction, and it matters more than usual here: with ~1k
    training videos the last-epoch weights are noticeably seed-dependent, and
    the experiment grid reports mean +/- std over three seeds. EMA shrinks that spread
    without changing the method.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Swap EMA weights in; returns the originals for restoration."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))
        return backup

    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        sd = model.state_dict()
        for k, v in backup.items():
            sd[k].copy_(v)


class SWAD:
    """Stochastic Weight Averaging, Densely (Cha et al., NeurIPS 2021).

    A plain average of every checkpoint past a start epoch. The domain
    generalisation result it comes from is exactly our problem: flat minima
    transfer across domains better than sharp ones, and the cross-dataset gap
    is a domain-shift gap. Reported as an ablation rather than assumed.
    """

    def __init__(self) -> None:
        self.avg: dict[str, torch.Tensor] | None = None
        self.n = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        sd = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        if self.avg is None:
            self.avg = sd
            self.n = 1
            return
        self.n += 1
        for k in self.avg:
            self.avg[k].add_((sd[k] - self.avg[k]) / self.n)

    def copy_to(self, model: torch.nn.Module) -> None:
        if self.avg is None:
            return
        sd = model.state_dict()
        for k, v in self.avg.items():
            sd[k].copy_(v.to(sd[k].dtype))


# ==========================================================================
def build_loaders(cfg: TrainConfig) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    splits = ["train", *cfg.eval_splits]

    for split in dict.fromkeys(splits):
        dcfg = DataConfig(
            manifest=cfg.manifest,
            cache_root=cfg.cache_root,
            split=split,
            n_frames=cfg.n_frames,
            image_size=cfg.image_size,
            audio_seconds=cfg.audio_seconds,
            max_sync_windows=cfg.max_sync_windows,
            aug=AugConfig(**cfg.aug) if split == "train" else AugConfig(enabled=False),
            audio_aug=AudioAugConfig(**cfg.audio_aug)
            if split == "train"
            else AudioAugConfig(enabled=False),
            sbi=SBIConfig(**cfg.sbi) if split == "train" else SBIConfig(enabled=False),
        )
        try:
            ds = DeepfakeClipDataset(dcfg)
        except RuntimeError as e:
            log.warning("skipping split %r: %s", split, e)
            continue

        limit = cfg.limit_train if split == "train" else cfg.limit_val
        if limit:
            ds.df = ds.df.head(limit).reset_index(drop=True)

        sampler = None
        shuffle = False
        if split == "train":
            if cfg.sampler == "balanced":
                sampler = BalancedMethodSampler(ds.df, seed=cfg.seed)
            elif cfg.sampler == "audio_stratified":
                sampler = AudioStratifiedSampler(ds.df, cfg.batch_size, seed=cfg.seed)
            else:
                shuffle = True

        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            collate_fn=collate,
            worker_init_fn=worker_init_fn,
            pin_memory=False,
            drop_last=split == "train" and len(ds) > cfg.batch_size,
            persistent_workers=cfg.num_workers > 0,
        )
        log.info("%-5s split: %d videos, %d batches", split, len(ds), len(loaders[split]))
    return loaders


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Two param groups: a low LR for pretrained backbones, higher for new heads.

    One global LR either destroys ImageNet features (too high) or leaves the
    randomly-initialised fusion head barely trained (too low).
    """
    backbone, heads = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if "backbone" in name or "encoder" in name else heads).append(p)

    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": cfg.lr, "name": "backbone"})
    if heads:
        groups.append({"params": heads, "lr": cfg.head_lr, "name": "heads"})
    return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()
    }


# ==========================================================================
@torch.no_grad()
def predict_split(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, desc: str = "eval"
) -> dict[str, Any]:
    """Run inference over a split and collect everything preds.csv needs."""
    model.eval()
    acc: dict[str, list[Any]] = {
        k: []
        for k in (
            "video_id",
            "label",
            "dataset",
            "forgery_method",
            "logit",
            "frame_scores",
            "sync_feats",
            "stream_scores",
            "emb",
            "has_audio",
        )
    }

    for batch in tqdm(loader, desc=desc, leave=False, unit="b"):
        b = to_device(batch, device)
        out = model(b)
        logit = out["logit"].float().cpu().numpy()

        acc["video_id"].extend(batch["video_id"])
        acc["label"].extend(b["label"].cpu().numpy().tolist())
        acc["dataset"].extend(batch["dataset"])
        acc["forgery_method"].extend(batch["forgery_method"])
        acc["logit"].extend(logit.tolist())
        acc["has_audio"].extend(b["has_audio"].cpu().numpy().tolist())

        fl = out.get("frame_logits")
        acc["frame_scores"].extend(
            torch.sigmoid(fl).float().cpu().numpy().tolist()
            if fl is not None
            else [[] for _ in logit]
        )
        sq = out.get("sync_seq")
        acc["sync_feats"].extend(
            [{"curve": c} for c in sq.float().cpu().numpy().tolist()]
            if sq is not None
            else [{} for _ in logit]
        )
        sl = out.get("stream_logits") or {}
        if sl:
            per = {k: torch.sigmoid(v).float().cpu().numpy() for k, v in sl.items()}
            acc["stream_scores"].extend(
                [{k: float(per[k][i]) for k in per} for i in range(len(logit))]
            )
        else:
            acc["stream_scores"].extend([{} for _ in logit])
        acc["emb"].extend(
            out["emb"].float().cpu().numpy().tolist() if "emb" in out else [[] for _ in logit]
        )

    return {
        k: (np.asarray(v) if k in ("label", "logit", "has_audio") else v) for k, v in acc.items()
    }


def write_preds(
    pred: dict[str, Any],
    path: Path,
    calibration: Any = None,
    ood_scores: np.ndarray | None = None,
) -> Path:
    """Write a contract-5.4 preds CSV."""
    import pandas as pd

    from ddetect.contracts import PREDS_COLUMNS

    logits = np.asarray(pred["logit"], dtype=float)
    score = 1.0 / (1.0 + np.exp(-logits))
    if calibration is not None:
        cal_p = calibration.probability(logits)
        abstain = calibration.abstains(cal_p)
    else:
        cal_p, abstain = score, np.zeros(len(score), dtype=bool)

    df = pd.DataFrame(
        {
            "video_id": pred["video_id"],
            "label": np.asarray(pred["label"], dtype=int),
            "dataset": pred["dataset"],
            "forgery_method": pred["forgery_method"],
            "compression": "unknown",
            "video_score": score,
            "calibrated_prob": cal_p,
            "frame_scores": [json.dumps([round(x, 5) for x in f]) for f in pred["frame_scores"]],
            "sync_feats": [json.dumps(s, default=float) for s in pred["sync_feats"]],
            "stream_scores": [json.dumps(s, default=float) for s in pred["stream_scores"]],
            "ood_score": ood_scores if ood_scores is not None else energy_score(logits),
            "has_audio": np.asarray(pred["has_audio"], dtype=bool),
            "abstain": abstain,
        }
    )[list(PREDS_COLUMNS)]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ==========================================================================
def train(cfg: TrainConfig) -> dict[str, Any]:
    from ddetect.metrics import auc, evaluate_predictions, threshold_from_val

    setup_logging()
    seed_everything(cfg.seed)
    device = pick_device()
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp_run(run_dir, asdict(cfg))
    jsonl = JsonlWriter(run_dir / "metrics.jsonl")

    from ddetect.tracking import build_tracker

    tracker = build_tracker(cfg.tracker, experiment="avforge")
    tracker.start(
        run_name=f"{cfg.exp}/seed{cfg.seed}",
        config=asdict(cfg),
        tags={"git_sha": stamp["git_sha"], "exp": cfg.exp, "seed": str(cfg.seed)},
    )

    log.info("run %s  seed=%d  device=%s  git=%s", cfg.exp, cfg.seed, device, stamp["git_sha"])

    loaders = build_loaders(cfg)
    if "train" not in loaders:
        raise RuntimeError("no train split available; did preprocessing run?")

    model = build_model(cfg.model).to(device)
    log.info("model: %s", describe(model))

    if cfg.freeze_epochs > 0:
        model.stage("I")
        log.info(
            "stage I: backbone frozen for %d epoch(s), %.1fM trainable",
            cfg.freeze_epochs,
            model.trainable_parameters() / 1e6,
        )

    # ---- domain generalisation ---------------------------
    from ddetect.dg import DomainDiscriminator, GroupDRO, dann_lambda, domain_ids_from_batch

    domain_vocab: dict[str, int] = {}
    discriminator = None
    dro = None
    if cfg.dann:
        # Sized generously: the vocabulary grows lazily as methods are seen,
        # and a discriminator that runs out of classes mid-epoch would silently
        # start mislabelling domains.
        emb_dim = getattr(model.fusion, "out_dim", None) or model.visual.out_dim
        discriminator = DomainDiscriminator(emb_dim, n_domains=32).to(device)
        log.info("DANN enabled: adversarial domain alignment on %r", cfg.domain_key)
    if cfg.groupdro:
        dro = GroupDRO(n_groups=32, eta=cfg.groupdro_eta)
        log.info("GroupDRO enabled: minimising the worst %r group", cfg.domain_key)
    if cfg.irm:
        log.info("IRM enabled (weight %.2f); note it costs a second backward pass", cfg.irm_weight)

    opt = build_optimizer(model, cfg)
    if discriminator is not None:
        # The discriminator is trained normally; only the gradient flowing back
        # into the encoder is reversed.
        opt.add_param_group(
            {
                "params": list(discriminator.parameters()),
                "lr": cfg.head_lr,
                "name": "domain_discriminator",
            }
        )
    loss_fn = CompositeLoss(
        gamma=cfg.focal_gamma,
        label_smoothing=cfg.label_smoothing,
        lambda_mask=cfg.lambda_mask,
        lambda_supcon=cfg.lambda_supcon,
        lambda_sync=cfg.lambda_sync,
    )

    steps_per_epoch = max(len(loaders["train"]) // cfg.accum_steps, 1)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(steps_per_epoch * cfg.warmup_epochs)

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = cfg.amp and supports_amp(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # no-op unless CUDA
    ema = EMA(model, cfg.ema_decay) if cfg.use_ema else None
    swad = SWAD() if cfg.swad else None

    best = {"metric": -np.inf, "epoch": -1}
    history: list[dict[str, Any]] = []
    bad_epochs = 0
    step = 0

    for epoch in range(cfg.epochs):
        if epoch == cfg.freeze_epochs and cfg.freeze_epochs > 0:
            model.stage("II")
            opt = build_optimizer(model, cfg)
            if discriminator is not None:
                opt.add_param_group(
                    {
                        "params": list(discriminator.parameters()),
                        "lr": cfg.head_lr,
                        "name": "domain_discriminator",
                    }
                )
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
            sched.last_epoch = step - 1
            log.info(
                "stage II: backbone unfrozen, %.1fM trainable", model.trainable_parameters() / 1e6
            )

        for s in (loaders["train"].sampler,):
            if hasattr(s, "set_epoch"):
                s.set_epoch(epoch)

        model.train()
        t0 = time.time()
        run_loss, parts_sum, nb = 0.0, {}, 0
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch + 1}/{cfg.epochs}", unit="b")

        for i, batch in enumerate(pbar):
            b = to_device(batch, device)
            # torch.autocast raises on device_type='mps' regardless of
            # enabled=False, so the context has to be skipped entirely rather
            # than disabled. supports_amp() is CUDA-only by design.
            ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype(device))
                if use_amp
                else contextlib.nullcontext()
            )
            with ctx:
                out = model(b)
                loss, parts = loss_fn(out, b)

                if discriminator is not None or dro is not None or cfg.irm:
                    dom_ids, domain_vocab = domain_ids_from_batch(
                        batch, cfg.domain_key, domain_vocab
                    )
                    dom_ids = dom_ids.to(device)

                    if discriminator is not None and dom_ids.numel():
                        lam = dann_lambda(step, total_steps) * cfg.dann_weight
                        from ddetect.dg import dann_loss

                        dl = dann_loss(discriminator, out["emb"], dom_ids, lam)
                        loss = loss + dl
                        parts["dann"] = float(dl.detach())

                    if dro is not None and dom_ids.numel():
                        from ddetect.losses import focal_bce

                        per_sample = focal_bce(
                            out["logit"],
                            b["label"],
                            cfg.focal_gamma,
                            label_smoothing=cfg.label_smoothing,
                            reduction="none",
                        )
                        dro_loss, dro_stats = dro(per_sample, dom_ids)
                        # Replace the mean objective rather than add to it:
                        # GroupDRO IS the objective, not a regulariser.
                        loss = dro_loss + (loss - loss.detach())
                        parts.update({f"dro_{k}": v for k, v in dro_stats.items()})

                    if cfg.irm and dom_ids.numel():
                        from ddetect.dg import irm_loss

                        # irm_loss already folds in the penalty weight and
                        # applies Arjovsky's rescaling, so it is added once.
                        il, istats = irm_loss(out["logit"], b["label"], dom_ids, cfg.irm_weight)
                        loss = loss + il
                        parts.update(istats)

                    # CompositeLoss set parts["total"] before any of the
                    # above was added, so the logged total would understate
                    # what was actually optimised.
                    parts["total"] = float(loss.detach())

                loss = loss / cfg.accum_steps

            if not torch.isfinite(loss):
                # A NaN/Inf loss is a config or data bug, not noise. Skip the
                # step, log it, and let A3's babysitter act on the count rather
                # than silently poisoning every weight with NaN.
                log.warning("non-finite loss at epoch %d step %d; skipping", epoch, i)
                opt.zero_grad(set_to_none=True)
                jsonl.write(event="nonfinite_loss", epoch=epoch, step=i)
                continue

            scaler.scale(loss).backward()

            if (i + 1) % cfg.accum_steps == 0:
                if cfg.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if ema:
                    ema.update(model)

            run_loss += float(loss) * cfg.accum_steps
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v
            nb += 1
            pbar.set_postfix(
                loss=f"{run_loss / max(nb, 1):.4f}", lr=f"{sched.get_last_lr()[0]:.2e}"
            )

        train_loss = run_loss / max(nb, 1)
        rec: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": sched.get_last_lr()[0],
            "secs": round(time.time() - t0, 1),
            **{f"loss_{k}": v / max(nb, 1) for k, v in parts_sum.items()},
        }

        if swad and epoch >= cfg.swad_start_epoch:
            swad.update(model)

        # ---- validation on SOURCE VAL only ----------------------------
        if "val" in loaders:
            backup = ema.copy_to(model) if ema else None
            vp = predict_split(model, loaders["val"], device, desc="val")
            if ema and backup:
                ema.restore(model, backup)

            v_auc = auc(vp["label"], 1 / (1 + np.exp(-vp["logit"])))
            thr = threshold_from_val(vp["label"], 1 / (1 + np.exp(-vp["logit"])))
            vr = evaluate_predictions(
                vp["label"],
                1 / (1 + np.exp(-vp["logit"])),
                threshold=thr,
                threshold_source="EER on source val (in-loop)",
                name="val",
            )
            rec.update(val_auc=v_auc, val_acc=vr.accuracy, val_f1=vr.f1, val_eer=vr.eer)
            log.info(
                "epoch %d: loss %.4f  val AUC %.4f  acc %.4f  (%.0fs)",
                epoch + 1,
                train_loss,
                v_auc,
                vr.accuracy,
                rec["secs"],
            )

            metric = rec.get(cfg.monitor, v_auc)
            if np.isfinite(metric) and metric > best["metric"]:
                best = {"metric": float(metric), "epoch": epoch}
                torch.save(
                    {
                        "model": (ema.shadow if ema else model.state_dict()),
                        "raw_model": model.state_dict(),
                        "cfg": asdict(cfg),
                        "model_cfg": asdict(make_config(cfg.model)),
                        "epoch": epoch,
                        "metric": float(metric),
                        "git_sha": stamp["git_sha"],
                    },
                    run_dir / "ckpt_best.pt",
                )
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    log.info("early stop: no val improvement for %d epochs", cfg.patience)
                    history.append(rec)
                    jsonl.write(**rec)
                    tracker.log_metrics(
                        {k: v for k, v in rec.items() if isinstance(v, (int, float))},
                        step=epoch,
                    )
                    break
        else:
            log.info("epoch %d: loss %.4f (no val split)", epoch + 1, train_loss)

        history.append(rec)
        jsonl.write(**rec)
        tracker.log_metrics(
            {k: v for k, v in rec.items() if isinstance(v, (int, float))}, step=epoch
        )

    # ---- finalise -----------------------------------------------------
    if swad:
        swad.copy_to(model)
        log.info("SWAD: averaged %d checkpoints into the final weights", swad.n)
    elif ema:
        ema.copy_to(model)

    torch.save(
        {
            "model": model.state_dict(),
            "cfg": asdict(cfg),
            "model_cfg": asdict(make_config(cfg.model)),
            "git_sha": stamp["git_sha"],
        },
        run_dir / "ckpt_last.pt",
    )

    # ---- calibration + OOD, fitted on SOURCE VAL ----------------------
    summary: dict[str, Any] = {"best": best, "history": history, "run_dir": str(run_dir)}
    cal = None
    if "val" in loaders:
        vp = predict_split(model, loaders["val"], device, desc="val/final")
        vl = np.asarray(vp["logit"], dtype=float)
        cal = fit_calibration(
            vl,
            np.asarray(vp["label"], dtype=float),
            method=cfg.calibration,
            conformal_alpha=cfg.conformal_alpha,
            fitted_on=f"source val of {Path(cfg.manifest).name} (n={len(vl)})",
            ood_scores=energy_score(vl),
        )
        cal.save(run_dir / "calibration.json")
        write_preds(vp, run_dir / "preds_val.csv", cal)

        embs = np.asarray([e for e in vp["emb"] if e], dtype=float)
        if embs.ndim == 2 and len(embs) > embs.shape[1]:
            MahalanobisOOD().fit(embs).save(run_dir / "ood_mahalanobis.json")
        summary["calibration"] = asdict(cal)

    for split in cfg.eval_splits:
        if split == "val" or split not in loaders:
            continue
        # Reached only when --eval-splits explicitly names a test split.
        log.warning(
            "writing predictions for split %r. If this is a TARGET test set, "
            "this must be a deliberate final evaluation on a frozen config "
            " -- not a number to iterate against.",
            split,
        )
        tp = predict_split(model, loaders[split], device, desc=split)
        write_preds(tp, run_dir / f"preds_{split}.csv", cal)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    tracker.log_summary({"best_metric": best["metric"], "best_epoch": best["epoch"]})
    for name in ("config.yaml", "metrics.jsonl", "summary.json", "calibration.json"):
        tracker.log_artifact(run_dir / name)
    tracker.finish()
    log.info(
        "done. best %s=%.4f at epoch %d -> %s",
        cfg.monitor,
        best["metric"],
        best["epoch"] + 1,
        run_dir,
    )
    return summary


# ==========================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for f, v in asdict(TrainConfig()).items():
        if isinstance(v, bool):
            ap.add_argument(f"--{f.replace('_', '-')}", dest=f, action="store_true", default=None)
            ap.add_argument(
                f"--no-{f.replace('_', '-')}", dest=f, action="store_false", default=None
            )
        elif isinstance(v, (int, float, str)) or v is None:
            t = type(v) if v is not None else str
            ap.add_argument(f"--{f.replace('_', '-')}", dest=f, type=t, default=None)
    ap.add_argument(
        "--eval-splits",
        dest="eval_splits",
        default=None,
        help="comma-separated; 'val' only unless you mean it",
    )
    ap.add_argument("--aug-preset", default=None, choices=["none", "degrade"])
    ap.add_argument("--sbi", dest="sbi_on", action="store_true", help="enable Self-Blended Images")
    a = ap.parse_args(argv)

    cfg = TrainConfig()
    for k, v in vars(a).items():
        if v is None or k in ("eval_splits", "aug_preset", "sbi_on"):
            continue
        setattr(cfg, k, v)
    if a.eval_splits:
        cfg.eval_splits = tuple(s.strip() for s in a.eval_splits.split(","))
    if a.aug_preset == "none":
        cfg.aug = {"enabled": False}
    elif a.aug_preset == "degrade":
        cfg.aug = {"enabled": True}
    if a.sbi_on:
        cfg.sbi = {"enabled": True, "p": 0.5}
        cfg.lambda_mask = 0.3

    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
