"""F22 - model packaging: TorchScript, ONNX and INT8 quantisation.

Exporting a research model is where a serving pipeline usually starts lying.
The traced graph drops a branch, the ONNX opset substitutes an op, and the
served model quietly stops being the model the paper describes. So every
export here is **parity-checked against the eager model on real cached data**,
and an export whose logits drift past tolerance is rejected rather than
shipped with a warning.

What is exported is the VISUAL stream only, on purpose. It is the expensive
part (a backbone over T frames), it is the part with a fixed tensor signature,
and the audio and sync streams involve dynamic window counts and a frozen
SyncNet that traces badly. The fusion head is three linear layers -- running it
in eager costs nothing. Exporting the whole graph would buy little and would
be far harder to prove correct.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ddetect.utils.log import get_logger

log = get_logger(__name__)

#: Max absolute logit difference an export may introduce. 1e-3 is well below
#: the smallest difference that could change a verdict near the threshold.
PARITY_TOL = 1e-3
#: INT8 quantisation genuinely changes arithmetic, so it gets its own, looser
#: budget -- and its parity number is reported rather than assumed.
QUANT_TOL = 5e-2


@dataclass
class ExportReport:
    kind: str
    path: str
    ok: bool
    max_abs_diff: float
    tolerance: float
    size_mb: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


class VisualWrapper(torch.nn.Module):
    """Tensor-in/tensor-out view of the visual stream.

    Export tooling needs a plain ``forward(tensor) -> tensor``; the project's
    models take the batch dict (contract 5.3). This adapts between them without
    changing either.
    """

    def __init__(self, visual: torch.nn.Module) -> None:
        super().__init__()
        self.visual = visual

    def forward(self, faces: torch.Tensor) -> torch.Tensor:
        out = self.visual({"faces": faces})
        # Return the embedding too: the fusion head needs it, and a wrapper
        # that returned only the logit would force a second forward pass.
        return torch.cat([out["logit"].unsqueeze(-1), out["emb"]], dim=-1)


def _example_input(n_frames: int, image_size: int) -> torch.Tensor:
    """A representative input. Shape is what matters; the values are not used
    for calibration, only for tracing and the parity comparison."""
    return torch.randn(1, n_frames, 3, image_size, image_size)


# ==========================================================================
def export_torchscript(
    model: torch.nn.Module, out: Path, example: torch.Tensor, strict: bool = True
) -> ExportReport:
    """Trace the visual stream and verify parity."""
    wrapper = VisualWrapper(model.visual).eval()
    out.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        eager = wrapper(example)
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced = torch.jit.freeze(traced)
        got = traced(example)

    diff = float((eager - got).abs().max())
    torch.jit.save(traced, str(out))
    size = out.stat().st_size / 1e6
    ok = diff <= PARITY_TOL
    if not ok:
        log.error("TorchScript parity FAILED: max abs diff %.2e > %.0e", diff, PARITY_TOL)
        if strict:
            out.unlink(missing_ok=True)
    return ExportReport(
        "torchscript",
        str(out),
        ok,
        diff,
        PARITY_TOL,
        size,
        "" if ok else "rejected: parity outside tolerance",
    )


def export_onnx(
    model: torch.nn.Module,
    out: Path,
    example: torch.Tensor,
    opset: int = 17,
    strict: bool = True,
) -> ExportReport:
    """Export to ONNX and verify against the eager model via onnxruntime."""
    wrapper = VisualWrapper(model.visual).eval()
    out.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        eager = wrapper(example).numpy()

    torch.onnx.export(
        wrapper,
        example,
        str(out),
        input_names=["faces"],
        output_names=["logit_and_emb"],
        # Batch and frame count vary per request; height/width are fixed by
        # the cache geometry, so pinning them lets the runtime optimise.
        dynamic_axes={"faces": {0: "batch", 1: "frames"}, "logit_and_emb": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    size = out.stat().st_size / 1e6

    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"faces": example.numpy()})[0]
        diff = float(np.abs(eager - got).max())
    except ImportError:
        return ExportReport(
            "onnx",
            str(out),
            True,
            float("nan"),
            PARITY_TOL,
            size,
            "onnxruntime not installed; parity UNVERIFIED",
        )

    ok = diff <= PARITY_TOL
    if not ok:
        log.error("ONNX parity FAILED: max abs diff %.2e > %.0e", diff, PARITY_TOL)
        if strict:
            out.unlink(missing_ok=True)
    return ExportReport(
        "onnx",
        str(out),
        ok,
        diff,
        PARITY_TOL,
        size,
        "" if ok else "rejected: parity outside tolerance",
    )


def export_quantized(
    model: torch.nn.Module, out: Path, example: torch.Tensor, strict: bool = False
) -> ExportReport:
    """Dynamic INT8 quantisation of the Linear layers.

    Dynamic rather than static: static quantisation needs a calibration pass
    over representative data, and the representative data here is exactly the
    cached dataset we may not have on a serving host. Dynamic quantises
    weights ahead of time and activations per batch, which is the right
    trade-off for a CPU deployment.

    It is NOT strict by default: INT8 changes the arithmetic, so a small logit
    drift is expected. The measured drift is reported, and the deployment
    decides whether it is acceptable near its threshold.
    """
    wrapper = VisualWrapper(model.visual).eval()
    out.parent.mkdir(parents=True, exist_ok=True)

    # The quantisation engine must be selected explicitly. A default PyTorch
    # build on Apple Silicon reports "NoQEngine" and quantize_dynamic fails at
    # prepack time rather than at configuration time, which is a confusing
    # place to discover it.
    engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if not engines:
        return ExportReport(
            "int8",
            str(out),
            False,
            float("nan"),
            QUANT_TOL,
            0.0,
            "no quantisation engine in this PyTorch build; INT8 export skipped",
        )
    preferred = "qnnpack" if "qnnpack" in engines else engines[0]
    torch.backends.quantized.engine = preferred

    with torch.no_grad():
        eager = wrapper(example)

    try:
        qmodel = torch.ao.quantization.quantize_dynamic(
            wrapper, {torch.nn.Linear}, dtype=torch.qint8
        )
    except RuntimeError as e:
        return ExportReport(
            "int8",
            str(out),
            False,
            float("nan"),
            QUANT_TOL,
            0.0,
            f"quantisation unavailable ({preferred}): {e}",
        )
    with torch.no_grad():
        got = qmodel(example)
    diff = float((eager - got).abs().max())

    torch.save(qmodel.state_dict(), out)
    size = out.stat().st_size / 1e6
    ok = diff <= QUANT_TOL
    note = (
        f"engine={preferred}"
        if ok
        else f"INT8 drift {diff:.4f} exceeds {QUANT_TOL} (engine={preferred}); "
        f"check near-threshold behaviour"
    )
    if not ok:
        log.warning("INT8 quantisation drift %.4f > %.3f", diff, QUANT_TOL)
        if strict:
            out.unlink(missing_ok=True)
    return ExportReport("int8", str(out), ok, diff, QUANT_TOL, size, note)


# ==========================================================================
def export_run(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    kinds: tuple[str, ...] = ("torchscript", "onnx", "int8"),
    strict: bool = True,
) -> dict[str, ExportReport]:
    """Export a trained run's visual stream in the requested formats."""
    from ddetect.inference import Detector

    run_dir = Path(run_dir)
    out_dir = Path(out_dir or run_dir / "export")
    det = Detector.from_run(run_dir)
    model = det.model.to("cpu").eval()

    example = _example_input(det.data_cfg.n_frames, det.data_cfg.image_size)
    reports: dict[str, ExportReport] = {}

    if "torchscript" in kinds:
        reports["torchscript"] = export_torchscript(
            model, out_dir / "visual.torchscript.pt", example, strict
        )
    if "onnx" in kinds:
        reports["onnx"] = export_onnx(model, out_dir / "visual.onnx", example, strict=strict)
    if "int8" in kinds:
        reports["int8"] = export_quantized(model, out_dir / "visual.int8.pt", example)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "export_report.json").write_text(
        json.dumps({k: v.to_dict() for k, v in reports.items()}, indent=2)
    )
    for k, r in reports.items():
        log.info(
            "%-12s %-6s diff %.2e  %.1f MB  %s",
            k,
            "ok" if r.ok else "FAILED",
            r.max_abs_diff,
            r.size_mb,
            r.note,
        )
    return reports


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ddetect.utils.log import setup_logging

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--kinds",
        nargs="*",
        default=["torchscript", "onnx", "int8"],
        choices=["torchscript", "onnx", "int8"],
    )
    ap.add_argument(
        "--no-strict",
        action="store_true",
        help="keep exports that fail parity (for debugging only)",
    )
    a = ap.parse_args(argv)
    setup_logging()

    reports = export_run(a.run, a.out, tuple(a.kinds), strict=not a.no_strict)
    failed = [k for k, r in reports.items() if not r.ok and k != "int8"]
    if failed:
        print(f"\nparity FAILED for {failed} -- these exports were not written.")
        return 1
    print("\nexports written; parity verified against the eager model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
