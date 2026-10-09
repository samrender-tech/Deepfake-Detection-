"""The agentic layer's guardrails.

Two things are tested, and only these two, because they are the ones whose
failure would damage the project rather than merely inconvenience it:

* the test-set firewall actually refuses a target-split read;
* A7 cannot emit an ungrounded, overclaiming or accusatory statement about a
  real person's video, and the product never depends on the LLM being
  available.

No test here calls an LLM. The agent path is exercised by feeding
``check_grounding`` adversarial text directly, which is both deterministic and
free.
"""

from __future__ import annotations

import pytest

from agents.core.audit import read_all, record
from agents.core.firewall import (
    FirewallViolation,
    ReadRequest,
    check_read,
    eval_final_allowed,
)
from api.explain_agent import check_grounding, explain_result, template_explanation


# ==========================================================================
# 9.1 the test-set firewall
# ==========================================================================
def test_firewall_allows_source_val_reads():
    check_read(ReadRequest(agent="a3_experiment", path="runs/x/preds_val.csv", split="val"))
    check_read(ReadRequest(agent="a4_failure", path="data/manifests/ffpp.parquet", split="train"))


def test_firewall_blocks_test_split_reads():
    with pytest.raises(FirewallViolation, match="may not read"):
        check_read(ReadRequest(agent="a3_experiment", path="runs/x/preds_val.csv", split="test"))


def test_firewall_blocks_test_artifact_paths():
    # Even without a declared split, the filename alone is enough.
    with pytest.raises(FirewallViolation):
        check_read(ReadRequest(agent="a4_failure", path="runs/x/preds_test.csv"))


def test_firewall_blocks_target_domain_datasets():
    with pytest.raises(FirewallViolation, match="target domain"):
        check_read(
            ReadRequest(agent="a3_experiment", path="m.parquet", split="val", dataset="celebdf"),
            source_dataset="ffpp",
        )


def test_firewall_permits_the_paper_agent():
    # A5 may read test results: by then the config is frozen and A5 has no
    # training tools, so it cannot act on what it sees.
    check_read(ReadRequest(agent="a5_paper", path="runs/x/preds_test.csv", split="test"))


def test_firewall_denials_are_audited(tmp_path, monkeypatch):
    import agents.core.firewall as fw

    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr("agents.core.audit.AUDIT_PATH", log_path)

    with pytest.raises(FirewallViolation):
        fw.check_read(ReadRequest(agent="a3_experiment", path="preds_test.csv"))

    entries = read_all(log_path)
    assert any(e["agent"] == "firewall" and e["action"] == "deny" for e in entries), (
        "a blocked read left no audit trail"
    )


def test_eval_final_reports_tree_state():
    ok, msg = eval_final_allowed()
    assert isinstance(ok, bool) and msg
    if not ok:
        assert "uncommitted" in msg or "reproducible" in msg


# ==========================================================================
# 9.8 A7 grounding
# ==========================================================================
@pytest.fixture
def payload():
    return {
        "verdict": "likely_manipulated",
        "calibrated_probability": 0.78,
        "abstained": False,
        "ood_flag": False,
        "has_audio": True,
        "n_faces_found": 8,
        "peak_frame_time_s": 1.6,
        "frame_score_max": 0.91,
        "frame_score_min": 0.22,
        "streams": [
            {"name": "visual", "score": 0.80, "available": True, "reliability": 0.9, "note": ""}
        ],
        "warnings": [],
    }


def test_grounded_text_passes(payload):
    text = (
        "The detector estimates a 78% probability that this video was manipulated. "
        "The strongest single-frame signal was 0.91 at about 1.6s."
    )
    assert check_grounding(text, payload) == []


def test_rejects_invented_numbers(payload):
    text = "The detector is 99.7% sure, with a sync offset of 412 milliseconds."
    problems = check_grounding(text, payload)
    assert any("absent from the payload" in p for p in problems)


def test_rejects_certainty_language(payload):
    for bad in (
        "This video was definitely manipulated.",
        "The analysis proves the footage is fake.",
        "This is conclusively a deepfake.",
    ):
        assert any("certainty" in p for p in check_grounding(bad, payload)), bad


def test_rejects_identity_and_intent_claims(payload):
    for bad in (
        "This person deliberately faked the video.",
        "The person in the clip is a scammer.",
        "She created this to deceive viewers.",
    ):
        problems = check_grounding(bad, payload)
        assert problems, bad
        assert any("identity or intent" in p or "certainty" in p for p in problems), bad


def test_abstained_result_must_lead_with_uncertainty(payload):
    payload = {**payload, "abstained": True, "verdict": "inconclusive"}
    buried = (
        "The video shows signs of manipulation around 1.6s. "
        "The system is not confident enough to call it."
    )
    assert any("lead with the uncertainty" in p for p in check_grounding(buried, payload))

    leading = (
        "The system is not confident enough to give a verdict. "
        "Its estimate of 0.78 falls inside the inconclusive band."
    )
    assert check_grounding(leading, payload) == []


def test_ood_result_must_lead_with_uncertainty(payload):
    payload = {**payload, "ood_flag": True}
    bad = "This video was manipulated with 78% probability."
    assert any("lead with the uncertainty" in p for p in check_grounding(bad, payload))


def test_rejects_empty_and_overlong(payload):
    assert check_grounding("ok", payload)
    assert any("too long" in p for p in check_grounding("word " * 200, payload))


# ==========================================================================
# the template fallback -- the thing the product actually ships
# ==========================================================================
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"verdict": "likely_authentic", "calibrated_probability": 0.12},
        {"abstained": True, "verdict": "inconclusive"},
        {"ood_flag": True, "verdict": "inconclusive"},
        {"has_audio": False},
        {"n_faces_found": 0, "abstained": True},
        {"warnings": ["no usable audio stream; audio+sync streams masked off"]},
        {
            "streams": [
                {
                    "name": "sync",
                    "score": None,
                    "available": False,
                    "reliability": 0.0,
                    "note": "mouth region not reliably visible",
                }
            ]
        },
    ],
)
def test_template_always_passes_its_own_guard(payload, overrides):
    """The shipped fallback must satisfy the same rules the agent is held to.

    If the template cannot pass, the product has no safe path at all.
    """
    p = {**payload, **overrides}
    text = template_explanation(p)
    assert text.strip()
    problems = check_grounding(text, p)
    assert problems == [], f"template failed its own guard for {overrides}: {problems}\n{text}"


def test_explain_result_never_raises_without_a_key(monkeypatch, payload):
    monkeypatch.setenv("DDETECT_DISABLE_AGENT", "1")
    text, source = explain_result(payload)
    assert source == "template"
    assert text.strip()


def test_explain_result_falls_back_when_the_agent_is_unavailable(monkeypatch, payload):
    monkeypatch.delenv("DDETECT_DISABLE_AGENT", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    text, source = explain_result(payload)
    # No key -> template, and no exception. The product never blocks on the LLM.
    assert source == "template" and text.strip()


def test_ungrounded_agent_output_is_discarded(monkeypatch, payload, tmp_path):
    """Simulate a hallucinating agent and assert the guard replaces it."""
    monkeypatch.setattr("agents.core.audit.AUDIT_PATH", tmp_path / "audit.jsonl")
    monkeypatch.delenv("DDETECT_DISABLE_AGENT", raising=False)
    monkeypatch.setattr(
        "api.explain_agent._call_claude",
        lambda p, timeout=20.0: ("This definitely proves the person faked it, 99.9% certain.", 100),
    )
    text, source = explain_result(payload)
    assert source == "template", "an overclaiming explanation reached the user"
    assert "definitely" not in text

    entries = read_all(tmp_path / "audit.jsonl")
    assert any(e["action"] == "rejected" for e in entries)


def test_grounded_agent_output_is_accepted(monkeypatch, payload, tmp_path):
    monkeypatch.setattr("agents.core.audit.AUDIT_PATH", tmp_path / "audit.jsonl")
    monkeypatch.delenv("DDETECT_DISABLE_AGENT", raising=False)
    good = (
        "The detector estimates a 78% probability of manipulation, with the "
        "strongest single-frame signal of 0.91 at about 1.6s."
    )
    monkeypatch.setattr("api.explain_agent._call_claude", lambda p, timeout=20.0: (good, 100))
    text, source = explain_result(payload)
    assert source == "agent" and text == good
    assert any(e["action"] == "accepted" for e in read_all(tmp_path / "audit.jsonl"))


# ==========================================================================
# audit log
# ==========================================================================
def test_audit_record_stores_hashes_not_payloads(tmp_path):
    log_path = tmp_path / "a.jsonl"
    secret = {"frame_scores": [0.1, 0.2], "video_id": "sensitive-name"}
    rec = record("t", "act", inputs=secret, output="text", path=log_path)

    assert rec["inputs_hash"] and len(rec["inputs_hash"]) == 16
    raw = log_path.read_text()
    assert "sensitive-name" not in raw, (
        "the audit log stored the raw payload; it must store only a hash"
    )


# ==========================================================================
# the research agents
# ==========================================================================
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agents.core.budget import BudgetExceeded, budget_for  # noqa: E402
from agents.core.cache import ResponseCache  # noqa: E402
from agents.core.tools import ToolDenied, ToolRegistry  # noqa: E402
from agents.registry import AGENTS, PERMISSIONS, build  # noqa: E402


def _manifest_rows(**over):
    base = dict(
        path="/x/v.mp4",
        label=0,
        dataset="ffpp",
        forgery_method="real",
        split="train",
        has_audio=False,
        source_identity="p1",
        compression="c23",
        fps=25.0,
        n_frames=100,
        duration_s=4.0,
        sha256="0" * 64,
    )
    return {**base, **over}


@pytest.fixture
def clean_manifest(tmp_path):
    from ddetect.data.manifest_io import write_manifest

    rows = [
        _manifest_rows(video_id="r1", split="train", source_identity="p1"),
        _manifest_rows(video_id="r2", split="val", source_identity="p2"),
        _manifest_rows(video_id="r3", split="test", source_identity="p3"),
        _manifest_rows(
            video_id="f1", split="train", source_identity="p4", label=1, forgery_method="Deepfakes"
        ),
        _manifest_rows(
            video_id="f2", split="val", source_identity="p5", label=1, forgery_method="Deepfakes"
        ),
    ]
    p = tmp_path / "clean.parquet"
    write_manifest(rows, p)
    return p


# ---- budget ---------------------------------------------------------------
def test_budget_allows_exactly_its_call_allocation():
    b = budget_for("a4_failure", max_calls=3)
    for _ in range(3):
        b.spend_call()
    with pytest.raises(BudgetExceeded):
        b.spend_call()


def test_only_the_experiment_agent_may_spend_gpu_hours():
    # A3 schedules training; nothing else should be able to burn the project's
    # GPU allocation.
    with pytest.raises(BudgetExceeded, match="no GPU-hour allocation"):
        budget_for("a4_failure").spend_gpu_hours(1.0)
    b = budget_for("a3_experiment")
    b.spend_gpu_hours(1.0)
    assert b.gpu_hours == 1.0
    with pytest.raises(BudgetExceeded):
        b.spend_gpu_hours(10_000)


# ---- cache ----------------------------------------------------------------
def test_cache_makes_a_repeated_tool_call_free(tmp_path):
    calls = {"n": 0}

    def fn(x):
        calls["n"] += 1
        return {"v": x * 2}

    c = ResponseCache(tmp_path)
    assert c.memoize("v1", "t", {"x": 21}, lambda: fn(21)) == {"v": 42}
    assert c.memoize("v1", "t", {"x": 21}, lambda: fn(21)) == {"v": 42}
    assert calls["n"] == 1, "the cached call re-invoked the function"
    # A different prompt version must miss: the answer may legitimately change.
    c.memoize("v2", "t", {"x": 21}, lambda: fn(21))
    assert calls["n"] == 2


def test_cache_can_be_disabled(tmp_path):
    calls = {"n": 0}
    c = ResponseCache(tmp_path, enabled=False)
    for _ in range(2):
        c.memoize("v1", "t", {}, lambda: calls.__setitem__("n", calls["n"] + 1))
    assert calls["n"] == 2


# ---- tool registry --------------------------------------------------------
def test_registry_denies_an_unregistered_tool():
    r = ToolRegistry(agent="a4_failure", budget=budget_for("a4_failure"))
    with pytest.raises(ToolDenied, match="may not use"):
        r.call("rm_rf")


def test_registry_applies_the_firewall_to_data_tools(tmp_path):
    r = ToolRegistry(
        agent="a4_failure", budget=budget_for("a4_failure"), cache=ResponseCache(tmp_path)
    )
    r.add("read", lambda path: {"p": path}, "reads", reads_data=True)
    r.call(path="runs/x/preds_val.csv", name="read")  # allowed
    with pytest.raises(FirewallViolation):
        r.call("read", path="runs/x/preds_test.csv")


def test_registry_blocks_unapproved_repo_writes(tmp_path):
    r = ToolRegistry(agent="a5_paper", budget=budget_for("a5_paper"), cache=ResponseCache(tmp_path))
    r.add("w", lambda: "done", "writes", writes=True, pure=False)
    with pytest.raises(ToolDenied, match="without approval"):
        r.call("w")
    r.approved_writes = True
    assert r.call("w") == "done"


# ---- the roster -----------------------------------------------------------
def test_every_agent_builds():
    for name in AGENTS:
        assert build(name) is not None


def test_only_frozen_config_agents_may_read_test_results():
    """A5 and A6 may, and only because they have no training tool."""
    allowed = {n for n, p in PERMISSIONS.items() if p.get("test_results")}
    assert allowed == {"a5_paper", "a6_defence"}
    for n in allowed:
        assert "no training tool" in PERMISSIONS[n].get("note", "")


def test_agents_that_propose_changes_cannot_read_test_results():
    for n in ("a3_experiment", "a4_failure", "a8_redteam"):
        assert not PERMISSIONS[n].get("test_results")


# ---- A2 data curation -----------------------------------------------------
def test_a2_passes_a_clean_manifest(clean_manifest):
    res = build("a2_data").run(manifest=clean_manifest)
    assert res.ok
    assert not res.blockers


def test_a2_blocks_on_identity_leakage(tmp_path):
    from ddetect.data.manifest_io import write_manifest

    rows = [
        _manifest_rows(video_id="a", split="train", source_identity="shared"),
        _manifest_rows(video_id="b", split="test", source_identity="shared"),
        _manifest_rows(
            video_id="c", split="val", source_identity="p2", label=1, forgery_method="Deepfakes"
        ),
    ]
    p = tmp_path / "leaky.parquet"
    write_manifest(rows, p)

    res = build("a2_data").run(manifest=p)
    assert not res.ok, "a leaky manifest was not blocked"
    assert any("identity overlap" in f.title for f in res.blockers)


def test_a2_reports_the_audio_evidence_table(clean_manifest):
    """The empirical basis for the no-audio finding -- that FF++ carries no audio."""
    res = build("a2_data").run(manifest=clean_manifest)
    assert "audio_table" in res.data
    assert res.data["audio_table"]["ffpp"]["with_audio"] == 0
    assert any("no usable audio" in f.title for f in res.findings)


# ---- A4 failure diagnosis -------------------------------------------------
def _preds_csv(path, n=40, seed=0):
    from ddetect.contracts import PREDS_COLUMNS

    rng = np.random.default_rng(seed)
    y = (rng.uniform(0, 1, n) < 0.5).astype(int)
    s = np.clip(np.where(y == 1, rng.normal(0.7, 0.2, n), rng.normal(0.3, 0.2, n)), 0, 1)
    df = pd.DataFrame({c: [""] * n for c in PREDS_COLUMNS})
    df["video_id"] = [f"v{i}" for i in range(n)]
    df["label"] = y
    df["dataset"] = "ffpp"
    df["forgery_method"] = np.where(y == 1, "Deepfakes", "real")
    df["compression"] = rng.choice(["c23", "c40"], n)
    df["video_score"] = s
    df["calibrated_prob"] = s
    df["has_audio"] = False
    df["abstain"] = False
    df.to_csv(path, index=False)
    return path


def test_a4_refuses_a_test_split_file(tmp_path):
    p = _preds_csv(tmp_path / "preds_test.csv")
    res = build("a4_failure").run(preds=p)
    assert not res.ok
    assert "FirewallViolation" in res.summary


def test_a4_diagnoses_val_predictions(tmp_path):
    p = _preds_csv(tmp_path / "preds_val.csv")
    res = build("a4_failure").run(preds=p)
    assert res.ok
    assert "overall_accuracy" in res.data
    assert res.data["n"] == 40
    assert res.data["worst_videos"], "no failure cases surfaced"


def test_a4_finds_a_planted_weak_slice(tmp_path):
    """A slice the model fails on must be surfaced with a concrete action."""
    from ddetect.contracts import PREDS_COLUMNS

    n = 80
    rng = np.random.default_rng(1)
    y = np.tile([0, 1], n // 2)
    # compression must be INDEPENDENT of label, or each slice is single-class
    # and "accuracy on this slice" measures the class balance, not the model.
    comp = np.repeat(["c23", "c40"], n // 2)
    # c23 is cleanly separable; c40 is pure noise.
    s = np.where(comp == "c23", np.where(y == 1, 0.9, 0.1), rng.uniform(0.45, 0.55, n))

    df = pd.DataFrame({c: [""] * n for c in PREDS_COLUMNS})
    df["video_id"] = [f"v{i}" for i in range(n)]
    df["label"] = y
    df["dataset"] = "ffpp"
    df["forgery_method"] = np.where(y == 1, "Deepfakes", "real")
    df["compression"] = comp
    df["video_score"] = s
    df["calibrated_prob"] = s
    df["has_audio"] = False
    df["abstain"] = False
    p = tmp_path / "preds_val.csv"
    df.to_csv(p, index=False)

    res = build("a4_failure").run(preds=p)
    hit = [f for f in res.findings if "compression" in f.title]
    assert hit, f"the planted compression weakness was not found: {[f.title for f in res.findings]}"
    assert hit[0].suggested_action, "a weak slice was reported with no action"


# ---- A1 literature --------------------------------------------------------
def test_a1_blocks_citations_without_identifiers(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(
        "@inproceedings{good, title={T}, author={A}, booktitle={B}, "
        "year={2022}, eprint={2204.08376}}\n"
        "@inproceedings{fabricated, title={Plausible Sounding Paper}, "
        "author={Smith, J.}, booktitle={Proc. CVPR}, year={2023}}\n"
    )
    res = build("a1_literature").run(bib=bib)
    assert not res.ok
    assert "fabricated" in res.data["unverified"]
    assert "good" not in res.data["unverified"]


def test_a1_parses_the_real_bibliography():
    res = build("a1_literature").run(bib="paper/refs.bib")
    assert res.data["n_entries"] >= 10
    assert res.data["related_work_table"]


# ---- A5 paper -------------------------------------------------------------
def test_a5_flags_a_claim_that_disagrees_with_results(tmp_path):
    (tmp_path / "r.csv").write_text("exp,auc_mean\nbaseline_a,0.9512\n")
    claims = tmp_path / "claims.csv"
    claims.write_text(
        "section,claim_id,sentence_fragment,results_file,column,row_selector,"
        "expected_value,status\n"
        f"results,c1,auc of X,{tmp_path / 'r.csv'},auc_mean,exp=baseline_a,0.9900,unverified\n"
    )
    res = build("a5_paper").run(claims=claims, tex=tmp_path / "none.tex")
    assert not res.ok
    assert res.data["mismatched"], "a stale number was not caught"


def test_a5_verifies_a_correct_claim(tmp_path):
    (tmp_path / "r.csv").write_text("exp,auc_mean\nbaseline_a,0.9512\n")
    claims = tmp_path / "claims.csv"
    claims.write_text(
        "section,claim_id,sentence_fragment,results_file,column,row_selector,"
        "expected_value,status\n"
        f"results,c1,auc of X,{tmp_path / 'r.csv'},auc_mean,exp=baseline_a,0.9512,verified\n"
    )
    res = build("a5_paper").run(claims=claims, tex=tmp_path / "none.tex")
    assert res.ok and res.data["verified"] == ["c1"]


def test_a5_flags_overclaiming_language(tmp_path):
    tex = tmp_path / "main.tex"
    tex.write_text("Our method proves that it solves deepfake detection.\n")
    claims = tmp_path / "claims.csv"
    claims.write_text(
        "section,claim_id,sentence_fragment,results_file,column,row_selector,expected_value,status\n"
    )
    res = build("a5_paper").run(claims=claims, tex=tex)
    assert any("claim more than the results support" in f.title for f in res.findings)


def test_the_real_claims_table_parses():
    """A commented-out CSV header makes DictReader eat the first data row."""
    from agents.a5_paper import _read_claims

    rows = _read_claims("paper/claims.csv")
    assert len(rows) >= 7
    assert all("claim_id" in r and r["claim_id"] for r in rows)


# ---- A6 defence -----------------------------------------------------------
def test_a6_separates_answerable_questions_from_gaps(tmp_path):
    res = build("a6_defence").run(results_dir=tmp_path)
    assert res.data["answerable"], "no question answerable from the design alone"
    assert res.data["gaps"], "questions needing results were not flagged as gaps"
    # It must not invent an answer it cannot ground.
    for g in res.data["gaps"]:
        assert g["missing_results"]


def test_a6_covers_the_anticipated_questions():
    res = build("a6_defence").run(results_dir="results")
    asked = " ".join(q["question"] for q in res.data["answerable"] + res.data["gaps"]).lower()
    for topic in ("train on all the datasets", "already solved", "compute"):
        assert topic in asked, f"anticipated question missing: {topic}"


# ---- A8 red team ----------------------------------------------------------
def test_a8_proposes_recipes_without_any_generative_step():
    res = build("a8_redteam").run()
    assert res.ok and res.findings
    import inspect

    import agents.a8_redteam as mod

    src = inspect.getsource(mod)
    for banned in ("diffusers", "insightface", "wav2lip", "StyleGAN", "generate_face"):
        assert banned not in src, f"a8 references a generative tool: {banned}"


def test_a8_flags_the_audio_swap_ambiguity():
    """Real audio on a different real video: true positive, or false accusation?"""
    res = build("a8_redteam").run(kinds=("audio",))
    swap = [f for f in res.findings if "audio_swap" in f.title]
    assert swap and swap[0].severity == "blocker"
    assert "authentic" in swap[0].detail


def test_a8_includes_an_invariance_check():
    """hflip changes no evidence; a score swing there is a model defect."""
    res = build("a8_redteam").run(kinds=("geometry",))
    assert any("hflip" in f.title for f in res.findings)


# ---- no agent calls a model in CI ----------------------------------------
def test_agents_run_without_an_api_key(monkeypatch, clean_manifest):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DDETECT_DISABLE_AGENT", "1")
    res = build("a2_data").run(manifest=clean_manifest)
    assert res.narrative is None
    assert res.narrative_source == "none"
    assert res.summary, "the deterministic analysis did not run without a model"
