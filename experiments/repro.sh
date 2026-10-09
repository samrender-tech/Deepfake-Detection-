#!/usr/bin/env bash
# Reproduce a table or figure from scratch.
#
#   ./experiments/repro.sh list
#   ./experiments/repro.sh table_main
#
# The reproducibility check's acceptance test: a clean clone reproduces a headline table from
# this script alone. If a command here does not work on a fresh checkout, the
# reproducibility claim in the paper is false.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
MANIFEST="${MANIFEST:-data/manifests/ffpp.parquet}"
CACHE="${CACHE:-processed}"

die() { echo "error: $*" >&2; exit 1; }

need_manifest() {
  [ -f "$MANIFEST" ] || die "missing $MANIFEST
  Build it first:
    $PY -m ddetect.data.build_manifest --dataset ffpp --root <FF++ root> --audit
    $PY -m ddetect.data.run_preprocess --manifest $MANIFEST --workers 8"
}

run_exp() {  # run_exp <exp> <model> <seeds...>
  local exp="$1" model="$2"; shift 2
  for seed in "$@"; do
    if [ -f "runs/$exp/seed$seed/preds_val.csv" ]; then
      echo "  [skip] $exp/seed$seed already exists"
      continue
    fi
    echo "  [run ] $exp/seed$seed"
    $PY -m ddetect.train --exp "$exp" --model "$model" --seed "$seed" \
        --manifest "$MANIFEST" --cache-root "$CACHE"
    $PY -m ddetect.evaluate --run "runs/$exp/seed$seed" --bootstrap 2000
  done
}

case "${1:-list}" in
  list)
    cat <<TXT
Reproducible targets:

  table_main      Table I -- detection performance across all experiments
  gap             The cross-dataset gap (the headline result)
  baselines       Baseline A and B only (the fastest meaningful reproduction)
  ablations       The ablation grid
  all             Everything. ~198 GPU-hours; see `make matrix` first.

Environment:
  PY=$PY  MANIFEST=$MANIFEST  CACHE=$CACHE
TXT
    ;;

  baselines)
    need_manifest
    echo "Reproducing Baseline A and B (3 seeds each)..."
    run_exp baseline_a baseline 0 1 2
    run_exp baseline_b baseline 0 1 2
    $PY -m experiments.aggregate
    echo "-> results/table_main.tex, results/cross_dataset_gap.csv"
    ;;

  table_main)
    need_manifest
    echo "Reproducing Table I (headline experiments, 3 seeds)..."
    run_exp baseline_a baseline   0 1 2
    run_exp baseline_b baseline   0 1 2
    run_exp v_full     v_full     0 1 2
    run_exp av_proposed avforge   0 1 2
    run_exp full_system avforge   0 1 2
    $PY -m experiments.aggregate
    echo "-> results/table_main.tex"
    ;;

  gap)
    # Reads existing runs; does not retrain.
    $PY -m experiments.aggregate
    [ -f results/cross_dataset_gap.csv ] \
      && cat results/cross_dataset_gap.csv \
      || echo "no cross-dataset results yet -- target test sets are evaluated via 'make eval-final'"
    ;;

  ablations)
    need_manifest
    $PY -m experiments.run_matrix --group ablation
    $PY -m experiments.aggregate
    ;;

  all)
    need_manifest
    $PY -m experiments.run_matrix
    $PY -m experiments.aggregate
    ;;

  *)
    die "unknown target '$1'. Run '$0 list'."
    ;;
esac
