#!/usr/bin/env bash
# Run the WK17a benchmark on two commits and produce a paired improvement report.
#
# Each side runs in its own throwaway git worktree, so it uses that commit's own checked-out src +
# benchmarks (necessary when the two commits differ in instrumentation columns). The working tree
# and current branch are left untouched.
#
# usage:
#   run_pair.sh --base <commit> --candidate <commit> --out <dir> \
#     [--samples 1:100] [--tightening lp] [--main-time-limit 120] [--norm-order Inf] \
#     [--base-objective feasibility|closest] \
#     [--candidate-objective feasibility|closest] \
#     [--base-label ...] [--candidate-label ...]
#
# Produces <out>/base, <out>/candidate (benchmark outputs) and <out>/analysis (plots + tables).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

BASE=""; CAND=""; OUT=""
SAMPLES="1:100"; TIGHTENING="lp"; MAIN_TL="120"; NORM="Inf"
BASE_OBJECTIVE=""; CAND_OBJECTIVE=""; BASE_LABEL=""; CAND_LABEL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --base) BASE="$2"; shift 2;;
        --candidate) CAND="$2"; shift 2;;
        --out) OUT="$2"; shift 2;;
        --samples) SAMPLES="$2"; shift 2;;
        --tightening) TIGHTENING="$2"; shift 2;;
        --main-time-limit) MAIN_TL="$2"; shift 2;;
        --norm-order) NORM="$2"; shift 2;;
        --base-objective) BASE_OBJECTIVE="$2"; shift 2;;
        --candidate-objective) CAND_OBJECTIVE="$2"; shift 2;;
        --base-label) BASE_LABEL="$2"; shift 2;;
        --candidate-label) CAND_LABEL="$2"; shift 2;;
        *) echo "unknown argument: $1" >&2; exit 2;;
    esac
done
if [ -z "$BASE" ] || [ -z "$CAND" ] || [ -z "$OUT" ]; then
    echo "usage: $0 --base <commit> --candidate <commit> --out <dir> [--samples 1:100]" \
         "[--tightening lp] [--main-time-limit 120] [--norm-order Inf]" \
         "[--base-objective OBJECTIVE] [--candidate-objective OBJECTIVE]" >&2
    exit 2
fi

validate_objective() {
    case "$1" in
        ""|closest|feasibility) ;;
        *) echo "invalid benchmark objective: $1" >&2; exit 2;;
    esac
}
validate_objective "$BASE_OBJECTIVE"
validate_objective "$CAND_OBJECTIVE"

BASE_LABEL="${BASE_LABEL:-base $BASE}"
CAND_LABEL="${CAND_LABEL:-candidate $CAND}"

mkdir -p "$OUT"
WORKTREES=()
cleanup() {
    for w in "${WORKTREES[@]:-}"; do
        [ -n "$w" ] && git -C "$REPO" worktree remove --force "$w" 2>/dev/null || true
    done
}
trap cleanup EXIT

run_side() {
    local name="$1" sha="$2" objective="$3"
    local objective_args=()
    if [ -n "$objective" ]; then
        objective_args=(--objective "$objective")
    fi
    local wt; wt="$(mktemp -d)"; WORKTREES+=("$wt")
    echo "[$name @ $sha] add worktree + instantiate benchmarks env"
    git -C "$REPO" worktree add -q --detach "$wt" "$sha"
    ( cd "$wt" && julia --project=benchmarks -e 'using Pkg; Pkg.instantiate()' )
    echo "[$name @ $sha] run WK17a benchmark (samples $SAMPLES, tightening $TIGHTENING, objective ${objective:-commit-default})"
    ( cd "$wt" && julia --project=benchmarks benchmarks/benchmark_wk17a_first100.jl \
        --out "$OUT/$name" --samples "$SAMPLES" --tightening "$TIGHTENING" \
        --main-time-limit "$MAIN_TL" --norm-order "$NORM" --log-level warn \
        "${objective_args[@]}" )
}

run_side base "$BASE" "$BASE_OBJECTIVE"
run_side candidate "$CAND" "$CAND_OBJECTIVE"

echo "analyzing pair -> $OUT/analysis"
( cd "$SCRIPT_DIR/analysis" && uv run analyze_pair.py \
    --baseline "$OUT/base" --candidate "$OUT/candidate" --out "$OUT/analysis" \
    --baseline-label "$BASE_LABEL" --candidate-label "$CAND_LABEL" )

echo "done. review $OUT/analysis, then publish with:"
echo "  benchmarks/publish_report.sh $OUT/analysis <YYYY-MM-DD-slug>"
