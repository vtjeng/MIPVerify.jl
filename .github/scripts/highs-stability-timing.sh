#!/usr/bin/env bash
set -euo pipefail

work_root="$(mktemp -d)"
base_commit="$(git rev-parse HEAD^)"
for pair in baseline candidate; do
    git worktree add --detach "$work_root/$pair" "$base_commit"
done
export MIPVERIFY_RUN_LARGE_DATASET_TESTS=true
export JULIA_PKG_PRECOMPILE_AUTO=0

# Force an exact solver pair in both participating workspace projects.
prepare() (
    cd "$work_root/$1"
    export JULIA_DEPOT_PATH="$work_root/depot-$1:"
    python3 - "$2" "$3" <<'PY'
from pathlib import Path
import re
import sys

for target in (Path("test/Project.toml"), Path("benchmarks/Project.toml")):
    text = target.read_text()
    for name, version in zip(("HiGHS", "HiGHS_jll"), sys.argv[1:]):
        text, count = re.subn(rf'(?m)^{name} = "~[^"]+"$', f'{name} = "={version}"', text)
        assert count == 1, (target, name)
    target.write_text(text)
PY
    julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
    julia --startup-file=no --project=test -e '
        using HiGHS, HiGHS_jll
        @assert Base.pkgversion(HiGHS) == VersionNumber(ARGS[1])
        @assert Base.pkgversion(HiGHS_jll) == VersionNumber(ARGS[2] * "+0")
        native = unsafe_string(HiGHS.Highs_version())
        @assert native == ARGS[2]
        println("Julia=", VERSION, " HiGHS.jl=", Base.pkgversion(HiGHS), " HiGHS_jll=", Base.pkgversion(HiGHS_jll), " native=", native)
    ' "$2" "$3"
)

run_suite() (
    pair="$1"
    run="$2"
    cd "$work_root/$pair"
    export JULIA_DEPOT_PATH="$work_root/depot-$pair:"
    printf 'Full suite: %s %s\n' "$pair" "$run"
    /usr/bin/time -f '%e' -o "$work_root/$pair-$run.seconds" \
        julia --startup-file=no --project=. -e 'using Pkg; Pkg.test(; coverage=false)'
    printf 'PASS %s %s (%s seconds)\n' "$pair" "$run" "$(cat "$work_root/$pair-$run.seconds")"
)

lscpu
prepare baseline 1.23.0 1.14.0
prepare candidate 1.27.0 1.16.0
run_suite baseline warmup
run_suite candidate warmup
run_suite baseline 1
run_suite candidate 1
run_suite candidate 2
run_suite baseline 2
run_suite baseline 3
run_suite candidate 3

python3 - "$work_root" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
times = {
    pair: [float((root / f"{pair}-{i}.seconds").read_text()) for i in range(1, 4)]
    for pair in ("baseline", "candidate")
}
for index, (before, after) in enumerate(zip(times["baseline"], times["candidate"]), 1):
    print(f"TIMING pair={index} baseline_s={before:.2f} candidate_s={after:.2f}")
baseline, candidate = (sum(times[k]) for k in ("baseline", "candidate"))
print(f"TIMING totals baseline_s={baseline:.2f} candidate_s={candidate:.2f} candidate_ratio={candidate / baseline:.4f}")
PY
