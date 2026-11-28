# Independent numerical audit

Run these scripts with the benchmark environment from the measured source checkout. The readers
accept both live `.csv` files and the archived `.csv.gz` files, so `runs` can point directly to the
committed `runs/` directory. Set `study` to this directory, `runs` to a completed raw-run directory
and `out` to a new audit-output directory. The network argument is `WK17a` or `RSL18a`.

```sh
mkdir -p "$out"
julia --project=benchmarks "$study/replay_witnesses.jl" WK17a "$runs" "$out"
julia --project=benchmarks "$study/check_affine_layers.jl" WK17a "$out/affine-layers.csv"
julia --project=benchmarks "$study/check_bounds.jl" WK17a fallback "$runs" "$out"
julia --project=benchmarks "$study/check_bounds.jl" WK17a certificates-first-10 "$runs" "$out"
```

For RSL18a, replace `WK17a` with `RSL18a` and use `certificates` in place of
`certificates-first-10`. Its audit checks every direct certificate.

`replay_witnesses.jl` evaluates every saved candidate on the native network with weights and
coordinates represented as exact rationals. It includes the original-input shortcut and records
separate totals for the baseline and screened verifier. It checks the stored coordinates without
clipping or repairing them. The required input domain uses the stored Float64 centers and the exact
value represented by the Float64 radius `0.1`, matching the public perturbation family. A tie
satisfies the score condition. The CSV also records optional checks against decimal radius `1/10`
and original MNIST byte values divided by 255. Those optional checks impose a different domain and
do not determine `exact_witness_valid`.

`witness-replay.csv` contains one row for every available saved point and a SHA-256 digest of its
encoded coordinates. `witness-summary.csv` separates the public verification flags from the exact
results. `route-counts.csv` covers every timed row, including certificates, timeouts and
original-input shortcuts. An accepted baseline point can fail an exact check because the public
baseline uses solver-scale tolerances. Such failures stay in the output.

`check_affine_layers.jl` verifies each compiled coefficient and bias against the native layer
implementation by evaluating every input basis vector with zero biases. The full-domain checker then
verifies the hidden-neuron intervals and target bounds with independent exact-rational propagation.
Its proof arithmetic was adapted from the archived 25-Nov checker. `fallback` selects every
candidate image sent to the MIP. `certificates` selects directly certified images and also requires
every target bound to be strictly negative. `all` selects both groups. `certificates-first-10`
prespecifies the ten directly certified WK17a images with the smallest one-based MNIST test indices.
This sample checks 48,040 hidden intervals and 90 target bounds; it does not cover every direct
WK17a certificate. Each image is checked once, even when it occurs in both repeats.

After all timed workers finish, validate the complete observation and audit coverage:

```sh
python3 "$study/check_coverage.py" WK17a "$runs" "$out"
```

`check_coverage.py` requires both repeats of every image under both methods, matches every witness
digest to the raw point, and checks that all final fallback IDs and the prespecified
direct-certificate IDs have full-domain proofs. It also checks the native coefficient-audit coverage
and writes `coverage.json`.

A completed bound audit verifies the enclosures used in the supplied model. It does not
independently prove the integer solver’s infeasibility result. That verdict still depends on the
solver’s numerical behavior.

## WK17a results

The final coverage check passed all 8,000 method/repeat/image observations. Exact native replay
accepted all 252 screened-verifier points, including 98 original-input shortcuts and all eight
MIP-returned points. The baseline recorded 250 publicly verified points. Exact replay accepted 102,
including its 98 original-input shortcuts. The other 148 points exceeded the exact binary-radius
budget by at most 2.776 × 10⁻¹⁷. Of those, 76 also had a negative exact score gap, as low as −3.552
× 10⁻¹². These are numerical-tolerance discrepancies in the baseline’s saved coordinates. The audit
preserves those points and flags separately.

The compiled affine representation matched all 7,533,672 native coefficients and 4,814 biases.
Full-domain proofs passed for every one of the 34 final fallback images and the prespecified ten
direct certificates. This covers 211,376 hidden-neuron intervals and 396 target bounds. The sample
of direct certificates consists of images 1–8, 10 and 11, selected by the smallest directly
certified test indices. It is a sample of the 1,844 direct certificates.

[RSL18a’s complete audit](../rsl18a/audit/README.md) covers all 400 observations, every saved point,
every fallback image and all direct certificates.
