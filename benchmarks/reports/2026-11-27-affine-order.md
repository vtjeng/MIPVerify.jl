# Which affine screen should run first?

This historical experiment used the
[verifier and replay driver at the measured revision](https://github.com/vtjeng/MIPVerify.jl/blob/05e83135d80004f2e96434f4bbe88ad2f7e11832/benchmarks/reports/affine_screen_order.jl).
Check out that revision to reproduce it.

The adaptive-first configuration applied the adaptive lower line first and tried forward
parallelograms on the rivals it left open. Both orders certify the same images when each rival
retains its smaller upper bound. The order changes how often the second pass runs and how much the
screen costs.

| Network | First pass                | Certified after first pass | Certified after both | Rival labels excluded after both | Run 1 (s) | Run 2 (s) |
| ------- | ------------------------- | -------------------------: | -------------------: | -------------------------------: | --------: | --------: |
| WK17a   | Adaptive lower line first |                       1844 |                 1846 |                            17770 |     187.9 |     178.5 |
| WK17a   | Parallel lower line first |                       1833 |                 1846 |                            17770 |     275.2 |     256.1 |
| RSL18a  | Adaptive lower line first |                         35 |                   35 |                              554 |       3.8 |       3.3 |
| RSL18a  | Parallel lower line first |                         29 |                   35 |                              554 |       6.4 |       5.4 |

Each run uses the first 2,000 WK17a or first 100 RSL18a MNIST test images at pixel radius 0.10.
Times sum per-image screening seconds across five single-threaded Julia workers on an Intel Core
i7-8700K, each pinned to a separate available core. The two orders run on the same images,
alternating which goes first and reversing that order for the repeat. We warmed up before measuring.
Counts matched across repeats. A time entry covers the two screens with early exits. It excludes
compilation, PGD, Target LP and the final MIP. The earlier end-to-end timings in the
[backward screening report](2026-11-26-backward-bounds.md) describe the adaptive-only version.

The adaptive screen maximizes each rival-minus-true objective backward. When a crossing ReLU has a
negative coefficient it uses the diagonal lower edge if the positive endpoint is at least as far
from zero as the negative endpoint, and the zero edge otherwise. The forward alternative propagates
a shared error variable per crossing ReLU and subtracts the resulting logit expressions before
maximizing.

[Per-image measurements](2026-11-27-affine-order.csv.gz) ·
[Replay driver](https://github.com/vtjeng/MIPVerify.jl/blob/05e83135d80004f2e96434f4bbe88ad2f7e11832/benchmarks/reports/affine_screen_order.jl).
Run the driver with the benchmarks project, the output filename, and the worker index and count. Use
index and count 1 for a single worker.
