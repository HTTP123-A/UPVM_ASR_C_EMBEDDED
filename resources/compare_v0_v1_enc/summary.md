# Encoder-0 latency: f32 vs v0 (W8A8) vs v5 (W8A8 + requant fusions)

Micro-benchmark of **encoder 0** (`pvss_ds`) in isolation — input `[16,128,128]` →
output `[32,64,64]` — for the three implementations:

| label | source folder | weights |
|---|---|---|
| f32 (naives_mp) | `sources/naives_mp` | `data/weight_f32` |
| v0 (W8A8) | `sources/sq_int8_v0` | `data/weight_sq_int8_v0` |
| v5 (W8A8 + fusions) | `sources/sq_int8_v5` | `data/weight_sq_int8_v5` |

Each version is its own executable (block/kernel/struct names collide across versions),
built with the project's `-O3 -fopenmp`. The `*_timing_*` block timestamps the SSM→MLP
boundary, so SSM / MLP / total are split. Input is a fixed random `[16,128,128]` reset
every iteration (identical work each time); real enc0 weights are loaded. Reported =
**min over 50 timed iterations** (after 5 warmup). `mp` = SSM group-parallelism, the only
threaded part of the block — **the MLP runs sequentially in all three versions**, so its
cost is thread-independent.

> Numbers below are from one run on the build/run host (112 cores). Re-run with
> `./run.sh` to regenerate `results.csv` + this file.

## A. Equal-thread comparison (apples-to-apples — same SSM threads for all three)

This isolates the effect of **quantization + fusions** from the threading config.

| SSM threads (mp) | version | SSM (ms) | MLP (ms) | **total (ms)** | speedup vs f32 |
|---|---|---:|---:|---:|---:|
| 1 | f32 (naives_mp) | 69.45 | 8.85 | **78.32** | 1.00x |
|  | v0 (W8A8) | 73.55 | 9.37 | **83.05** | 0.94x |
|  | v5 (W8A8+fusions) | 73.72 | 10.49 | **84.45** | 0.93x |
| | | | | | |
| 2 | f32 (naives_mp) | 35.27 | 8.88 | **44.17** | 1.00x |
|  | v0 (W8A8) | 37.41 | 9.44 | **46.89** | 0.94x |
|  | v5 (W8A8+fusions) | 37.38 | 10.38 | **48.00** | 0.92x |
| | | | | | |
| 4 | f32 (naives_mp) | 28.38 | 8.86 | **37.28** | 1.00x |
|  | v0 (W8A8) | 28.94 | 9.44 | **38.48** | 0.97x |
|  | v5 (W8A8+fusions) | 29.88 | 10.33 | **40.26** | 0.93x |
| | | | | | |
| 8 | f32 (naives_mp) | 18.08 | 8.87 | **26.99** | 1.00x |
|  | v0 (W8A8) | 18.68 | 9.49 | **28.27** | 0.95x |
|  | v5 (W8A8+fusions) | 17.06 | 10.37 | **27.55** | 0.98x |
| | | | | | |
| 16 | f32 (naives_mp) | 10.22 | 8.92 | **19.18** | 1.00x |
|  | v0 (W8A8) | 10.72 | 9.51 | **20.27** | 0.95x |
|  | v5 (W8A8+fusions) | 10.35 | 10.31 | **20.68** | 0.93x |
| | | | | | |

## B. As-configured comparison (each version's native `INNER_MP_PARALLEL`)

The codebase ships v5 with `INNER_MP_PARALLEL = 16` but v0 / naives_mp with `2`. This row
shows the *deployed* latency — but the difference is **threading, not the math** (see §A).

| version | native mp (INNER_MP_PARALLEL) | SSM (ms) | MLP (ms) | **total (ms)** |
|---|---:|---:|---:|---:|
| f32 (naives_mp) | 2 | 35.27 | 8.88 | **44.17** |
| v0 (W8A8) | 2 | 37.41 | 9.44 | **46.89** |
| v5 (W8A8+fusions) | 16 | 10.35 | 10.31 | **20.68** |

## C. Findings — did we gain performance?

**1. Quantization (f32 → int8) gives no CPU latency gain here — it is marginally slower.**
At equal threads, v0 total ≈ **+6%** vs f32 and v5 ≈
**+8%** (mp=1). Reasons:
  - The SSM is dominated by the **shared f32 core** (selective scan, cross scan/merge,
    grouped conv1d) — *byte-identical f32 code in all three versions*. Quantization only
    touches the small in_proj / dw_conv / out_proj convs, and the naive scalar int8 kernels
    have **no SIMD advantage** over f32 at these tiny channel counts (SUB_C=1), while the
    added quant/dequant passes cost time.
  - The MLP (C=16→16→32) is likewise not faster in int8: f32 8.85 ms <
    v0 9.37 ms < v5 10.49 ms.

**2. v5's fusions vs v0 are latency-neutral at equal threads.**
v5/v0 total = **1.02x** at mp=1 and **1.02x** at mp=16;
the SSM (where v5 adds the in_proj→dw_conv REQUANT) is **0.97x** vs v0 at
mp=16 — i.e. tied. The requant removes one quant/dequant pass, but on C=16/SUB_C=1 that
pass is a negligible fraction of the f32-dominated SSM, so it does not move the needle. v5's
fused MLP is even slightly slower than v0's at these dimensions.

**3. The only real wall-clock win is threading, not the math.**
As-configured, v5 (20.68 ms @16 threads) is **2.27x** faster than
v0 (46.89 ms @2) and **2.14x** faster than f32 (44.17 ms
@2) — but purely because v5 runs the SSM with 16-way parallelism vs 2. At **equal** threads
(§A) they converge. SSM scales ~6.8x
from 1→16 threads; the sequential MLP does not scale and becomes the floor at high thread
counts.

**Bottom line:** on CPU with these naive scalar kernels, neither v0 nor v5 beats the f32
baseline in latency at the encoder-0 stage; the W8A8 path's value is memory / accelerator
offload (and bit-width), not naive-CPU speed, and v5's requant fusions are latency-neutral
vs v0. Real wall-clock differences between the shipped binaries come from the
`INNER_MP_PARALLEL` thread count, which is independent of quantization.
