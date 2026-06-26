#!/usr/bin/env python3
"""Aggregate the encoder-0 latency sweep (results.csv) into summary.md + summary.csv.

results.csv columns:
  version,mp,iters,mean_ssm_us,mean_mlp_us,mean_tot_us,min_ssm_us,min_mlp_us,min_tot_us

We report the MIN over the timed iterations (the cleanest latency estimate, least
affected by scheduler jitter). `mp` is the SSM group-parallelism (the only threaded
part of the block; the MLP runs sequentially in every version).
"""
import csv, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.csv")

VERS = ["f32_naive_mp", "sq_int8_v0", "sq_int8_v5"]
LABEL = {"f32_naive_mp": "f32 (naives_mp)", "sq_int8_v0": "v0 (W8A8)", "sq_int8_v5": "v5 (W8A8+fusions)"}
NATIVE_MP = {"f32_naive_mp": 2, "sq_int8_v0": 2, "sq_int8_v5": 16}  # INNER_MP_PARALLEL per version

rows = {}
mps = []
with open(SRC) as f:
    for r in csv.DictReader(f):
        v, mp = r["version"], int(r["mp"])
        rows[(v, mp)] = {k: float(r[k]) for k in r if k not in ("version",)}
        if mp not in mps:
            mps.append(mp)
mps.sort()

def g(v, mp, col):
    return rows.get((v, mp), {}).get(col, float("nan"))

def md_table_equalthreads():
    out = ["| SSM threads (mp) | version | SSM (ms) | MLP (ms) | **total (ms)** | speedup vs f32 |",
           "|---|---|---:|---:|---:|---:|"]
    for mp in mps:
        f32t = g("f32_naive_mp", mp, "min_tot_us")
        for v in VERS:
            ssm = g(v, mp, "min_ssm_us") / 1e3
            mlp = g(v, mp, "min_mlp_us") / 1e3
            tot = g(v, mp, "min_tot_us") / 1e3
            sp = f32t / g(v, mp, "min_tot_us")
            mpcell = str(mp) if v == VERS[0] else ""
            out.append(f"| {mpcell} | {LABEL[v]} | {ssm:.2f} | {mlp:.2f} | **{tot:.2f}** | {sp:.2f}x |")
        out.append("| | | | | | |")
    return "\n".join(out)

def md_table_native():
    out = ["| version | native mp (INNER_MP_PARALLEL) | SSM (ms) | MLP (ms) | **total (ms)** |",
           "|---|---:|---:|---:|---:|"]
    for v in VERS:
        mp = NATIVE_MP[v]
        out.append(f"| {LABEL[v]} | {mp} | {g(v,mp,'min_ssm_us')/1e3:.2f} | "
                   f"{g(v,mp,'min_mlp_us')/1e3:.2f} | **{g(v,mp,'min_tot_us')/1e3:.2f}** |")
    return "\n".join(out)

# headline ratios
mp1 = 1 if 1 in mps else mps[0]
mpmax = max(mps)
v0_over_f32_1 = g("sq_int8_v0", mp1, "min_tot_us") / g("f32_naive_mp", mp1, "min_tot_us")
v5_over_f32_1 = g("sq_int8_v5", mp1, "min_tot_us") / g("f32_naive_mp", mp1, "min_tot_us")
v5_over_v0_1  = g("sq_int8_v5", mp1, "min_tot_us") / g("sq_int8_v0", mp1, "min_tot_us")
v5_over_v0_max = g("sq_int8_v5", mpmax, "min_tot_us") / g("sq_int8_v0", mpmax, "min_tot_us")
v5_ssm_over_v0_max = g("sq_int8_v5", mpmax, "min_ssm_us") / g("sq_int8_v0", mpmax, "min_ssm_us")
# native deployed
nat_v5 = g("sq_int8_v5", NATIVE_MP["sq_int8_v5"], "min_tot_us")
nat_v0 = g("sq_int8_v0", NATIVE_MP["sq_int8_v0"], "min_tot_us")
nat_f32 = g("f32_naive_mp", NATIVE_MP["f32_naive_mp"], "min_tot_us")
mlp_v0 = g("sq_int8_v0", mp1, "min_mlp_us"); mlp_v5 = g("sq_int8_v5", mp1, "min_mlp_us"); mlp_f32 = g("f32_naive_mp", mp1, "min_mlp_us")

md = f"""# Encoder-0 latency: f32 vs v0 (W8A8) vs v5 (W8A8 + requant fusions)

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

{md_table_equalthreads()}

## B. As-configured comparison (each version's native `INNER_MP_PARALLEL`)

The codebase ships v5 with `INNER_MP_PARALLEL = 16` but v0 / naives_mp with `2`. This row
shows the *deployed* latency — but the difference is **threading, not the math** (see §A).

{md_table_native()}

## C. Findings — did we gain performance?

**1. Quantization (f32 → int8) gives no CPU latency gain here — it is marginally slower.**
At equal threads, v0 total ≈ **{v0_over_f32_1*100-100:+.0f}%** vs f32 and v5 ≈
**{v5_over_f32_1*100-100:+.0f}%** (mp=1). Reasons:
  - The SSM is dominated by the **shared f32 core** (selective scan, cross scan/merge,
    grouped conv1d) — *byte-identical f32 code in all three versions*. Quantization only
    touches the small in_proj / dw_conv / out_proj convs, and the naive scalar int8 kernels
    have **no SIMD advantage** over f32 at these tiny channel counts (SUB_C=1), while the
    added quant/dequant passes cost time.
  - The MLP (C=16→16→32) is likewise not faster in int8: f32 {mlp_f32/1e3:.2f} ms <
    v0 {mlp_v0/1e3:.2f} ms < v5 {mlp_v5/1e3:.2f} ms.

**2. v5's fusions vs v0 are latency-neutral at equal threads.**
v5/v0 total = **{v5_over_v0_1:.2f}x** at mp=1 and **{v5_over_v0_max:.2f}x** at mp={mpmax};
the SSM (where v5 adds the in_proj→dw_conv REQUANT) is **{v5_ssm_over_v0_max:.2f}x** vs v0 at
mp={mpmax} — i.e. tied. The requant removes one quant/dequant pass, but on C=16/SUB_C=1 that
pass is a negligible fraction of the f32-dominated SSM, so it does not move the needle. v5's
fused MLP is even slightly slower than v0's at these dimensions.

**3. The only real wall-clock win is threading, not the math.**
As-configured, v5 ({nat_v5/1e3:.2f} ms @16 threads) is **{nat_v0/nat_v5:.2f}x** faster than
v0 ({nat_v0/1e3:.2f} ms @2) and **{nat_f32/nat_v5:.2f}x** faster than f32 ({nat_f32/1e3:.2f} ms
@2) — but purely because v5 runs the SSM with 16-way parallelism vs 2. At **equal** threads
(§A) they converge. SSM scales ~{g('f32_naive_mp',1,'min_ssm_us')/g('f32_naive_mp',mpmax,'min_ssm_us'):.1f}x
from 1→{mpmax} threads; the sequential MLP does not scale and becomes the floor at high thread
counts.

**Bottom line:** on CPU with these naive scalar kernels, neither v0 nor v5 beats the f32
baseline in latency at the encoder-0 stage; the W8A8 path's value is memory / accelerator
offload (and bit-width), not naive-CPU speed, and v5's requant fusions are latency-neutral
vs v0. Real wall-clock differences between the shipped binaries come from the
`INNER_MP_PARALLEL` thread count, which is independent of quantization.
"""

with open(os.path.join(HERE, "summary.md"), "w") as f:
    f.write(md)

# tidy summary.csv (min, ms)
with open(os.path.join(HERE, "summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["version", "mp_ssm_threads", "ssm_ms", "mlp_ms", "total_ms", "speedup_vs_f32"])
    for mp in mps:
        f32t = g("f32_naive_mp", mp, "min_tot_us")
        for v in VERS:
            w.writerow([v, mp, f"{g(v,mp,'min_ssm_us')/1e3:.3f}", f"{g(v,mp,'min_mlp_us')/1e3:.3f}",
                        f"{g(v,mp,'min_tot_us')/1e3:.3f}", f"{f32t/g(v,mp,'min_tot_us'):.3f}"])

print("wrote summary.md and summary.csv")
