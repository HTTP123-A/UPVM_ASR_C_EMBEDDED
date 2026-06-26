# Encoder-0 latency comparison — f32 vs v0 vs v5

Isolated micro-benchmark of the **first encoder block** (`pvss_ds`, encoder 0),
input `[16,128,128]` → output `[32,64,64]`, for three implementations:

| binary | version | source | weights |
|---|---|---|---|
| `bench_f32` | original float32 | `sources/naives_mp` | `data/weight_f32` |
| `bench_v0`  | W8A8 baseline | `sources/sq_int8_v0` | `data/weight_sq_int8_v0` |
| `bench_v5`  | W8A8 + requant fusions | `sources/sq_int8_v5` | `data/weight_sq_int8_v5` |

**Results: see [`summary.md`](summary.md)** (and `summary.csv` / `results.csv`).

## Why three separate executables
The block, kernel and struct names are identical across versions
(`pvss_ds_timing_mp_*`, `layernorm_f32`, `struct pvss_ds_weight`, …), so they cannot be
linked into one program. Each benchmark links **only** its own version's kernels +
timing block + weight loader. `bench_int8.c` is shared by v0/v5 (compiled twice with a
different `-I`); `bench_f32.c` is the float reference.

## Methodology (fairness)
- Drives **only** encoder 0 via the version's `*_timing_*` block, which stamps the
  **SSM→MLP boundary** and the **block end**, so SSM / MLP / total are reported separately.
  (Per the task: the op chains differ, so only the coarse SSM / MLP / total split is
  compared, not finer per-op timings.)
- Input is a **fixed random `[16,128,128]`** (same seed for all three) reset every
  iteration → identical work each iteration. Real enc0 weights are loaded
  (well-conditioned, no denormals).
- `mp` = SSM group-parallelism, passed as a CLI arg so all versions run at **equal thread
  counts** (the apples-to-apples comparison). The MLP runs **sequentially** in every
  version, so its cost is thread-independent. Each version's *native* `INNER_MP_PARALLEL`
  (f32=2, v0=2, v5=16) is also reported.
- Build flags match the project (`-O3 -fopenmp`); reported value = **min over 50 timed
  iterations** after 5 warmups.

## Run
```bash
# noexec mount (NAS/QNAP): stage a runnable copy on local disk
SANDBOX=/tmp/encbench FFTW_INC=/path/to/fftw3/local/include ./run.sh
# runnable checkout: build+run in place
FFTW_INC=/path/to/fftw3/local/include ./run.sh
```
`FFTW_INC` only needs to expose `fftw3.h` as a **header** (pulled in transitively by
`model.h`); no FFTW symbol is referenced or linked. Tunables:
`MPS="1 2 4 8 16"`, `WARM=5`, `ITERS=50`.

Manual single run:
```bash
make ROOT=../.. FFTW_INC=/path/to/fftw3/local/include
OMP_NUM_THREADS=4 ./bench_v5 4 5 50 ../../data/weight_sq_int8_v5
# args: <ssm_threads> <warmup_iters> <timed_iters> <weight_root>
# stdout: version,mp,iters,mean_ssm_us,mean_mlp_us,mean_tot_us,min_ssm_us,min_mlp_us,min_tot_us
```

## Files
| file | purpose |
|---|---|
| `bench_int8.c` | driver for v0 / v5 (compiled per version) |
| `bench_f32.c` | driver for the float32 reference |
| `bench_common.h` | timing + deterministic-RNG helpers |
| `Makefile` | builds the 3 executables (own `-I` each) |
| `run.sh` | build + thread sweep + aggregate (sandbox-aware) |
| `aggregate.py` | `results.csv` → `summary.md` + `summary.csv` |
| `summary.md` | **the writeup / findings** |
| `results.csv` | raw mean+min per (version, mp) |
