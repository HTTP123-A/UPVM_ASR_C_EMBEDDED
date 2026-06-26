# UPVM_ASR_C_EMBEDDED

C host application for UPVM-ASR speech super-resolution inference (16 kHz → 48 kHz)
and timing on WAV files. A dual-stream (magnitude / phase) VMamba model runs over
STFT frames; the C port mirrors the PyTorch reference and supports float32 and
W8A8 quantized execution.

---

## 1. Versions

| `VERSION`      | Precision                          | Parallelism      | Weights dir              | Modes |
|----------------|------------------------------------|------------------|--------------------------|-------|
| `NAIVES`       | float32, sequential                | none             | `data/weight_f32/`       | `NORMAL`, `STAGE_TIMING`, `DETAIL_STAGE_TIMING` |
| `NAIVES_MP`    | float32, OpenMP                     | `-fopenmp`       | `data/weight_f32/`       | `NORMAL`, `STAGE_TIMING`, `DETAIL_STAGE_TIMING` |
| `SQ_INT8_V0`   | INT8/F32 mixed (W8A8 SmoothQuant)  | `-fopenmp`       | `data/weight_sq_int8_v0/`| `NORMAL`, `STAGE_TIMING`, `DETAIL_STAGE_TIMING`, `MEM_PROFILE` |
| `SQ_INT8_V1`   | INT8/F32 mixed (W8A8); MLP `fc1` ReLU fused on the int32 accumulator (bit-identical to V0) | `-fopenmp` | `data/weight_sq_int8_v0/` (shared with V0) | `NORMAL`, `STAGE_TIMING`, `DETAIL_STAGE_TIMING`, `MEM_PROFILE` |
| `SQ_INT8_V2`   | INT8/F32 mixed (W8A8); V1 **+ encoder MLP SumPool fused into int32** before dequant (more accurate; not bit-identical to V1) | `-fopenmp` | `data/weight_sq_int8_v2/` (copied from V0) | `NORMAL`, `STAGE_TIMING`, `DETAIL_STAGE_TIMING`, `MEM_PROFILE` |
| `SQ_INT8_V3`   | INT8/F32 mixed (W8A8); V2 **+ SSM `x` ReLU fused into int32** before the dw-conv dequant (bit-identical to V2) | `-fopenmp` | `data/weight_sq_int8_v3/` (copied from V2) | `NORMAL`, `STAGE_TIMING`, `DETAIL_STAGE_TIMING`, `MEM_PROFILE` |
| `SQ_INT8_V4`   | INT8/F32 mixed (W8A8); V3 **+ MLP `fc1`→`fc2` dequant+quant fused into one int32→int8 REQUANT** (encoder/latent/decoder; output stage untouched; not bit-identical to V3) | `-fopenmp` | `data/weight_sq_int8_v4/` (V3 bytes + `mlp_requant_scale`, from `weight_extractor_sq_int8_v4.py`) | `NORMAL`, `STAGE_TIMING`, `DETAIL_STAGE_TIMING`, `MEM_PROFILE` |
| `SQ_INT8_V5`   | INT8/F32 mixed (W8A8); V4 **+ SSM in_proj→dw_conv2d dequant+quant fused into one int32→int8 REQUANT** on the `x` branch (encoder/latent/decoder; `z` branch + output stage untouched; not bit-identical to V4) | `-fopenmp` | `data/weight_sq_int8_v5/` (V4 bytes + `ssm_requant_scale`, from `weight_extractor_sq_int8_v5.py`) | `NORMAL`, `STAGE_TIMING`, `DETAIL_STAGE_TIMING`, `MEM_PROFILE` |
| `SQ_INT8_V6`   | INT8/F32 mixed (W8A8); V5 **+ MLP REQUANT made integer-only** (fixed-point `mul`+`shift`, no f32/`lrintf` at runtime; encoder/latent/decoder; SSM requant stays f32; output untouched; not bit-identical to V5) | `-fopenmp` | `data/weight_sq_int8_v6/` (V5 bytes + int32 `mlp_requant_mul`/`mlp_requant_shift`, from `weight_extractor_sq_int8_v6.py`) | `NORMAL`, `STAGE_TIMING`, `DETAIL_STAGE_TIMING`, `MEM_PROFILE` |
| `SQ_INT8_V7`   | INT8/F32 mixed (W8A8); V6 **+ SSM REQUANT also made integer-only** (same fixed-point `mul`+`shift`; the SSM in_proj→dw_conv2d `x`-branch requant, encoder/latent/decoder; `z` branch dequant + output untouched; not bit-identical to V6) | `-fopenmp` | `data/weight_sq_int8_v7/` (V6 bytes + int32 `ssm_requant_mul`/`ssm_requant_shift`, from `weight_extractor_sq_int8_v7.py`) | `NORMAL`, `STAGE_TIMING`, `DETAIL_STAGE_TIMING`, `MEM_PROFILE` |
| `SQ_INT8_V8`   | INT8/F32 mixed (W8A8); V7 **+ PatchEmbed LayerNorm rsqrt replaced by a range-reduced piece-wise-linear LUT** (the 2 patch-embed norms only; conv dequant fused into the norm, reads the int32 acc; no `sqrtf`/divide; other LNs + gelu stay f32; not bit-identical to V7) | `-fopenmp` | `data/weight_sq_int8_v8/` (V7 bytes + shared f32 PWL `pwl_rsqrt_knot`/`slope`/`intercept`, from `weight_extractor_sq_int8_v8.py`) | `NORMAL`, `STAGE_TIMING`, `DETAIL_STAGE_TIMING`, `MEM_PROFILE` |
| `SQ_INT8_V9`   | INT8/F32 mixed (W8A8); V8 **+ PatchEmbed LayerNorm made FULLY INTEGER** (int64 mean/var, fixed-point rsqrt LUT + affine) with **norm1 → int8** (calibrated `S_ln`) feeding an **int8 GELU LUT** (fused dequant+GELU+quant) feeding `conv2` directly; **norm2 → f32** (encoder residual). Only f32 left in patch-embed = norm2 output; not bit-identical to V8 | `-fopenmp` | `data/weight_sq_int8_v9/` (V8 bytes + int fixed-point LN tables + per-channel GELU LUT, from `weight_extractor_sq_int8_v9.py` + `calibrate_v9.py`) | `NORMAL`, `STAGE_TIMING`, `DETAIL_STAGE_TIMING`, `MEM_PROFILE` |
| `SQ_INT8_V10`  | INT8/F32 mixed (W8A8); **numerically identical to V9** — V9's hot int8 kernels **vectorized with AVX-VNNI** (`vpdpbusd`, 256-bit): `pointwise_conv2d_int8`/`_split2` (MLP + SS2D in/out_proj) via an on-stack transpose tile + `+128` offset compensation; `depthwise_conv2d_int8`/`_int32` via AVX2 widening. **Bit-exact** to V9 (same SNR) — only latency changes. Scalar fallback when not built with `-mavxvnni`. See §7.10. | `-fopenmp`; `-mavx2 -mfma -mavxvnni` on the 4 int8 kernels only | `data/weight_sq_int8_v10/` (**identical bytes to V9**; `weight_extractor_sq_int8_v10.py`) | `NORMAL`, `STAGE_TIMING`, `DETAIL_STAGE_TIMING`, `MEM_PROFILE` |

All versions share the same input wavs (`data/test_audio/downsample/`) and write to
their own `data/test_audio/generated/<version>/` and `results/<version>/` folders.
`SQ_INT8_V1` reuses V0's exported weights unchanged (the fusion alters no weights or
scales), so it needs no separate weight export. See §7 for what the fusion does.

---

## 2. Prerequisites

- **Compiler:** `gcc` with C11 and OpenMP (`-fopenmp`).
- **FFTW (single precision, `fftw3f`):** auto-detected. The Makefile uses a local
  build at `../fftw3/local` (relative to this directory) if present — preferring the
  static `lib/libfftw3f.a` — otherwise it links the system `-lfftw3f`.
  Override with `make LOCAL_FFTW_PREFIX=/path/to/fftw`.
- **System libs:** `libm`, `pthread`, and `librt` (Linux).
- **Data in place before running** (see sections 5–6):
  - weights exported into the version's weights dir,
  - degraded input wavs in `data/test_audio/downsample/`.

---

## 3. Compile & Run Each Version

> Build artifacts are namespaced per `build/<version>/<mode>/`, so versions and
> modes coexist — **no `make clean` is needed when switching `VERSION`/`MODE`**.
> Run binaries **from this directory** (`UPVM_ASR_C_EMBEDDED/`); all data paths are
> relative (`./data/...`, `./results/...`).

### 3.1 `naives` — float32, sequential
```bash
make VERSION=NAIVES MODE=NORMAL
./upvm_asr_naives
```

### 3.2 `naives_mp` — float32, OpenMP
```bash
make VERSION=NAIVES_MP MODE=NORMAL
OMP_NUM_THREADS=8 ./upvm_asr_naives_mp     # OMP_NUM_THREADS controls thread count
```

### 3.3 `sq_int8_v0` — INT8/F32 mixed (W8A8), OpenMP
```bash
make VERSION=SQ_INT8_V0 MODE=NORMAL
OMP_NUM_THREADS=8 ./upvm_asr_sq_int8_v0
```

### 3.4 `sq_int8_v1` — INT8/F32 mixed (W8A8) with fused int32 MLP ReLU, OpenMP
```bash
make VERSION=SQ_INT8_V1 MODE=NORMAL
OMP_NUM_THREADS=8 ./upvm_asr_sq_int8_v1
```

### 3.5 `sq_int8_v2` — V1 + encoder MLP SumPool fused into int32, OpenMP
```bash
make VERSION=SQ_INT8_V2 MODE=NORMAL
OMP_NUM_THREADS=8 ./upvm_asr_sq_int8_v2
```

### 3.6 `sq_int8_v3` — V2 + SSM `x` ReLU fused into int32, OpenMP
```bash
make VERSION=SQ_INT8_V3 MODE=NORMAL
OMP_NUM_THREADS=8 ./upvm_asr_sq_int8_v3
```

### 3.7 `sq_int8_v4` — V3 + MLP `fc1`→`fc2` dequant+quant fused into one REQUANT, OpenMP
```bash
# one-time: generate v4 weights (V3 bytes + per-stage mlp_requant_scale)
python model_pytorch/weight_extractor_sq_int8_v4.py
make VERSION=SQ_INT8_V4 MODE=NORMAL
OMP_NUM_THREADS=8 ./upvm_asr_sq_int8_v4
```

### 3.8 `sq_int8_v5` — V4 + SSM in_proj→dw_conv2d dequant+quant fused into one REQUANT, OpenMP
```bash
# one-time: generate v5 weights (V4 bytes + per-stage ssm_requant_scale)
python model_pytorch/weight_extractor_sq_int8_v5.py
make VERSION=SQ_INT8_V5 MODE=NORMAL
OMP_NUM_THREADS=8 ./upvm_asr_sq_int8_v5
```

### 3.9 `sq_int8_v6` — V5 + integer-only (fixed-point) MLP REQUANT, OpenMP
```bash
# one-time: generate v6 weights (V5 bytes + per-stage int32 mlp_requant_mul/shift)
python model_pytorch/weight_extractor_sq_int8_v6.py
make VERSION=SQ_INT8_V6 MODE=NORMAL
OMP_NUM_THREADS=8 ./upvm_asr_sq_int8_v6
```

### 3.10 `sq_int8_v7` — V6 + integer-only (fixed-point) SSM REQUANT, OpenMP
```bash
# one-time: generate v7 weights (V6 bytes + per-stage int32 ssm_requant_mul/shift)
python model_pytorch/weight_extractor_sq_int8_v7.py
make VERSION=SQ_INT8_V7 MODE=NORMAL
OMP_NUM_THREADS=8 ./upvm_asr_sq_int8_v7
```

### 3.11 `sq_int8_v8` — V7 + PatchEmbed LayerNorm rsqrt via PWL+LUT, OpenMP
```bash
# one-time: generate v8 weights (V7 bytes + shared f32 PWL rsqrt table: knot/slope/intercept)
python model_pytorch/weight_extractor_sq_int8_v8.py
make VERSION=SQ_INT8_V8 MODE=NORMAL
OMP_NUM_THREADS=8 ./upvm_asr_sq_int8_v8
```

### 3.12 `sq_int8_v9` — V8 + fully-integer PatchEmbed LayerNorm (int8 out) + int8 GELU LUT, OpenMP
```bash
# one-time: calibrate S_ln (needs the Revision PyTorch model + a torch env), then generate v9 weights
python model_pytorch/calibrate_v9.py                       # -> model_pytorch/calib_v9_s_ln.json
python model_pytorch/weight_extractor_sq_int8_v9.py        # V8 bytes + int LN tables + GELU LUT
make VERSION=SQ_INT8_V9 MODE=NORMAL
OMP_NUM_THREADS=8 ./upvm_asr_sq_int8_v9
```

Each run scans `data/test_audio/downsample/`, performs inference, and writes the
upsampled 48 kHz wavs plus timing reports (section 4).

---

## 4. Build Selectors, Binaries & Outputs

### 4.1 Selectors
- `VERSION` ∈ `NAIVES`, `NAIVES_MP`, `SQ_INT8_V0`, `SQ_INT8_V1`, `SQ_INT8_V2`, `SQ_INT8_V3`, `SQ_INT8_V4`, `SQ_INT8_V5`, `SQ_INT8_V6`, `SQ_INT8_V7`, `SQ_INT8_V8`, `SQ_INT8_V9` (default `NAIVES`).
- `MODE` ∈ `NORMAL`, `STAGE_TIMING`, `DETAIL_STAGE_TIMING`, `MEM_PROFILE` (default `NORMAL`).
  All versions support the three timing modes; `MEM_PROFILE` (per-op memory report)
  is `SQ_INT8_V0` / `SQ_INT8_V1` / `SQ_INT8_V2` / `SQ_INT8_V3` / `SQ_INT8_V4` / `SQ_INT8_V5` / `SQ_INT8_V6` / `SQ_INT8_V7` / `SQ_INT8_V8` / `SQ_INT8_V9` only.
- Useful: `make info VERSION=… MODE=…` prints the resolved paths/target.
- `make clean` is global: it removes the whole `build/` tree (every version/mode)
  and every `upvm_asr_*` binary at the repo root (`VERSION`/`MODE` args are ignored).
  It does not touch `data/`, `results/`, or weights.

### 4.2 Binary names (at repo root)
- NORMAL: `./upvm_asr_<version>` (e.g. `upvm_asr_naives`, `upvm_asr_sq_int8_v0`, `upvm_asr_sq_int8_v1`, `upvm_asr_sq_int8_v2`, `upvm_asr_sq_int8_v3`, `upvm_asr_sq_int8_v4`, `upvm_asr_sq_int8_v5`, `upvm_asr_sq_int8_v6`, `upvm_asr_sq_int8_v7`, `upvm_asr_sq_int8_v8`, `upvm_asr_sq_int8_v9`).
- Timing: `./upvm_asr_<version>_stage_timing`, `./upvm_asr_<version>_detail_stage_timing`.
- Memory: `./upvm_asr_<version>_mem_profile` (`SQ_INT8_V0` / `SQ_INT8_V1` / `SQ_INT8_V2` / `SQ_INT8_V3` / `SQ_INT8_V4` / `SQ_INT8_V5` / `SQ_INT8_V6` / `SQ_INT8_V7` / `SQ_INT8_V8` / `SQ_INT8_V9`); writes `results/<version>/memory_profile.csv`.

### 4.3 Per-run outputs (`results/<version>/`)
- `processing_time.csv`, `processing_time.txt` — always.
- `stage_timing.csv` — when `MODE=STAGE_TIMING`.
- `detail_stage_timing.csv` — when `MODE=DETAIL_STAGE_TIMING`.

Generated audio lands in `data/test_audio/generated/<version>/`.

---

## 5. Weight Extraction

Both extractors read PyTorch checkpoints from `model_pytorch/model/` and export the
flat `.bin` tensors (+ `manifest.csv` / `meta.csv`) consumed by the C loader.
Run from the repo root (`UPVM-ASR-0`):

```bash
# float32 weights  ->  data/weight_f32/        (for naives, naives_mp)
python UPVM_ASR_C_EMBEDDED/model_pytorch/weight_extractor.py

# W8A8 int8 weights ->  data/weight_sq_int8_v0/ (for sq_int8_v0 AND sq_int8_v1)
python UPVM_ASR_C_EMBEDDED/model_pytorch/weight_extractor_sq_int8_v0.py
```

`sq_int8_v1` consumes the same `data/weight_sq_int8_v0/` tensors — the int32-ReLU
fusion changes no weights or scales, so there is no `weight_extractor_sq_int8_v1.py`.
`sq_int8_v2` uses its own copy `data/weight_sq_int8_v2/` (bytes copied from
`data/weight_sq_int8_v0/`); the added int32 SumPool fusion also changes no weights or
scales, so there is no `weight_extractor_sq_int8_v2.py` either.
`sq_int8_v3` uses its own copy `data/weight_sq_int8_v3/` (bytes copied from
`data/weight_sq_int8_v2/`); the added SSM-`x`-ReLU fusion likewise changes nothing, so
there is no `weight_extractor_sq_int8_v3.py`.
`sq_int8_v4` is the **first int8 version that needs a new scale**: the fused
`fc1`→`fc2` REQUANT consumes a per-stage `mlp_requant_scale = fc1.deq_scale / fc2.act_scale`.
`weight_extractor_sq_int8_v4.py` reuses the V0 fold logic for every base tensor (so the
`.bin` set is byte-identical to V3) and just appends the 14 `mlp_requant_scale.bin`
files (encoder×6, latent×2, decoder×6) into `data/weight_sq_int8_v4/`:

```bash
# W8A8 int8 weights + requant scales ->  data/weight_sq_int8_v4/ (for sq_int8_v4)
python UPVM_ASR_C_EMBEDDED/model_pytorch/weight_extractor_sq_int8_v4.py
```

`sq_int8_v5` adds a second per-stage scale for the fused SSM in_proj→dw_conv2d REQUANT:
`ssm_requant_scale = ssm_in_projection.deq_scale[:SUB_D_INNER] / ssm_dw_conv2d.act_scale`
(only the `x` half of the `[x|z]` in-projection feeds the depthwise conv).
`weight_extractor_sq_int8_v5.py` reuses V4 verbatim (so every base + `mlp_requant_scale.bin`
is byte-identical to V4) and appends the 14 `ssm_requant_scale.bin` files
(encoder×6, latent×2, decoder×6) into `data/weight_sq_int8_v5/`:

```bash
# W8A8 int8 weights + mlp & ssm requant scales ->  data/weight_sq_int8_v5/ (for sq_int8_v5)
python UPVM_ASR_C_EMBEDDED/model_pytorch/weight_extractor_sq_int8_v5.py
```

`sq_int8_v6` turns the MLP requant integer-only: for each of the 14 fused MLP stages it
adds a fixed-point pair `mlp_requant_mul[MLPh]` / `mlp_requant_shift[MLPh]` (both int32) with
`mlp_requant_scale[c] ~= mul[c] / 2^shift[c]`. `weight_extractor_sq_int8_v6.py` reuses V5
verbatim (every base + `*_requant_scale.bin` is byte-identical to V5) and appends the 14
`mlp_requant_mul.bin` + 14 `mlp_requant_shift.bin`. The shift is **searched** per channel
(all of `[1,30]`, pick the one minimizing the integer-vs-f32 error — see §7.6):

```bash
# V5 weights + per-stage int32 mlp_requant_mul/shift ->  data/weight_sq_int8_v6/ (for sq_int8_v6)
python UPVM_ASR_C_EMBEDDED/model_pytorch/weight_extractor_sq_int8_v6.py
```

`sq_int8_v7` does the same to the **SSM** requant: for each of the 14 fused SSM stages it adds
`ssm_requant_mul[D]` / `ssm_requant_shift[D]` (both int32) with `ssm_requant_scale[c] ~= mul[c] /
2^shift[c]`. `weight_extractor_sq_int8_v7.py` reuses V6 verbatim (every base + `*_requant_scale.bin`
+ V6's `mlp_requant_mul/shift` byte-identical) and appends the 14 `ssm_requant_mul.bin` + 14
`ssm_requant_shift.bin` (same per-channel shift search as V6):

```bash
# V6 weights + per-stage int32 ssm_requant_mul/shift ->  data/weight_sq_int8_v7/ (for sq_int8_v7)
python UPVM_ASR_C_EMBEDDED/model_pytorch/weight_extractor_sq_int8_v7.py
```

`sq_int8_v8` adds a shared **PWL rsqrt table** for the fused PatchEmbed LayerNorm:
`pwl_rsqrt_knot` [`PWL_RSQRT_SEGMENTS`+1], `pwl_rsqrt_slope` / `pwl_rsqrt_intercept`
[`PWL_RSQRT_SEGMENTS`] (all f32), written into **both** embed stages (the table is data-independent).
`weight_extractor_sq_int8_v8.py` reuses V7 verbatim (every base + requant `.bin` byte-identical to V7)
and appends the 6 PWL files; the segment count is tuned to the fewest meeting a relative-error target
(§7.8) and frozen as `PWL_RSQRT_SEGMENTS` in `headers/sq_int8_v8/datatypes.h`:

```bash
# V7 weights + shared f32 PWL rsqrt table ->  data/weight_sq_int8_v8/ (for sq_int8_v8)
python UPVM_ASR_C_EMBEDDED/model_pytorch/weight_extractor_sq_int8_v8.py
```

`sq_int8_v9` makes the PatchEmbed LayerNorm **fully integer**. It adds two steps:
`calibrate_v9.py` runs the Revision PyTorch model (forward hook on `norm1`) over real wavs to measure
the int8 output scale **`S_ln`** (per stream) → `model_pytorch/calib_v9_s_ln.json`; then
`weight_extractor_sq_int8_v9.py` reuses V8 verbatim (every base + requant `.bin` byte-identical to V8)
and appends the fixed-point LN tables (`norm1_/norm2_deq_mul`, `gamma_q`, `bfix`, `amul`/`ash`/`deq_sh`,
shared `rsqrt_knot_q`/`sl_q`/`ic_q`) plus the per-channel int8 `gelu_lut` (`[C_MID][256]`). Q-formats are
in `headers/sq_int8_v9/micro_kernels.h`; the math is proven bit-exact in `/tmp/v9_intexact.py`:

```bash
python UPVM_ASR_C_EMBEDDED/model_pytorch/calibrate_v9.py                 # -> calib_v9_s_ln.json (needs torch + Revision model)
python UPVM_ASR_C_EMBEDDED/model_pytorch/weight_extractor_sq_int8_v9.py  # V8 bytes + int LN tables + GELU LUT
```

---

## 6. Performance Verification

`sources/helpers/performance_verify.py` compares generated audio against the original
clean references (`data/test_audio/original/`) using the metrics in
`model_pytorch/metric.py` (SNR, LSD, LSD_HF, LSD_LF) with tester-protocol alignment.

```bash
# Set VERSION="..." at the top of the script to pick the generated folder, then:
python sources/helpers/performance_verify.py
```

Outputs: `results/<version>/performance_verify.txt` and `..._per_file.csv`.

---

## 7. Quantized Path (W8A8) — How `sq_int8_v0` Works

Mixed-precision, symmetric W8A8 PTQ (SmoothQuant-migrated, weight zero-point = 0):

- **INT8 layers:** conv / linear (in_proj, out_proj, depthwise conv, mlp fc1/fc2,
  patch-embed convs, skip handlers) run in INT8 with an INT32 accumulator.
- **F32 islands:** layer norms, activations (GELU/ReLU), the SSM selective-scan
  core, cross-scan/merge, gating, and residuals stay float32.
- **Per quantized layer:** `f32 → quant (per-input-channel a_scale, int8)` →
  `int8 matmul/conv (int32 acc, folded int32 bias)` →
  `dequant (per-output-channel scale → f32)`. Scales are frozen at export time,
  so kernels carry no runtime fallback/null branches.

Validated against the PyTorch W8A8 reference: C output ≈ SNR 22.8 dB / LSD 0.455
vs clean refs, tracking the reference (SNR 23.4 / LSD 0.48) within ~0.6 dB.
See `sources/sq_int8_v0/IMPLEMENTATION_NOTES.md` and `sources/sq_int8_v0/PROGRESS.md`
for the full kernel/buffer design and validation details.

### 7.1 `sq_int8_v1` — fused int32 MLP ReLU

V1 is V0 with one micro-optimization in every MLP block (encoder / latent / decoder /
output). The MLP `fc1` activation is reordered:

- **V0:** `int8 conv → dequant (int32→f32) → relu_f32`
- **V1:** `int8 conv → relu_int32 (clamp the int32 accumulator) → dequant`

This is **bit-identical**, not approximate: dequant multiplies by a per-output-channel
scale `deq_scale[oc] = a_scale · a_scale_mul · w_scale[oc] > 0`, and for a positive
scale `relu(acc · s) = relu(acc) · s` exactly. Running the ReLU ahead of dequant keeps
it on the int8/int32 (accelerator) side and drops the negative-half f32 write. Only the
MLP `fc1` ReLU is fused; the SSM-branch ReLUs stay f32. New kernel:
`sources/sq_int8_v1/kernels/relu_int32.c` (declared in
`headers/sq_int8_v1/micro_kernels.h`). The `MEM_PROFILE` report accordingly tags the
fused ReLU as an `INT8-ACCEL` int32 op.

### 7.2 `sq_int8_v2` — fused int32 encoder MLP SumPool

V2 is V1 plus one more fusion, in the **encoder** MLP blocks only (`pvss_ds`, enc0/1/2).
The MLP "SumPool" is a depthwise 2×2 stride-2 conv whose weights are **exactly 1.0** in
this checkpoint (a pure sum of 4 neighbours), so the main MLP `fc1` path becomes:

- **V1:** `int8 conv → relu_int32 → dequant → f32 SumPool` (sum 4 dequantised values)
- **V2:** `int8 conv → relu_int32 → SumPool (sum 4 int32 accs, in place) → dequant`

Because `deq_scale[oc] > 0`, the scale factors out of the sum, so the pool is exact in
int32 (4·max acc ≈ a few M, far below `INT32_MAX`). Doing it before dequant keeps it on
the int8/int32 (accelerator) side, reuses `acc_buf` in place, and shrinks the dequant
4× (it now runs at H/2·W/2). Only the **main** MLP SumPool is fused; the **residual**
SumPool stays f32 (it has no quantised counterpart). The pool runs through a new general
kernel `sources/sq_int8_v2/kernels/depthwise_conv2d_int32.c` (the int32 sibling of
`depthwise_conv2d_f32`, same `k`/`stride`/`pad` signature), invoked with `weight=NULL`
= unit taps — mirroring how the f32 build called `depthwise_conv2d_f32` with the all-ones
`mlp_sumpool_w`.

Unlike V1↔V0, **V2 is not bit-identical** to V1: V1 rounds 4 f32 products then adds in
f32; V2 sums in exact int32 then rounds once at dequant (V2 is the *more* accurate of the
two). The end-to-end effect is negligible — SNR(V2 vs V0) ≈ 63 dB, ~40 dB below the
model's own ~22.8 dB error floor, so SNR/LSD versus the clean references is unchanged.

### 7.3 `sq_int8_v3` — fused int32 SSM `x` ReLU

V3 is V2 plus one more fusion, in the SSM branch of the **encoder / latent / decoder**
blocks (`pvss_ds` / `pvss_latent` / `pvss_us`). After the int8 depthwise conv on `x`
there is a ReLU before CrossScan; V3 moves it ahead of the dequant:

- **V2:** `int8 dw-conv → dequant (int32→f32) → relu_f32(x)`
- **V3:** `int8 dw-conv → relu_int32(x) (on the accumulator) → dequant`

Like the V1 MLP-ReLU fusion this is **bit-identical** (`deq_scale[oc] > 0` ⇒
`relu(acc·s) = relu(acc)·s`), reuses the existing `acc_slice` int32 scratch (no extra
buffer), and lets the ReLU run int32 on the accelerator side. Only the `x` ReLU is fused;
the `z` ReLU stays f32 (its source is the in-proj dequant), and the output stages
(`output0/1/3`) keep their f32 SS2D unchanged (no int8 dw-conv accumulator there). No new
kernel — it reuses `relu_int32` from V1. Verified bit-identical to V2 end-to-end
(SNR(V3 vs V0) = 63.1 dB, same as V2).

### 7.4 `sq_int8_v4` — fused MLP `fc1`→`fc2` REQUANT

Between the two int8 MLP layers, V0–V3 cross the int8 boundary twice — the `fc1`
**dequant** (int32→f32) immediately feeds the `fc2` **quant** (f32→int8), with the
**same channel count** on both sides (fc1 out-channel == fc2 in-channel). V4 fuses that
`dequant → quant` pair into one int32→int8 **REQUANT** per channel `c`:

- **V0–V3:** `f = (float)acc·deq[c]` ; `q = clamp(lrintf(f / act_next[c]), -128, 127)`
- **V4:** `q = clamp(lrintf((float)acc · requant_scale[c]), -128, 127)`,
  with `requant_scale[c] = deq[c] / act_next[c]` precomputed by the extractor.

Algebraically the two are the same map `acc ↦ acc·deq/act_next` followed by round+clamp;
V4 just folds the constant and removes the intermediate f32 round-trip (one fewer f32
rounding, no f32 staging buffer — the new int8 lands straight in `quant_buf`, so `fc2`
skips its quant entirely). It is applied in the **encoder / latent / decoder** MLPs
(`pvss_ds` after the int32 ReLU+SumPool, `pvss_latent` / `pvss_us` after the int32 ReLU);
the **output** stage is intentionally left on the V3 path.

This is the **first int8 version that changes a scale**: the kernel
`sources/sq_int8_v4/kernels/requant_int32_to_int8_perchannel.c` reads the new per-stage
`mlp_requant_scale` (loaded as a plain f32 tensor; lrintf + clamp identical to the quant
kernel). Like V2 it is **not bit-identical** — dropping one f32 rounding flips a few
rounding ties — but the divergence is bounded to **±1 LSB** and empirically `< 3e-6` of
values, far below the model's ~22.8 dB error floor. A later step can replace the f32
multiply with an integer `mul`+`shift` requant for a fully int path.

### 7.5 `sq_int8_v5` — fused SSM in_proj→dw_conv2d REQUANT (`x` branch)

The same `dequant → quant` boundary appears inside the SSM, between the int8 in-projection
(`PW-CONV-2D`) and the int8 depthwise conv (`DW-CONV-2D`). V5 fuses it with the **same
REQUANT kernel as V4** — no new kernel, just V4's `requant_int32_to_int8_perchannel`:

- **V0–V4:** `in_proj: int8 conv → dequant(int32→f32)` ; `dw_conv: quant(f32→int8) → int8 conv`
- **V5:** `in_proj: int8 conv → requant(int32→int8)` straight into `q_slice`; `dw_conv` reads it directly

with `ssm_requant_scale[c] = in_proj.deq_scale[c] / dw_conv.act_scale[c]` precomputed by
the extractor. The one wrinkle vs. the MLP: the in-projection emits `[x | z]` =
`2·SUB_D_INNER` channels but **only the `x` half (`[0, SUB_D_INNER)`) feeds the depthwise
conv**, so the dequant is *split* — `x` is requantized to int8 while `z` keeps its f32
dequant (it is consumed by the f32 gate / scan core). `requant_scale` therefore uses
`in_proj.deq_scale[:SUB_D_INNER]`, the channels shared with `dw_conv.act_scale`.

**No buffer growth:** the requantized `x` lands in the very `q_slice` (int8 scratch) that
the dw-conv quant used to fill, so the footprint is identical to V4 — the fusion only
*removes* the f32 staging of `x`. Applied in **encoder / latent / decoder** only; the SSM
**out-projection** (no immediately-following int8 quant) and the **output** stages (f32
SS2D) are untouched. The V3 `relu_int32(x)` on the dw-conv accumulator is preserved.

Like V4 it is **not bit-identical** — one fewer f32 rounding flips a few ties, bounded to
**±1 LSB**, empirically `5.7e-6` of values (per-stage extractor self-check) — well below
the error floor.

### 7.6 `sq_int8_v6` — integer-only (fixed-point) MLP REQUANT

V0–V5 run the MLP REQUANT in **floating point**: `q = clamp(lrintf((float)acc · scale[c]))`
(one f32 multiply + `lrintf`). V6 makes it **integer-only** with the gemmlowp / TFLite
*fixed-point ("quantized") multiplier* technique: a real per-channel multiplier `M` (here
`M = mlp_requant_scale[c] = fc1.deq / fc2.act`) is approximated by a dyadic rational
`M ≈ mul[c] / 2^shift[c]`, so

> **`q[c,l] = clamp( ( (int64)acc[c,l] · mul[c]  +  (1 << (shift[c]-1)) ) >> shift[c],  -128, 127 )`**

— one integer **multiply**, one rounding **add**, one arithmetic **shift**, no f32 / `lrintf`
at runtime (new kernel `requant_int32_to_int8_perchannel_fixed`). The product is taken in
int64 (`acc, mul < 2^31` ⇒ `< 2^62`, no overflow); the `+ (1<<(shift-1))` gives round-half-up.

**Choosing `(mul, shift)` — the only hyperparameters.** `weight_extractor_sq_int8_v6.py`
**searches every shift** `s ∈ [1, 30]` per channel, sets `mul = round(M · 2^s)` (rejecting
`mul ∉ (0, 2^31)`), and keeps the `s` minimizing the integer-vs-f32 mismatch over a
representative band of int32 accumulators (tie-break: smaller LSB error, then larger `s` =
more multiplier precision). This is the canonical gemmlowp `frexp` choice (largest shift
keeping `mul < 2^31`, typically `s≈30`) but *verified by the actual error*, not assumed. The
pair is stored as int32 `mlp_requant_mul[MLPh]` / `mlp_requant_shift[MLPh]` per stage and
loaded alongside the existing weights (no buffer/interface change; the f32 `mlp_requant_scale`
is kept for reference only).

Applied to the **encoder / latent / decoder** MLPs (the 14 fused stages). The **SSM** requant
stays the f32 V5 kernel, and the **output** stages have no requant (they are on the unfused
quant→conv→dequant path — there is no MLP requant to convert there). **Not bit-identical** to
the f32 requant — the dyadic approximation flips a few rounding ties — but bounded to **±1 LSB**,
empirically `7.4e-6` of values (per-stage extractor self-check). A natural follow-up is to give
the SSM requant the same integer treatment.

### 7.7 `sq_int8_v7` — integer-only (fixed-point) SSM REQUANT

`sq_int8_v7` applies the exact §7.6 technique to the **SSM** `in_proj → dw_conv2d` requant on the
`x` branch (the V5 fusion), so that requant is now integer-only too:

> **`q[c,l] = clamp( ( (int64)acc[c,l] · ssm_requant_mul[c]  +  (1 << (shift[c]-1)) ) >> shift[c],  -128, 127 )`**

with `ssm_requant_scale[c] = in_proj.deq(x) / dw_conv.act ≈ ssm_requant_mul[c] / 2^ssm_requant_shift[c]`.
Same searched `(mul, shift)` (`weight_extractor_sq_int8_v7.py`, shift over `[1,30]`), the **same kernel**
`requant_int32_to_int8_perchannel_fixed` (reused — no new kernel), stored as int32 `ssm_requant_mul[D]`
/ `ssm_requant_shift[D]` per stage. Applied to the **encoder / latent / decoder** SSM stages; the `z`
branch keeps its f32 dequant and the **output** stages stay f32 (no requant). **Not bit-identical** to
the f32 SSM requant — bounded to **±1 LSB**. Together with V6 this leaves no float in either the MLP or
SSM requant path; the remaining floats are the LayerNorms, the SSM selective-scan core, and the
(deliberately untouched) output stages. §7.8 (planned) tackles the first of these — the patch-embed
LayerNorms.

### 7.8 `sq_int8_v8` — PatchEmbed LayerNorm via piece-wise-linear (PWL+LUT) rsqrt

The Patch Embed (see model diagram) runs, per conv group, `QUANT → CONV-2D → DE-QUANT → LN(*)`
(group 1 adds `→ GELU`), then `+ INTERACTION`. V0–V7 run each `LN` in **f32** on the dequantized conv
output, using `inv_std = 1/sqrtf(var+eps)`. V8 replaces the transcendental `sqrtf` with a **piece-wise
-linear LUT** — the FPGA/ASIC-friendly **O(1)** form — and **fuses the conv dequant into the norm**, so
the two PatchEmbed norms (`norm1`, `norm2`, both streams) consume the int8 conv's **int32 accumulator
directly** with no separate dequant pass. New kernel
`sources/sq_int8_v8/kernels/layernorm_int32_pwl_rsqrt.c`; only the **two PatchEmbed LayerNorms** are
converted — the ~14 LayerNorms on the f32 residual stream (pvss_ds/latent/us, output stages) and the
`gelu` stay f32.

**Range-reduced PWL rsqrt (the only approximation).** `inv_std = 1/sqrt(t)`, `t = var+eps > 0`:
> `frexpf(t)` → `t = mant·2^e2`, `mant ∈ [0.5,1)`; normalise to `m' ∈ [1,4)` with an **even** exponent
> `2k` (`e2` odd → `m'=mant·2, k=(e2-1)/2`; `e2` even → `m'=mant·4, k=(e2-2)/2`); then
> **`1/sqrt(t) = 2^-k · ( slope[i]·m' + intercept[i] )`**, segment `i` chosen by `m'`.

`frexpf`/`ldexpf` are an exponent read + a shift (a leading-zero-count + barrel shift in hardware); the
PWL eval is one multiply + one add; the segment lookup is a small comparator chain — **no sqrt, no
divide**. Because the mantissa is always reduced to `[1,4)`, **one small table covers every variance
magnitude** (no per-layer range calibration, no fragility near `v→0`).

**Why the dequant fuses but the multiply stays.** LayerNorm normalization is invariant to a positive
*per-tensor* scale, but this model's conv dequant is **per-output-channel** and the norm reduces over
the channel axis, so the per-channel `deq_scale[c]` does **not** cancel. V8 therefore keeps the
per-channel multiply but folds it *into the norm's first read* (`x[c] = acc[c]·deq[c]`), eliminating
the separate `DE-QUANT` **pass/buffer**, not the arithmetic. Mean/variance (double accumulation) and
that dequant multiply are **identical to the f32 path**, so V8 diverges from V7 *solely* by the PWL
rsqrt.

**Tuning (`weight_extractor_sq_int8_v8.py`).** Knots are placed **adaptively** on `[1,4)` (density
∝ `|f''|^{1/2}` ∝ `x^{-5/4}`, denser where the curve bends) and each segment is a **min-max (Chebyshev)
linear fit** of rsqrt. `_suggest_segments(tol)` sweeps the count and picks the **fewest** segments whose
worst-case relative error ≤ `1.5e-3`; range reduction makes this one count serve **every** LayerNorm
(the "max rule" — the table is data-independent). The result is **`PWL_RSQRT_SEGMENTS = 10`**
(`headers/sq_int8_v8/datatypes.h`; the load-time file-size check enforces it matches the extractor).
The table — `knot[11]`, `slope[10]`, `intercept[10]` (f32) — is written into **both** embed stages and
loaded through the existing per-stage spec mechanism, so **no buffer or interface changes** (the
`patch_embed_int8` signature is unchanged; the table rides inside `struct embed_weight`).

**Not bit-identical** to the f32 LayerNorm (the PWL flips a few low bits), but bounded — extractor
self-check: worst-case rsqrt rel-err `1.29e-3`, end-to-end **LayerNorm SNR ≈ 63.8 dB**, far above the
model's ~22.8 dB floor and in line with V4/V6/V7. Code delta vs V7 = 6 files + 1 new kernel
(datatypes.h, micro_kernels.h, weight_loader, patch_embed_int8.c, model_memory/model.c, main.h);
weights byte-identical to V7 plus the 6 PWL files. All 4 modes compile **and link** clean against FFTW.

> *Design background (alternatives considered).* The hardware-norm literature also offers
> **Newton/Newton-shift** integer sqrt (I-BERT `I←(I+Var//I)>>1`; I-ViT shift-Newton — exact, no table,
> but needs an integer **divider**) and **LUT-seeded 1-step NR rsqrt**. For LN's sqrt the field often
> leans Newton; V8 uses **PWL+LUT** by design choice — fixed O(1) latency, divider-free, multiplier +
> small ROM — which suits the FPGA/ASIC target. Refs: I-BERT (arXiv 2101.01321), I-ViT (arXiv
> 2207.01405, ICCV 2023); HW LayerNorm-approx accelerators (MDPI Electronics 2025; SOLE arXiv 2510.17189).
> A *fully* integer patch-embed would also integerise `gelu` (I-BERT i-GELU / I-ViT ShiftGELU); V8
> stops at the LayerNorm rsqrt (the `gelu` stays f32).

### 7.9 `sq_int8_v9` — fully-integer PatchEmbed LayerNorm (int8 out) + int8 GELU LUT

V9 finishes the patch-embed integer path: `conv1 → I-LayerNorm(int32→int8 @ S_ln) → int8 GELU LUT →
conv2(int8) → I-LayerNorm(int32→f32)`. Both PatchEmbed LayerNorms now run **fully integer** (int64
mean/variance, a **fixed-point** range-reduced rsqrt LUT returning a (mantissa, exponent) pair, and a
fixed-point affine — no f32/double in the datapath); the only f32 left is `norm2`'s output (for the f32
encoder residual). `norm1` emits **int8** at a **calibrated** scale `S_ln` (per stream, from
`calibrate_v9.py` — a forward hook on `norm1` over real wavs), feeding an **int8 GELU** that is a
**per-channel 256-entry LUT** (`gelu_lut[c][q]=clamp(round(GELU(q·S_ln)/conv2.act_scale[c]))`, absorbing
dequant+GELU+quant); `conv2` then reads that int8 directly (its own quant is dropped). Kernels:
`sources/sq_int8_v9/kernels/layernorm_int_q.c` (`..._to_int8`/`..._to_f32`) and `gelu_int8_lut.c`.

The norm1→GELU→conv2 chain runs in int8 through the existing `quant_buf` (no buffer growth; the f32
`embed_act_buf` is unused in v9). Fixed-point Q-formats (`LN_QM/QC/QG/QN/QY`, `LN_SEG`, `LN_EPSP`) are in
`headers/sq_int8_v9/micro_kernels.h` and **must match** `weight_extractor_sq_int8_v9.py`; the design is
proven bit-exact in `/tmp/v9_intexact.py` (norm2 ≈64 dB vs f32 = the V8 rsqrt ceiling; norm1 int8 at the
~50 dB int8 floor; every stage int64-safe). **Not bit-identical** to V8. Weights = V8 byte-identical +
the int fixed-point LN tables + per-channel GELU LUT. All 4 modes compile **and link** clean.

The remainder of this section is the **design reasoning** (the int32/int8 interface and why int8 GELU is
precise enough) that motivated the above.

**The width rule.** In W8A8, **int32 appears only at conv/GEMM accumulators** (a sum of int8×int8
products genuinely needs 32 bits); **every activation wire *between* ops is int8**. So an op's
interface width is set by its neighbours, not chosen:

```
conv1 → [int32 acc] → I-LN → [int8 act] → I-GELU → [int8 act] → conv2 → [int32 acc] → …
        ^^accumulator^^      ^^^^^^^^^^ activations ^^^^^^^^^^      ^^accumulator^^
```

- **LayerNorm is int32-IN** because it eats conv1's *accumulator* (and wants that width for the
  variance estimate) and **int8-OUT** because it emits a normalised *activation*.
- **GELU is int8-IN/OUT** because it is a pure activation→activation op. Decisively, **`conv2` is an
  int8 GEMM, so its operands must be int8** — GELU sits immediately before `conv2`, so GELU's output
  is int8 *by necessity*. There is no int32 accumulator near GELU to consume; LN already drained it.
  (This is why the int8/int32 split looks asymmetric but isn't a preference.)

**Is int8 GELU precise enough? Yes.** GELU is a smooth, bounded pointwise nonlinearity, and:
- A **256-entry int8→int8 LUT** makes the GELU map itself *exact for the chosen scales* — the only
  error is the int8 activation quantisation, i.e. the **same noise floor as every other activation**
  in the W8A8 model, not an added approximation. (One BRAM ROM, O(1) — FPGA-cheap.)
- GELU **does not amplify** that noise: `|GELU′| ≲ 1.13`, and it saturates toward 0 for `x<0`, so input
  quant error is passed through (or shrunk), never magnified.
- The int8 noise at GELU's *output* is **already present in V8** — today `gelu` runs f32 but `conv2`
  immediately requantises its output to int8. Going int8-GELU only newly quantises GELU's *input*
  (= the LN int8 output you produce anyway), a small increment well under the model's ~22.8 dB floor.
- Published integer transformers (I-BERT **i-GELU**, I-ViT **ShiftGELU**) report **<~0.3 %** end-to-end
  drop from fully-integer GELU.

**Recommended V9 kernel:** a 256-entry **int8→int8 LUT** GELU (simplest, exact-for-scales, single ROM);
`ShiftGELU` (shifts) / `i-GELU` (erf polynomial) are the table-free alternatives. **Escape hatch** if a
measured accuracy need appears: carry **int16 fixed-point** on the `LN-out → GELU` segment and quantise
to int8 only at `conv2`'s input (more mantissa through the nonlinearity; you must hit int8 before
`conv2` regardless). The V9 shape is then `conv1 → I-LN(int32→int8) → I-GELU(int8→int8) → conv2(int8)` —
fully integer, **no standalone dequant/quant/requant** between the ops (each nonlinear kernel folds its
own input dequant + output quant, exactly as V8's LN already folds its input dequant).

### 7.10 `sq_int8_v10` — AVX-VNNI vectorized int8 kernels (bit-exact to V9)

V0–V9's pure-int8 kernels are **scalar** naive ports of the f32 code, so the int8 path was
~5–8% *slower* than f32 on CPU (it pays the quant/dequant cost but uses none of the hardware's
int8 throughput; see §11). V10 leaves the math **exactly** as V9 and only vectorizes the hot int8
kernels with **AVX-VNNI** (`vpdpbusd`, 256-bit VEX). It is **numerically identical to V9** — same
weights (`data/weight_sq_int8_v10/` is byte-identical to V9), same SNR; **only latency changes**.
Every kernel keeps its signature, buffers, and call sites — the change is internal.

Vectorized kernels (each keeps the original scalar body as a fallback under `#else`):

- **`pointwise_conv2d_int8` / `pointwise_conv2d_split2_int8`** (MLP `fc1`/`fc2`, SS2D `in_proj`/`out_proj`
  — the dominant int8 GEMM): activations are `[C_IN, L]` (channel-major), so the contraction dim is
  strided; `vpdpbusd` needs 4 contiguous contraction bytes per lane, so an 8-pixel block is transposed
  into an **on-stack** tile `[C_IN/4][8][4]` (no model buffer touched). `vpdpbusd` is `u8×s8` but
  activations are signed, so `x` is biased by `+128` (→ `uint8`) and the per-output term
  `128·Σ_c w[o,c]` is subtracted back:
  `Σ_c (x[c]+128)·w[o,c] − 128·Σ_c w[o,c] = Σ_c x[c]·w[o,c]` → **integer-exact**, identical to the
  scalar int32 accumulator. Tail pixels (`L % 8`) use the scalar body.
- **`depthwise_conv2d_int8`** (SSM dw-conv, stride-1 same-pad): no cross-channel contraction, so
  AVX2 widening (`vpmovsxbd`/`vpmulld`/`vpaddd`) over the contiguous width `W`, accumulating each
  output row in place over the `k×k` taps (integer add is associative ⇒ bit-exact).
- **`depthwise_conv2d_int32`** (encoder MLP sumpool): AVX2 for the `stride==1` case; the real use is
  the **stride-2** 2×2 sumpool whose output columns are strided in the input (and which may alias
  `y==x`), so that path keeps the exact scalar kernel.

**Build:** the Makefile compiles **only the four vectorized int8 kernels** with
`-mavx2 -mfma -mavxvnni`; the rest of v10 keeps v9's exact codegen, so the **f32 datapath stays
bit-identical to v9** (no auto-vectorization / FMA-contraction drift) and no other version is
affected. It deliberately does **not** use `-march=native`: the int8 targets are the
**i7-12700K** (Alder Lake, *no* AVX-512) and **Ryzen 9700X** (Zen 5, has AVX-512), so 256-bit
**AVX-VNNI** is the common ISA across both; `-march=native` on an AVX-512 box would `SIGILL` on the
12700K. Compiled without `-mavxvnni`, the kernels fall back to scalar and stay correct. ARM
Cortex-A55 is a later port (NEON `sdot`, 128-bit — same tiling, narrower).

**Verification:** each kernel was checked **bit-exact vs the scalar reference** on randomized int8
inputs across many shapes (non-multiples of 8/4, `L=1` linear layers, large `C_IN`, all kernel
sizes, the split boundary, and the stride-2 sumpool). Build & run with
`VERSION=SQ_INT8_V10 MODE=DETAIL_STAGE_TIMING make` to compare latency against V9.

---

## 8. Project Layout

```
UPVM_ASR_C_EMBEDDED/
├── Makefile                 # VERSION / MODE driven build
├── headers/<version>/       # per-version headers (+ headers/helpers/)
├── sources/<version>/       # common/ kernels/ model/ (model_timing/ where applicable)
├── sources/helpers/         # shared audio I/O + performance_verify.py
├── main/<version>/main.c    # per-version host entry
├── model_pytorch/           # checkpoints (model/) + extractors + metric.py
├── data/weight_*/           # exported weights consumed by the C loader
├── data/test_audio/         # downsample/ (input), original/ (clean), generated/<version>/
└── results/<version>/       # timing + verification reports
```

---

## 9. Troubleshooting

- **FFTW link errors:** ensure the system provides `libfftw3f`, or point
  `LOCAL_FFTW_PREFIX` at a local FFTW build (default `../fftw3/local`).
- **No generated output:** confirm input wavs exist in `data/test_audio/downsample/`
  and the version's weights dir (`data/weight_f32/` or `data/weight_sq_int8_v0/`)
  is populated by the matching extractor.
- **Performance script reports missing files:** make `VERSION` in the script match
  the generated folder you ran.

---

## 10. Hardware Accelerator Feasibility (FPGA / ASIC) — Design Notes

> Forward-looking analysis for porting inference to a custom accelerator. Values are
> **MEASURED** where marked (from `results/sq_int8_v9/`); all others are **first-order
> estimates** calibrated to the published per-operation energy and device-resource
> figures cited in §10.6. They are pre-synthesis envelopes (±2–3×), **not**
> place-&-route or silicon results.

### 10.1 Design constraint
- Target: hearing-aid-class device. **Whole model must complete in < 150 ms** per
  processing segment (latency budget consistent with ITU-T G.114 [1]; note a true
  in-ear path is far tighter, ~10 ms [2] — the 150 ms here is the adopted *compute*
  budget, separate from segment-buffering/algorithmic latency).
- One segment = **2.555 s** of audio; the full clip (`AUDIO_LENGTH=122640` @ 16 kHz
  ≈ 7.66 s) ≈ 3 segments processed sequentially.
- **MEASURED CPU baseline: 285 ms / segment** (`results/sq_int8_v9/processing_time.txt`,
  mean over 3,659 segments) → **1.9× over budget**. The accelerator must close this gap.

### 10.2 Where the time goes (MEASURED)
Mean per-stage time of the SS2D (`*_pvss_*`) branch, magnitude+phase summed, over all
3,659 segments of `results/sq_int8_v9/detail_stage_timing.csv`:

| Stage | ms | | Stage | ms |
|---|---:|---|---|---:|
| Encoder 0 | 40.3 | | Decoder 2 | 13.4 |
| Encoder 1 | 14.1 | | Output 0 | 22.1 |
| Encoder 2 | 7.4 | | **Output 1** | **47.0** |
| Latent | 3.2 | | **Output 3** | **73.3** |
| Decoder 0 | 3.2 | | | |
| Decoder 1 | 7.3 | | **SS2D total** | **231.3** |

- **SS2D is 81 % of the 285 ms total** → the primary offload target.
- Output 3 (73 ms) and Output 1 (47 ms) dominate: they are ungrouped / low-group
  (`reduce_factor` = 1 / 4), so `INNER_MP_PARALLEL` cannot spread them — on CPU they
  run near-serial over long `L` (262 144 / 65 536) with scalar `expf`. An accelerator
  that pipelines the recurrence at lane level (and optionally uses a parallel scan)
  removes exactly this bottleneck.

### 10.3 Proposed accelerator module inventory
- **STFT / iSTFT front+back end**: 1024-pt FFT/IFFT [10], Hann windowing, overlap-add,
  and polar SFUs — `sqrt` (magnitude), `atan2` (phase), `sin`/`cos` (synthesis),
  `log2` (spectrogram compression).
- **INT8 MAC array** for all conv/linear (patch-embed, in/out-proj, fc1/fc2, DW conv,
  interaction); int8 DSP packing gives ~2 MAC/DSP on Xilinx [8].
- **Quant / Dequant / Requant** (dyadic mul+shift, gemmlowp/TFLite style [7]).
- **Integer LayerNorm** (§7.8–7.9) and **integer GELU LUT** (§7.9).
- **SS2D f32 coprocessor** (§10.4).
- **Vector ALU**: ReLU / SiLU (sigmoid reuses the exp SFU), residual add, SSM gate mul.
- **Up/down-samplers**: 2×2 stride-2 sum-pool (encoder) and pixel-shuffle/transpose
  (decoder/output).
- **Memory subsystem**: on-chip SRAM tiles + U-Net skip/residual buffers + DDR for
  weights/large activations + DMA. (Per §10.2 reasoning, keep SS2D operands on-chip.)
- **Governor**: an external MCU (PS on FPGA) for sequencing / DMA setup / irregular ops.

### 10.4 SS2D f32 coprocessor
The f32 island (`selective_scan_f32.c`, `cross_scan_f32.c`, `cross_merge_f32.c`,
`grouped_pointwise_conv1d*_f32.c`, `layernorm_f32.c`, `elemwise_mul_f32.c`) needs five
functional-unit types: **FP-FMA lanes**, an **exp/softplus SFU**, an **rsqrt SFU**
(reuses the §7.8 PWL design), a **hidden-state RAM**, and a **lane-interleaved scan
sequencer**. Two enablers make 100 MHz feasible:
- **Hoist transcendentals out of the recurrence**: `δ = softplus(dt)` and
  `a = exp(A·δ)` depend only on inputs/constants, not on state — precompute them in a
  fully pipelined pass so no `exp` sits on the serial critical path. (Mirrors Mamba's
  hardware-aware kernel [5].)
- **Lane interleaving**: with N=1 there are `K·D_inner` × groups independent state lanes
  (128 for Encoder 0), enough to keep a mul-add recurrence at II=1. For the low-group
  long-`L` output stages, a work-efficient **associative/parallel scan** [5][6] breaks
  the `O(L)` latency floor.

Op budget (both streams, per segment): SS2D ≈ **57 M `exp` + 255 M FMA ≈ 0.31 G f32-ops**;
int8 conv/MLP ≈ ~0.3 G MAC; STFT ≈ ~15 M ops.

### 10.5 Estimates — latency, power, area
**Latency** (whole model; int8 MAC work overlaps, SS2D is the long pole):

| Platform | Whole-model | vs 150 ms |
|---|---:|---:|
| FPGA ZCU104 @ 300 MHz | ~50–70 ms | ~2–3× margin |
| ASIC 28 nm @ ~400 MHz | ~35 ms | ~4× |
| ASIC 16 nm @ ~800 MHz | ~10–20 ms | ~7–15× |

**Power** — accelerator *core* only (excl. PS/MCU), from ~1.2 mJ/segment (28 nm) /
~0.6 mJ (16 nm), where energy ≈ Σ(ops × per-op energy) using Horowitz/Dally figures
[3][4] scaled to node:

| Node | Active (running ~20–35 ms) | Avg, power-gated between segments (~1 % duty) |
|---|---:|---:|
| 28 nm | ~60–110 mW | ~0.8–1.5 mW |
| 16 nm | ~30–50 mW | ~0.3–0.7 mW |

Energy is the invariant (≈ ops × pJ/op); parallelism/clock trade latency for power at
fixed energy. Per-segment **power-gating** is what makes the average hearing-aid-feasible;
16 nm FinFET wins on leakage during the ~99 % idle window.

**Area** (SRAM-dominated; target 5 mm²):

| Block | 28 nm | 16 nm |
|---|---:|---:|
| INT8 MAC array (32×32) | ~0.6 | ~0.2 |
| SS2D f32 coproc (modest) | ~0.7 | ~0.2 |
| STFT FFT + SFUs + quant + int-LN + GELU + vector | ~0.8 | ~0.25 |
| On-chip SRAM (~1 MB) | ~1.5 | ~0.5 |
| + routing/overhead (~30 %) | ~1.1 | ~0.35 |
| **Total (mm²)** | **~4.7 (tight)** | **~1.5–2 (roomy)** |

- **28 nm** fits 5 mm² only with a small (~1 MB) SRAM + aggressive DDR streaming — SRAM-bound.
- **16 nm** fits in ~2 mm², leaving headroom to keep U-Net skips on-chip (lower power) or
  add compute. **Recommended node.** Keep the MCU off-die (separate power domain).

ZCU104 (XCZU7EV: 1 728 DSP, 11.0 Mb BRAM + 27.0 Mb URAM ≈ 4.8 MB on-chip, 2 GB DDR4 [9])
is a sufficient prototyping target: every stage fits on-chip if the 4 scan directions are
processed serially into the merge accumulator (the only overflow case, Output 3 at 8 MB
f32, drops to 4 MB this way, or 2 MB in bf16, or streams from DDR4 at ~0.4 ms).

### 10.6 Assumptions & references
- Energy/op (int/fp add·mult, SRAM, DRAM) and "data-movement ≫ compute" basis:
  [3] M. Horowitz, *Computing's Energy Problem (and what we can do about it)*, IEEE ISSCC 2014;
  [4] W. J. Dally, Y. Turakhia, S. Han, *Domain-Specific Hardware Accelerators*, Comm. ACM 63(7), 2020.
- Accelerator dataflow, MAC-array & memory-hierarchy energy, roofline (compute- vs
  memory-bound): [11] V. Sze, Y.-H. Chen, T.-J. Yang, J. Emer, *Efficient Processing of
  Deep Neural Networks*, Proc. IEEE 105(12), 2017; [12] S. Williams, A. Waterman,
  D. Patterson, *Roofline*, Comm. ACM 52(4), 2009.
- Activation/nonlinearity as a LUT in a special-function unit beside the MAC array (not
  in it): [13] N. Jouppi et al., *In-Datacenter Performance Analysis of a TPU*, ISCA 2017;
  [14] NVDLA (NVIDIA Deep Learning Accelerator) — LUT/SDP unit, nvdla.org.
- Selective scan, hoisting transcendentals, work-efficient parallel scan:
  [5] A. Gu, T. Dao, *Mamba*, arXiv:2312.00752, 2023; [6] G. E. Blelloch, *Prefix Sums and
  Their Applications*, CMU-CS-90-190, 1990.
- Integer LayerNorm/GELU and dyadic requant (basis for §7.6–7.9 and the int modules):
  [7] B. Jacob et al., *Integer-Arithmetic-Only Inference*, CVPR 2018 (gemmlowp);
  [15] S. Kim et al., *I-BERT: Integer-only BERT Quantization*, ICML 2021.
- FPGA int8 DSP packing / device resources / FFT IP:
  [8] AMD-Xilinx, *Deep Learning with INT8 Optimization on Xilinx Devices*, WP486;
  [9] AMD-Xilinx, *Zynq UltraScale+ MPSoC Data Sheet*, DS891 (XCZU7EV / ZCU104);
  [10] AMD-Xilinx, *Fast Fourier Transform LogiCORE IP*, PG109.
- Latency budget: [1] ITU-T Rec. G.114, *One-way transmission time* (≤150 ms);
  [2] M. A. Stone, B. C. J. Moore, *Tolerable hearing-aid delays*, Ear & Hearing (≈10 ms).
- ASIC area/power are envelopes: per-op energy from [3] scaled across nodes by standard
  node-energy trends; SRAM area from published foundry/ISSCC 6T-bitcell disclosures
  (~0.13 µm²/bit @28 nm, ~0.07 µm²/bit @16 nm FinFET). Not from synthesis/P&R.

---

## 11. CPU Vectorization Roadmap (int8 VNNI + SS2D scan) and FPGA Portability — Design Notes

> Status: **§11.1 (int8 GEMM/depthwise) is now implemented in `SQ_INT8_V10`** — see §7.10.
> §11.2 (SS2D selective scan) and §11.3 (FPGA) remain design notes. Measurements below are from
> `SQ_INT8_V9`, `inner_8`, `results/sq_int8_v9/inner_8/detail_stage_timing_summ.log`
> (n=3659, ~284 ms/segment).

### 11.1 Why the current int8 path is *slower* than f32 (and what fixes it)
- Measured: int8 (V0/V5) runs **~5–8% slower than f32** on CPU; the only real win so far
  is `INNER_MP_PARALLEL` threading, not the quantization itself.
- Root cause: the int8 kernels are **scalar** — the source tree has **0 SIMD intrinsics**
  (`grep -E 'immintrin|epi8|_mm512|vdotq|sdot' sources headers` → none) and the build has
  no `-march` flag. So we pay quant/dequant overhead and collect **none** of int8's
  compute benefit.
- The dev host exposes `avx512_vnni` **and** `avx_vnni` (the `vpdpbusd` int8×int8→int32
  fused 4-MAC dot-product). **Quantization on a CPU is a compute optimization that only
  materializes through int8 SIMD** — until VNNI is emitted, int8 is just "f32 with extra
  overhead." Vectorizing the int8 ops is the step that *justifies* quantization.

**Throughput ceiling.** f32 AVX-512 FMA = 16 lanes × 1 MAC. int8 VNNI `vpdpbusd` packs
64 int8/512-bit register and fuses a 4-element dot-product per lane → **~4× the MAC
throughput of f32 and ~4× less memory traffic.**

**Numerics are unchanged.** int8×int8 accumulates exactly into int32 (no overflow until
~130k terms, far beyond any `K` here), so SNR is identical — only faster. The V4/V5/V6/V7
REQUANT fusions (§7.4–7.7) were the prerequisite: they keep activations int8 across the
fused boundary so the whole chain can be vectorized without an f32 round-trip mid-pipeline.

**Where it lands (not a flat 4× on 284 ms):**

| Part | Shape | int8-vector win |
|------|-------|-----------------|
| MLP `fc1`/`fc2` (~109 ms single-branch) | GEMM | near 4× — maps directly to VNNI |
| SS2D `in_proj` / `out_proj` | GEMM | near 4× |
| SS2D selective scan (dw-conv + state recurrence) | sequential/elementwise | **does NOT use int8** (see §11.2) |

> `mag` and `pha` branches run concurrently under `INNER_MP_PARALLEL`; summing one branch's
> stage times ≈ `total_time`, so single-branch numbers ≈ wall-clock. SS2D ≈ 42% (~121 ms),
> MLP ≈ 38% (~109 ms) of the 284 ms.

First kernels to convert: write `_mm512_dpbusd_epi32` inner loops for MLP `fc1`/`fc2` and
SS2D `in_proj`/`out_proj`; build with `-mavx512vnni` / `-march=native` (without the flag
even existing loops won't auto-emit VNNI).

### 11.2 Vectorizing the SS2D selective scan on CPU (it stays float)
The scan kernel ([`selective_scan_f32.c`](sources/sq_int8_v9/kernels/selective_scan_f32.c))
is fully scalar with a per-element `expf` in the innermost loop — even in the int8 build.
That is correct (int8 in a recurrence is unsafe — see below) but it also gets **zero SIMD**
today. `output3` is ~40 ms and *almost pure scan*, which int8/VNNI cannot touch.

**Key fact: the recurrence `s[n] = s[n]*a + Δ·u·b` is sequential only in *time*; it is fully
independent across the `K·D·N` (direction × channel × state) axes.** That is the
SIMD/thread width.

1. **SIMD across channels `D` (best mapping).** Make the vector lane = channel. Then `a`,
   `s`, `u`, `Δ` are vector loads; `b`,`c` are **shared across `D` within a group** →
   broadcast scalars; `y_vec[d] += c·s_vec[d]` has **no horizontal reduction** (each lane is
   its own output row). Pure FMA, no cross-lane shuffles — ~16-wide on AVX-512.
2. **SIMD across state `N` (alternative).** The `n`-loop is elementwise too, but `y` reduces
   over `n` (horizontal sum each step) and `B`/`C` are strided by `L` (needs a transpose to
   `[G,L,N]`). Channel-major (option 1) avoids both.
3. **Vectorized `exp` — the real bottleneck.** Scalar `expf` is called `K·D·L·N` times and
   dominates even after the FMAs are vectorized. Replace with a vector polynomial/LUT exp
   (SLEEF / SVML via `-fveclib` / hand-rolled AVX-512). Range is small and `a∈(0,1)`, so a
   low-order minimax poly is accurate.
4. **(Optional) associative/blocked scan over time.** `s_t = a_t·s_{t-1}+b_t` is a
   first-order linear recurrence = an associative prefix scan ([6] Blelloch), so it *can* be
   parallelized along time. Only worth it where channel width runs out (e.g. the latent
   stage, small `D`); otherwise the `K·D·N` parallelism is plenty and lower-cost.

**Do NOT push the scan recurrence to int8.** The leaky accumulator `s = s·a + …` compounds
rounding over `L` steps and would hurt SNR — this is why every Mamba impl (and this one)
keeps the scan in f32 while quantizing only the projections. The scan's CPU lever is
**float SIMD across channels + vector exp**, not quantization. Expectation: ~**4–8×** on the
scan portion, which (with VNNI on the projections) is how SS2D's ~42% comes down toward the
150 ms ITU-T budget (§10.1).

### 11.3 Does the CPU vectorization affect a later FPGA SS2D scan?
**No — it forecloses nothing, and structurally it helps**, because CPU SIMD and an FPGA
scan exploit the *same* parallel axis. The decomposition is a property of the algorithm.

| CPU vectorization decision | FPGA equivalent |
|----------------------------|-----------------|
| Channel-major SIMD (lane = channel) | parallel PEs across channels (unroll = DSP budget) |
| Time = outer sequential loop | pipelined streaming dim; state feedback = the recurrence |
| `B`/`C` broadcast across `D` in a group | one value fanned out to all PEs → fewer mem ports |
| Transpose `B`/`C` to contiguous | on-chip BRAM banking / stream order |
| Per-lane output, no horizontal reduce | independent PE outputs, no reduction tree |

**Three divergence points to manage on purpose:**
1. **AVX-512 intrinsics** are a non-synthesizable leaf detail — only the *loop structure*
   transfers, not the `_mm512_*` calls.
2. **The `exp` approximation** is where CPU↔FPGA can be deliberately coupled: prototype the
   *FPGA-realistic* exp (LUT / PWL / CORDIC) on CPU now → the CPU becomes a near-bit-accurate
   **reference model**. Using a libm-grade poly on CPU but a coarse LUT on FPGA creates an
   SNR gap to re-debug later. (This mirrors the §7.8/§7.9 PWL/LUT approach already used for
   rsqrt/GELU.)
3. **Float vs fixed-point** is orthogonal and the bigger one. The CPU scan stays **f32**; an
   FPGA scan is often **fixed-point** (Qm.n) to save DSPs. The f32-SIMD work does **not**
   pre-validate fixed-point — the leaky-accumulator error-over-`L` reappears and needs a
   **separate fixed-point CPU reference model**. Vectorization doesn't block fixed-point, it
   just doesn't solve it.

**The one design fork — decide before vectorizing:**
- **FPGA via HLS** (Vitis/Intel): the refactor maps ~1:1 to pragmas (`PIPELINE` on time,
  `UNROLL`/`ARRAY_PARTITION` on channels, `B`/`C` as a stream) — **but AVX-512 intrinsics are
  not HLS-synthesizable.** If HLS is plausible, write the kernel as **clean auto-vectorizable
  plain loops** (`#pragma omp simd`, `-march=native`) so the *same source* feeds CPU auto-vec
  *and* HLS. Hand-rolled intrinsics fork the codebase.
- **FPGA via hand-written RTL:** no code transfers either way — only the algorithm/dataflow/
  numeric spec does; CPU vectorization is then a reference-model + baseline exercise, and
  intrinsics are fine.

**Reporting nuance.** A well-vectorized CPU baseline shrinks the *headline* FPGA speedup on
the scan (no more scalar strawman), so the "why FPGA" case leans on latency / power /
throughput-per-watt (consistent with §10.5) rather than raw scan speedup. This is a fairer,
more defensible comparison.
