# INT8 Mixed-Precision C Implementation — Progress Tracker
**Date:** 2026-06-10
**Checkpoint:** `model_pytorch/model/gen_ptq_int8pack.pth` (W8A8 PTQ, `ptq_int8_pack_v2`, source: logs_6)

---

## 0. Structure refactor — DONE ✅ (2026-06-10)

`sources/sq_int8_v0` now follows the same split-file scheme as `naives` / `naives_mp`:

```
common/        weight_loader_f32_i8.c, model_buf_init_f32_i8.c
kernels/       one micro kernel per file (13 F32 + 6 INT8/quant/dequant, 19 files
               after the Step-7 dead-kernel cleanup; see "Removed 2026-06-12")
model/         patch_embed_int8.c, pvss_ds_mp_int8.c, pvss_latent_mp_int8.c,
               pvss_us_mp_int8.c, vss_output0_mp_f32i8.c, vss_output1_mp_f32i8.c,
               vss_output3_f32i8.c   (Step 4 blocks), model.c (Step 5 int8 orchestrator)
model_timing/  pvss_ds_timing_mp_int8.c, pvss_latent_timing_mp_int8.c,
               pvss_us_timing_mp_int8.c, vss_output0_timing_mp_f32i8.c,
               vss_output1_timing_mp_f32i8.c, vss_output3_timing_f32i8.c,
               model.c (STAGE), model_detail.c (DETAIL)   (Step 8 int8 timing path)
```

Cleanups applied during the refactor (same scheme as naives/naives_mp):
- Naming standardized to the `conv2d` convention everywhere:
  `pointwise_conv2d_f32`, `pointwise_conv2d_split2_f32`,
  `pointwise_conv2d_int8`, `pointwise_conv2d_split2_int8`.
- Removed unused F32 kernels: `sumpool2x2_stride2_fp` (was also misnamed),
  `xz_chunk_f32`, `split_core_projection_f32`.
- `micro_kernels.h` fully re-synced with the implementations: missing INT8
  declarations added (`pointwise_conv2d_split2_int8`, `conv2d_int8`, quant/dequant),
  declarations without implementations removed (`grouped_pointwise_conv1d_int8`,
  `grouped_pointwise_conv1d_type_major_int8`, `sumpool2x2_stride2_int8`,
  `elemwise_add_int8`, `elemwise_mul_int8` — these layers stay float in the checkpoint).
- Makefile: `-fopenmp` now enabled for `sq_int8_v0` (was naives_mp-only).

---

## 1. Micro kernels (kernels/) — checkpoint coverage DONE ✅

Checkpoint re-verified on 2026-06-10 (110 qpacks, all `w_zp == 0` symmetric):

| Checkpoint layers | Count | Kernel (file) | Status |
|---|---|---|---|
| `conv2d` 1×1 pointwise | 36 | `pointwise_conv2d_int8.c` | ✅ |
| `linear` (call with H=W=1) | 34 | `pointwise_conv2d_int8.c` | ✅ |
| `conv2d` 1×1, c_in = 2·c_out (skip/out_proj) | 22 | `pointwise_conv2d_split2_int8.c` | ✅ |
| `conv2d` depthwise k=3 s=1 | 14 | `depthwise_conv2d_int8.c` | ✅ |
| `conv2d` regular k=3 s=2 (patch_embed) | 4 | `conv2d_int8.c` | ✅ |

Quant/dequant helpers:

| Kernel | Status | Notes |
|---|---|---|
| `quant_f32_to_int8_perchannel` | ✅ | per-channel `a_scale [C_IN]` — used for **all** 110 quantized layers (per-tensor handled as the C_IN==1 case; the per-tensor `quant_f32_to_int8` was removed 2026-06-12, see below) |
| `dequant_int32_to_f32_perchannel` | ✅ | scale_oc = a_scale_in · w_scale[oc] — every layer is per-channel |

### Removed 2026-06-10 — not required by this checkpoint

Pruned so `kernels/` holds only what the mixed-precision model actually uses
(22 files: 15 F32 float-island/compute + 7 INT8). Each removed kernel had **zero**
call sites in `model/`, `model_timing/`, `common/`:

| Removed | Why not required |
|---|---|
| `relu_int8` | activations run in **float** between dequant/quant — `relu_f32` is used |
| `xz_chunk_int8` | x/z chunk is pointer arithmetic on the buffer, no kernel (F32 version was already dropped) |
| `split_core_projection_int8` | `x_proj_conv` (dts/Bs/Cs) stays **float** (in `non_quant_state_dict`); split is pointer arithmetic |
| `dequant_int32_to_f32` | per-tensor dequant never occurs — all 110 layers have per-output-channel `w_scale`, so dequant is always per-channel |

(`silu_f32` is **kept**: `nn.SiLU` is the SSM activation in the PyTorch model —
`ssm_act_layer=nn.SiLU` in vmamba.py — even though the current C forward substitutes
`relu_f32` in the PVSS blocks.)

Float islands (stay F32, no INT8 version needed): `layernorm_f32`,
`selective_scan_f32`, `cross_scan_f32`, `cross_merge_f32`,
`grouped_pointwise_conv1d*_f32` (x_proj_conv), `elemwise_add/mul_f32`, activations.
Confirmed from `non_quant_state_dict`: all `SumPool`/`mlp_sumpool`/`skip_sumpool`
(learned stride-2 depthwise downsamplers), `skip_proj`/`skip_reduce`, `x_proj_conv`,
and `sampler.norm` weights stay float. Also note: `output_layer.*.ss2d.*`
(in_proj / conv2d / out_proj of the output blocks) are NOT quantized — only
their `mlp.fc*`, `sampler.expand`, and `skip_handler` are.

### Removed 2026-06-12 — dead after the int8 model was assembled (Steps 4–5)

Once the full mixed-precision path was wired, three kernels had **zero call sites**
in the built model (`model/` + `common/`, verified by grep) and were removed
(`kernels/` 22 → 19 files):

| Removed | Why dead in the int8 model |
|---|---|
| `conv2d_f32` | patch_embed now runs `conv2d_int8`; nothing else uses a regular f32 conv |
| `pointwise_conv2d_split2_f32` | the skip handlers now run `pointwise_conv2d_split2_int8` |
| `quant_f32_to_int8` (per-tensor) | superseded by always-`quant_f32_to_int8_perchannel` (per-tensor == C_IN==1 case, bit-identical) |

Kept despite being unused: **`silu_f32`** — `nn.SiLU` is the real PyTorch SSM
activation (`ssm_act_layer=nn.SiLU`), so it stays available for future exact-fidelity
work even though the current C forward substitutes `relu_f32` in the PVSS blocks.
(The f32 `pointwise_conv2d_f32` / `depthwise_conv2d_f32` are **kept** — live in the
output-stage ss2d and `mlp_fc_res`/sumpool.) Build re-verified clean afterward.

---

## 1b. Quantization runtime semantics — verified against `utils/ptq_w8a8.py` (2026-06-10)

The frozen forward of `QConv2dW8A8` / `QLinear2dW8A8` is:

```
x_div = x / sq_s[c]                       (sq_s = 1 for grouped/depthwise; already folded into stored weights)
x_q8  = fake_quant(x_div, a_scale[c] * a_scale_mul, zp=0, clamp [-128,127])
y     = conv(x_q8_dequantized, w_q8_dequantized) + bias_f32   (bias added in float)
```

`a_scale` is **per-input-channel [C_IN] for 104 of 110 qpacks** (per-tensor for 6).
`a_scale_mul` ≠ 1 for 20 layers (1.05 / 1.10). All zero-points are 0 (symmetric).

**Loader recipe (critical — plain int8 GEMM is wrong without it):**

- **Activation quant scales** (one fold, feeds `quant_f32_to_int8_perchannel`):
  `act_scales[c] = sq_s[c] * a_scale[c] * a_scale_mul`
- **Depthwise layers** (1:1 channel map — exact, no weight re-fold):
  dequant `scales[c] = w_scale[c] * a_scale[c] * a_scale_mul` (sq_s = 1 there).
- **groups=1 layers with per-channel a_scale** (pointwise / split2 / regular conv /
  linear): the per-channel act scale sits *inside* the accumulation, so the loader
  must re-fold it into the weights before the int8 GEMM is valid:
  1. `W_fp[oc,c] = weight_int8[oc,c] * w_scale[oc]`        (dequantize)
  2. `W~_fp[oc,c] = W_fp[oc,c] * a_scale[c] * a_scale_mul` (fold act scale)
  3. `w~_scale[oc] = max_c |W~_fp[oc,:]| / 127` ; `W~_q8 = round(W~_fp / w~_scale)`
  4. dequant `scales[oc] = w~_scale[oc]` ; `bias_i32[oc] = round(bias_f32[oc] / w~_scale[oc])`
- **Per-tensor a_scale layers** (6): standard —
  dequant `scales[oc] = a_scale * a_scale_mul * w_scale[oc]`,
  `bias_i32[oc] = round(bias_f32[oc] / scales[oc])`.

Rounding matches: `torch.fake_quantize_*` rounds half-to-even; `lrintf` does the
same under the default FP rounding mode. int32 overflow safe (max 2·256·127² ≈ 8.3M).

---

## 2. Remaining work to construct the mixed-precision model

The micro kernels (§1) are complete and sufficient — they are *necessary but not
sufficient*. The model still needs the scaffolding below to be assembled and run.
Each step lists its files, sub-tasks, what it depends on, and how to know it is done.

**Build order (dependencies):**

```
  Step 1 (int8 structs) ─┬─> Step 3 (loader) ─┐
  Step 2 (pth export)  ──┘                     ├─> Step 5 (model entry) ─> Step 6 (main) ─> Step 7 (validate)
  Step 1 ───────────────────> Step 4 (macro) ─┘
```

Steps 2 and 4 can be done in parallel once Step 1 fixes the struct shapes.

---

### Step 1 — INT8 weight structs — `headers/sq_int8_v0/datatypes.h`  ✅ DONE (2026-06-11, v3)
**Design: checkpoint-faithful, fully deterministic — no fallback, no runtime
switching.** Every struct is an exact compile-time image of its checkpoint block
(all 110 qpacks run INT8 in C); per-block structs absorb every deviation.

- `struct qconv_int8` = one **pure INT8** layer:
  `w_i8 / deq_scale [C_OUT] / bias_i32 [C_OUT]|NULL / act_scale / act_scale_len`.
  No `qconv_int8_output*` variant was needed — the output stage's quantized
  layers (fc1/fc2/expand/skip) use the same quad; its float layers are plain
  `const float*` members.
- **INT8-bearing structs** (quantized members are `struct qconv_int8`):
  `embed_weight` (`conv1`/`conv2`); `pvss_ds_weight` (ss2d trio, fc1, fc2);
  `pvss_latent_weight` (ss2d trio, fc1, fc2); `pvss_us_weight` —
  now **decoder-only** (ss2d trio, fc1, fc2, `pe`); `pvss_us_skip_weight`
  (all four skips, incl. `skip_conv_out0` — int8 in ckpt, bias).
- **Dedicated output structs** (replace the old shared use of `pvss_us_weight` /
  `vss_output_weight`), each matching `output_layer_*.{0,1,3}` exactly:
  - `vss_output0_weight`: ss2d all f32, **no block norms** (none in ckpt),
    `mlp_fc1`/`mlp_fc2` int8 (bias), `pe` int8 (no bias), pe_norm f32.
  - `vss_output1_weight`: same + `ssm_in_norm` / `mlp_norm` f32
    (ckpt `blocks.0.norm` / `norm2` — only out1 has them).
  - `vss_output3_weight`: `pre_conv2d` f32 (= ckpt `output_layer_*.2`, 1×1 4C→C),
    ss2d all f32, no block norms (Identity; the .c never used them),
    `mlp_fc1`/`mlp_fc2` int8 (bias).
- Float islands unchanged (`const float*`): norms, `ssm_dw_conv1d` (x_proj_conv),
  `ssm_dt_projection` (dt_proj_conv), A/Ds/delta_bias, `mlp_sumpool`, pe_norm,
  and the vestigial `mlp_dim_reduce_w` / `skip_reduce_w` dummies.
- Added `struct qconv_file_spec` (int8 sibling of `weight_file_spec`) for Step 3.
- Checkpoint↔member mapping pinned in comments. Non-obvious pair (encoder MLP),
  **corrected 2026-06-11 against weight_extractor.py**:
  C `mlp_fc2` ↔ ckpt `mlp.fc2.mlp_dim_redu` (main path, 1×1 C→2C, int8 no bias);
  C `mlp_fc_res` ↔ ckpt `skip_proj.skip_dim_redu` (residual path) — **f32 in the
  ckpt** (skip_proj excluded from PTQ), so `mlp_fc_res_w/_b` stay `const float*`.
  The quantized ckpt module `mlp.dim_reduce` (6 qpacks) is **dead** — never used
  by the C model (PTQ quantizes never-executed modules via its weight-stat
  fallback). ⇒ C consumes **104 of 110** qpacks.
- No-bias layers in ckpt: in_proj, out_proj, sampler.expand, encoder fc2
  → their `bias_i32` stays NULL.

**Verified:** header compiles standalone (`-Wall -Wextra`), all structs
instantiable; naives / naives_mp untouched (they keep their own
`vss_output_weight`). All kernels compile; `common/weight_loader_f32_i8.c` +
`common/model_buf_init_f32_i8.c` migrated in Step 3. The 12 awaiting migration
are the `model/` + `model_timing/` block files (Step 4). Per-block structs let Step 4
drop the `input_selection`/`norm_ena` runtime flags for the output blocks —
each gets its own deterministic function.

---

### Step 2 — pth → per-layer `.bin` export — `model_pytorch/weight_extractor_sq_int8_v0.py`  ✅ DONE (2026-06-11)
**Output:** `data/weight_sq_int8_v0/` — 24 stage dirs, **662 files**
(104 int8 weights · 60 int32 biases · 498 float32), plus per-stage `meta.csv`,
`manifest.csv`, `summary.json`. Full extraction patterns documented in
**`IMPLEMENTATION_NOTES.md`** (read that before starting Step 3).

- File pattern per int8 member `<m>`: `<m>_w.bin` (int8) ·
  `<m>_w_quant_scale.bin` (f32 `[C_IN]` or `[1]`) · `<m>_w_dequant_scale.bin`
  (f32 `[C_OUT]`) · `<m>_b_i32.bin` (int32, absent when no ckpt bias).
  Float islands keep the weight_f32 names/layouts (incl. `A = -exp(A_logs)`).
- §1b fold applied at export (loader does **no math**): depthwise/per-tensor
  exported as stored; groups=1 per-channel layers refolded + re-quantized per OC.
- Output-stage dirs renamed to the v3 structs: `vss_output0/1/3_weight_{mag,pha}`;
  no norm placeholder files (output1 is the only output block with real norms).
- **Validated in-script** against the PyTorch fake-quant reference (random inputs,
  one layer per class): activation integers match exactly; rel. error ≤ 3e-05
  non-refolded, ≤ 1e-02 refolded (weight re-quantization noise). 104/110 qpacks
  consumed; the 6 dead `mlp.dim_reduce` packs intentionally skipped (asserted).

---

### Step 3 — INT8-F32 weight loader — `common/weight_loader_f32_i8.c` (+ `headers/helpers/weight_loader_f32_i8.h`)  ✅ DONE (2026-06-11)
**Files:** `headers/helpers/weight_loader_f32_i8.h` (new, sq_int8_v0-only — the shared
`weight_loader.h` is untouched; `model.h` now includes the new header),
`common/weight_loader_f32_i8.c` (rewritten), `common/model_buf_init_f32_i8.c`
(call sites updated; function renamed `model_buf_init_f32` → `model_buf_init_f32_i8`).

- One generic `load_qconv_stage()` driven by `struct qconv_file_spec` arrays fills each
  `struct qconv_int8` from `<m>_w` / `<m>_w_quant_scale` / `<m>_w_dequant_scale` /
  `<m>_b_i32` (NULL when `has_bias == 0`); internals share a single
  `read_file_exact(path, dst, elem_size, numel)` + `alloc_bytes_or_die` — no per-dtype
  reader duplication. f32 members reuse the unchanged `load_weight_stage` path.
- Per-stage loaders for the v3 structs incl. `load_vss_output0/1/3_weight_stage`;
  decoder `load_pvss_us_weight_stage` dropped the `norm_dim` param (decoder-only now).
- **Validated** (test harness, gcc `-Wall -Wextra`): all 24 stages load with every file
  size checked; per-dir spec counts equal disk counts (331 files/stream × 2 = **662**,
  zero orphans); spot values (int8 w, quant/dequant scales, bias_i32, f32 islands,
  per-tensor act_scale_len=1 layers, NULL biases) byte-match numpy reads of the `.bin`s.

---

### Step 4 — INT8 macro kernels — `model/*_int8.c` / `*_f32i8.c` + `headers/sq_int8_v0/macro_kernels.h`  ✅ DONE (2026-06-11)
**Goal:** block-level functions that run the quantized path and leave float islands in
float. One per block, matching the v3 per-block structs — fully deterministic, no
`input_selection`/`norm_ena`-style precision flags.

- Every block has ≥1 int8 layer now, so all blocks get a mixed/int8 function:
  `patch_embed_int8` (conv1/conv2 int8), `pvss_ds_mp_int8`, `pvss_latent_mp_int8`,
  `pvss_us_mp_int8` (dec0..2, takes `pvss_us_weight`), and per-output-block
  `vss_output0_mp_f32i8`, `vss_output1_mp_f32i8`, `vss_output3_f32i8`
  (ss2d/SSM f32; only their fc1/fc2 [+pe in out0/out1] run int8).
- Per quantized layer the body is: `quant_f32_to_int8[_perchannel]` → int8 conv
  (`pointwise_conv2d_int8` / `pointwise_conv2d_split2_int8` / `depthwise_conv2d_int8`
  / `conv2d_int8`) → `dequant_int32_to_f32_perchannel`. Everything between stays float
  and calls the existing f32 kernels (norm, SSM, cross scan/merge, x_proj, residual
  add, gated mul, activations, SumPool).
- Skip handlers: dec0..2 + out0 call `pointwise_conv2d_split2_int8` via the
  `pvss_us_skip_weight` qconv members (wired in the model orchestrator, Step 5).
- Add the prototypes to `macro_kernels.h` (replace the `/* --- INT8 -- */` stub and
  the now-stale `pvss_us_mp_f32` / `vss_output3_f32` declarations as blocks migrate).

**Depends on:** Step 1 (struct types in signatures). **Done when:** each block compiles
and, fed dequantized==float inputs, its output tracks the `*_f32` block within int8
tolerance on a single-block test.

**Outcome (see `IMPLEMENTATION_NOTES.md` §6):** the 5 f32 block files were renamed to
`*_int8.c` and the 2 output blocks split into dedicated `vss_output{0,1}_mp_f32i8.c`
(+ `vss_output3_f32i8.c`). Each quantized layer is written out **explicitly** in the
block body as `quant_f32_to_int8[_perchannel]` → `*_int8` conv → `dequant_int32_to_f32_perchannel`
calling the micro kernels directly (no `mp_*` wrappers — keeps the kernel calls
visible). Each block takes new `int8_t *quant_buf` / `int32_t *acc_buf` scratch
(sliced per thread by `act_buf_slice_sz` in the parallel blocks; allocation is
Step 5). All 7 compile clean with `-Wall -Wextra -fopenmp`. Makefile unchanged
(`find`-based + `-fopenmp` already on for `sq_int8_v0`).

---

### Step 5 — INT8 buffers + inference entry — `model/model.c`, `headers/sq_int8_v0/model.h`, `common/model_buf_init_f32_i8.c`, `Makefile`  ✅ DONE (2026-06-11)
**Goal:** allocate the extra int8/int32 scratch and wire the top-level run function to
the Step-4 blocks.

- `model_buf_init_f32_i8(...)` now also allocates, per stream, `int8_t *quant_buf` and
  `int32_t *acc_buf`, each `QUANT_BUF_ELEMENT` / `ACC_BUF_ELEMENT` = `ACT_BUF_ELEMENT`
  elements (model.h). That bound is the largest single block working set and is sliced
  the same way the f32 act buffer is (per-thread by `act_buf_slice_sz` in the parallel
  PVSS blocks; base for the sequential MLP/PE/output, patch-embed, and split2 skip
  handlers), so it covers every quantized layer's quant input and conv accumulator.
- `dual_stream_unet_mp_int8(...)` (renamed from `dual_stream_unet_mp_f32`): calls the
  Step-4 blocks (`patch_embed_int8`, `pvss_ds/latent/us_mp_int8`,
  `vss_output{0,1}_mp_f32i8`, `vss_output3_f32i8`) with the v3 structs, no
  `input_selection`/`norm_ena` args, threading `quant_buf`/`acc_buf` through. The 4 skip
  handlers (dec0..2 + out0) are written **inline & explicit** (no wrapper):
  `quant_f32_to_int8_perchannel(x_inout, …, act_scale)` +
  `quant_f32_to_int8_perchannel(x_res, …, act_scale + C_IN)` →
  `pointwise_conv2d_split2_int8` → `dequant_int32_to_f32_perchannel`, where the
  `[2·C_IN]` act_scale's first half quantizes `x_inout` (offset-0 of `quant_buf`) and
  second half `x_res` (offset `C_IN*L`). STFT/iSTFT and all float islands unchanged.
- **Makefile (at Step 5):** the stale `model_timing/` (still on the f32 API) was
  temporarily excluded from the `sq_int8_v0` build via a version-specific
  `find … -not -path '*/model_timing/*'`, so only the NORMAL path built. **Superseded
  by Step 8** (2026-06-12): `model_timing/` is now migrated to int8 and the exclusion
  is removed — `sq_int8_v0` builds all three MODEs like naives/naives_mp.

**Verified:** all 32 `sq_int8_v0` sources compile clean with `-Wall -Wextra -fopenmp -O3`
(local FFTW include); a relocatable link (`ld -r`) of every version + helper object
leaves only external libc/libm/fftw/libgomp symbols undefined (plus `main`, = Step 6) —
i.e. orchestrator ↔ blocks ↔ kernels ↔ loader all resolve. End-to-end run + numerical
check come with Steps 6–7.

---

### Step 6 — entry point — `main/sq_int8_v0/main.{c,h}` (new)  ✅ DONE (2026-06-11)
**Goal:** make `make VERSION=SQ_INT8_V0` build a runnable binary.

- `main/sq_int8_v0/main.h`: copy of naives_mp's, repointed paths only —
  `WEIGHT_PATH = ./data/weight_sq_int8_v0/`,
  `GENERATED_AUDIO_PATH = ./data/test_audio/generated/sq_int8_v0/`,
  `RESULT_PATH`/CSV → `sq_int8_v0` (shared `DEGRADED_AUDIO_PATH`). At Step 6 a
  `#error` rejected the timing MODEs (timing not ported); **removed in Step 8**, which
  also adds the `STAGE_TIMING_CSV_PATH` / `DETAIL_STAGE_TIMING_CSV_PATH` macros.
- `main/sq_int8_v0/main.c`: at Step 6 a minimal **NORMAL-only** entry (all timing/CSV
  machinery dropped — **restored in Step 8**). Declares the 4 extra scratch pointers (`quant_buf_mag/pha` int8,
  `acc_buf_mag/pha` int32), calls `model_buf_init_f32_i8(...)` with them appended after
  `hidden_state_buf_*`, then `dual_stream_unet_mp_int8(...)` with them inserted after
  `hidden_state_buf_*` (before STFT args). File loop + per-segment `processing_time.csv`
  + summary unchanged from naives_mp.

**Verified:** `make VERSION=SQ_INT8_V0` builds `upvm_asr_sq_int8_v0` clean (no warnings).
A run over `data/test_audio/downsample/` processed segments at ~590 ms each (no
crashes, empty stderr) and wrote enhanced wavs to
`data/test_audio/generated/sq_int8_v0/`; spot-checked output is **48 kHz / 16-bit /
mono, finite, RMS ≈ 2.7e3, 99.9% non-zero** (valid speech, not NaN/silence).
Quality vs the PyTorch W8A8 reference is Step 7.

---

### Step 7 — numerical validation (acceptance)  ✅ DONE (2026-06-12)
**Goal:** confirm the assembled C model matches the quantized PyTorch reference.

Ran the full 2937-file tester through `upvm_asr_sq_int8_v0`, then
`sources/helpers/performance_verify.py` (metric `model_pytorch/metric.py`, 2048-pt
STFT, highcut 171) against the clean references in `data/test_audio/original/`. The
PyTorch W8A8 reference is the tester run in `logs/nohup/log_quantize_test.log`
(verified to be **this** checkpoint: loads `logs/logs_6/Ultra_Light/…`, modules are
`QConv2dW8A8`/`QLinear2dW8A8` with the exact checkpoint names incl. the dead
`mlp.dim_reduce` / `mlp.fc2.mlp_dim_redu`, same 2937-file 16k→48k tester).

| Metric (mean over 2937) | PyTorch W8A8 (reference) | C `sq_int8_v0` |
|---|---|---|
| SNR (dB) | 23.41 | **22.80** |
| LSD | 0.4799 | **0.4547** |
| LSD_HF | 0.5256 | **0.4978** |
| LSD_LF | 0.0055 | 0.0161 |

The C int8 model tracks the W8A8 reference within ~0.6 dB SNR (and is marginally
better on LSD / LSD_HF). The small LSD_LF gap is STFT/iSTFT numerics (FFTW vs
`torch.stft`) in the copied low-freq band — negligible (both ≪ 0.02). **Within
tolerance → accepted.** Report: `results/sq_int8_v0/performance_verify.txt`.

---

### Step 8 — INT8 timing path (`model_timing/` + main + Makefile)  ✅ DONE (2026-06-12)
**Goal:** bring `MODE=STAGE_TIMING` / `DETAIL_STAGE_TIMING` to `sq_int8_v0`, mirroring
naives/naives_mp, so the int8 forward can be profiled per stage and per PVSS-vs-MLP
sub-stage. Derived from the validated `model/` int8 blocks (not the stale f32 timing
files), so the timed compute is bit-identical to NORMAL.

- **6 timing block variants** (each = its NORMAL `model/` block + two `struct timespec*`
  outputs `pvss_time`/`mlp_time`, sampled at the SSM→MLP boundary and function end):
  `pvss_ds_timing_mp_int8.c`, `pvss_latent_timing_mp_int8.c`, `pvss_us_timing_mp_int8.c`,
  `vss_output0_timing_mp_f32i8.c`, `vss_output1_timing_mp_f32i8.c`,
  `vss_output3_timing_f32i8.c`. (output0/1 need dedicated variants because the int8
  output stage uses dedicated fns, unlike f32 which reused `pvss_us` with flags.)
- **2 orchestrators:** `model_timing/model.c` (`dual_stream_unet_timing_profiling_mp_int8`,
  STAGE) = NORMAL int8 `model/model.c` + a `timing_record_points*` and a `clock_gettime`
  per stage (calls the NORMAL blocks); `model_timing/model_detail.c`
  (`dual_stream_unet_detail_timing_profiling_mp_int8`, DETAIL) = same + a
  `detail_timing_record_points*`, calling the *timing* block variants with the two
  trailing `timespec*` args. The 4 int8 skip-handlers (dec0/1/2, out0) and the dedicated
  output0/1/3 calls are preserved verbatim from NORMAL.
- **Stale f32 timing files deleted** (`pvss_ds_timing_mp_f32.c`,
  `pvss_latent_timing_mp_f32.c`, `pvss_us_timing_mp_f32.c`, `vss_output3_timing_f32.c`).
- **Headers:** `macro_kernels.h` — the 4 stale f32 timing protos replaced by the 6 int8
  timing protos (each with `int8_t *quant_buf, int32_t *acc_buf` + the two `timespec*`);
  `model.h` — the two timing-orchestrator protos renamed `…_mp_f32` → `…_mp_int8` and
  given the 4 quant/acc scratch params. The `timing_record_points` /
  `detail_timing_record_points` structs in `datatypes.h` are precision-independent and
  were already present — no change.
- **main:** `main.h` drops the `#error` guard, adds `STAGE_TIMING_CSV_PATH` /
  `DETAIL_STAGE_TIMING_CSV_PATH`; `main.c` restores the full timing path (the two
  `#ifdef` record/`double` blocks, CSV headers, `elapsed_ms` chains, fprintf) byte-for-byte
  from naives_mp, with the model calls swapped to the int8 names + 4 scratch args. Timing
  FILE* declarations are `#ifdef`-guarded (clean `-Wall -Wextra` in every MODE).
- **Makefile:** the `sq_int8_v0`-specific `model_timing/` exclusion removed — back to the
  generic `find` used by all versions.

**Verified:** clean rebuild of **all three MODEs** (`NORMAL`, `STAGE_TIMING`,
`DETAIL_STAGE_TIMING`) with `-Wall -Wextra -O3 -fopenmp` — **no warnings/errors** — each
producing its binary (`upvm_asr_sq_int8_v0[_…]`). Smoke runs: STAGE → 66-col
`stage_timing.csv` (header≡data cols, all `total_time>0`/finite); DETAIL → 106-col
`detail_stage_timing.csv`, and the PVSS/MLP split is self-consistent
(e.g. enc0 `pvss 57.13 + mlp 8.52 = 65.65` = stage total). NORMAL compute path
unchanged → Step 7 numerics still hold.

**Done.** The mixed-precision (f32 + int8) C implementation is complete and validated.
