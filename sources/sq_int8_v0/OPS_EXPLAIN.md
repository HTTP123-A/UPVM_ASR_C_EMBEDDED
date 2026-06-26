# sq_int8_v0 — Operation & Memory Scheme (for the hardware team)

Mixed-precision (F32 + INT8) C implementation of the dual-stream VMamba speech
super-resolution model (16 kHz → 48 kHz). This is the **INT8 sibling** of the f32
`VSS_Block` scheme (`resources/VSS_Block_f32.png` / `.drawio`): same dataflow and
the same buffer layout, plus the W8A8 quant/dequant points and two extra scratch
buffers. Numbers below are derived directly from `headers/sq_int8_v0/model.h` and the
`model/` block sources; this checkpoint is `gen_ptq_int8pack.pth`.

---

## 0. TL;DR

- **Two independent streams** run in parallel on 2 OpenMP threads: **magnitude** (tid 0)
  and **phase** (tid 1). They have identical structure and **separate buffers**; they
  only exchange data at the per-stage "interaction" (`mag += pha; pha += mag`).
- **Precision split (W8A8, symmetric, weight zero-point = 0):**
  - **INT8** (int32 accumulator): all conv / linear layers — patch-embed conv1/conv2,
    SS2D `in_proj`/`dw_conv2d`/`out_proj`, MLP `fc1`/`fc2`, patch-expand `pe`, and the
    decoder/output `skip_conv` handlers.
  - **F32** (unchanged): every LayerNorm, the SSM `x_proj`/`dt_proj` grouped conv1d, the
    selective scan, cross-scan/merge, the gated multiply, all residual adds, the MLP
    sum-pool & `fc_res`, pixel-shuffle, GeLU/ReLU, and STFT/iSTFT.
  - **Exception — output stages:** in `output0/1/3` the **entire SS2D runs F32**; only
    `fc1`/`fc2` (+`pe` for out0/1) are INT8. (Reason: those ss2d weights were left in f32
    in the checkpoint.)
- **Every INT8 layer is an explicit 3-step micro-op:**
  `quant (f32→int8)` → `int8 conv/matmul (int32 acc, int32 bias folded in)` →
  `dequant (int32→f32)`. No fused kernels; the f32 islands read/write f32 around it.
- **Memory:** identical f32 working buffers as the f32 build, **plus** two INT8 scratch
  buffers per stream — `quant_buf` (int8) and `acc_buf` (int32), each
  `ACT_BUF_ELEMENT = 7,864,320` elements. Net activation footprint ≈ **152 MiB**
  (both streams) vs ≈ 77 MiB for f32-only. See §4.

---

## 1. Precision map (per operation)

| Op (PyTorch name)                  | Kernel                              | Precision | Notes |
|------------------------------------|-------------------------------------|-----------|-------|
| patch-embed `conv1`,`conv2`        | `conv2d_int8`                       | **INT8**  | k3 s2 p1 |
| `norm1`/`norm2`, GeLU              | `layernorm_f32`, `gelu_f32`         | F32       | |
| SS2D `in_proj`                     | `pointwise_conv2d_int8`             | **INT8**¹ | per group |
| SS2D `dw_conv2d`                   | `depthwise_conv2d_int8`             | **INT8**¹ | k3 same-pad |
| SS2D `x_proj` (type-major)         | `grouped_pointwise_conv1d_type_major_f32` | F32 | |
| SS2D `dt_proj`                     | `grouped_pointwise_conv1d_f32`      | F32       | |
| SS2D selective scan                | `selective_scan_f32`                | F32       | uses hidden-state buf |
| CrossScan / CrossMerge             | `cross_scan_f32` / `cross_merge_f32`| F32       | 4 directions |
| SS2D `out_norm`, gated `x*z`       | `layernorm_f32`, `elemwise_mul_f32` | F32       | |
| SS2D `out_proj`                    | `pointwise_conv2d_int8`             | **INT8**¹ | per group |
| MLP `fc1`,`fc2`                    | `pointwise_conv2d_int8`             | **INT8**  | |
| MLP `sumpool`, `fc_res`            | `depthwise_conv2d_f32`, `pointwise_conv2d_f32` | F32 | DS branch only |
| patch-expand `pe`                  | `pointwise_conv2d_int8`             | **INT8**  | US/out0/out1 |
| pixel shuffle, `pe_norm`           | (inline), `layernorm_f32`           | F32       | |
| decoder/out0 `skip_conv`           | `pointwise_conv2d_split2_int8`      | **INT8**  | 2·C_in → C_out |
| output3 `pre_conv`                 | `pointwise_conv2d_f32`              | F32       | 4C → C |
| ReLU, residual add, freq-replace   | `relu_f32`, `elemwise_add_f32`, memcpy | F32    | |

¹ **INT8 only in encoder/latent/decoder PVSS blocks.** In `output0/1/3` the whole SS2D
(`in_proj`/`dw_conv2d`/`out_proj` included) is **F32**; only the MLP (`fc1`/`fc2`) and
`pe` are INT8.

---

## 2. The INT8 micro-op (what every quantized layer expands to)

```
   f32 activation  x[C_in, L]
        │
        ▼  quant_f32_to_int8_perchannel(x, qbuf, C_in, L, act_scale[C_in], zp=0)
   int8 qbuf[C_in, L]                         (per-INPUT-channel scale; symmetric)
        │
        ▼  pointwise/depthwise/conv2d_int8(qbuf, w_i8, bias_i32, acc, ...)
   int32 acc[C_out, L]                         (int32 MAC; bias pre-folded as int32)
        │
        ▼  dequant_int32_to_f32_perchannel(acc, y, C_out, L, deq_scale[C_out])
   f32 activation  y[C_out, L]                 (per-OUTPUT-channel scale)
```

- Weights are **int8** (`w_i8`), zero-point 0 (symmetric). Bias is **int32**, pre-scaled
  at export so it adds directly into the int32 accumulator.
- `act_scale` is **per input channel** `[C_in]`; `deq_scale` is **per output channel**
  `[C_out]`. (The 6 layers whose `C_in == 1` use a length-1 `act_scale` — handled by the
  same per-channel kernel with C=1, bit-identical to per-tensor.)
- Scratch: `qbuf` is a slice of `quant_buf` (int8), `acc` is a slice of `acc_buf` (int32).
  Both are caller-provided; see §4.

---

## 3. Operation pipelines per block (the "graph")

Legend:  `[INT8]` = quant→int8 conv→dequant (§2);  `(f32)` = stays float;
shapes are `C×H×W`;  `L = H·W`.

### 3.1 Patch Embed   `[1,512,512] → [16,128,128]`
```
in[1,512,512] ─[INT8 conv1 k3s2p1]→ [8,256,256] ─(norm1, f32)→ ─(GeLU, f32)→
              ─[INT8 conv2 k3s2p1]→ [16,128,128] ─(norm2, f32)→ out
```

### 3.2 SS2D / "Parallel SSM Branch"  (encoder / latent / decoder)
Run once per block on `[C,H,W]`; the **group loop** repeats `reduce_factor` times over
`SUB_C`-channel slices and is split across `INNER_MP_PARALLEL=2` threads.
```
x[C,H,W] ─(SSM pre-norm LN, f32)→ ─(copy → residual_buf, f32)
   for g in reduce_factor:                       ── per group, sub-tensor [SUB_C,H,W] ──
     [INT8 in_proj]   SUB_C → 2·D_inner   → {x, z}
     [INT8 dw_conv2d] depthwise k3 on x   → x'
     (ReLU x, f32) (ReLU z, f32)
     (CrossScan x, f32: 4 dirs)
     (x_proj  grouped-conv1d type-major, f32) → [dts; Bs; Cs] = R2N per dir
     (dt_proj grouped-conv1d, f32)            → dts[R]
     (selective scan, f32; A,Ds,delta_bias; hidden_state_buf)
     (CrossMerge, f32) (out-norm LN, f32) (gated x*z, f32)
     [INT8 out_proj]  D_inner → SUB_C
     (residual add, f32)
   → x[C,H,W]
```
This is **the** memory-driving stage; its working set is `act_buf_1 + act_buf_2` (§4.2).

### 3.3 MLP branch — 4 variants (this is what differs between block types)

**(a) DS / encoder — downsamples `C×H×W → 2C×(H/2)×(W/2)`**
```
x ─(copy→residual, f32)─(mlp-norm LN, f32)─[INT8 fc1: C→MLPh]─(ReLU,f32)
  ─(SumPool 2×2 s2, depthwise f32: MLPh, H,W → H/2,W/2)        # ÷4 spatial
  ─(SumPool 2×2 s2 on residual, f32: C → C, H/2,W/2)
  ─[INT8 fc2: MLPh → 2C  @ half-spatial] ─(fc_res PW f32: C→2C) ─(add, f32)
  → out[2C, H/2, W/2]
```

**(b) Latent — keeps `C×H×W` (no sumpool / no fc_res)**
```
x ─(copy→residual,f32)─(mlp-norm,f32)─[INT8 fc1: C→MLPh]─(ReLU,f32)
  ─[INT8 fc2: MLPh→C]─(add residual,f32) → out[C,H,W]
```

**(c) US / decoder — upsamples `C×H×W → (C/2)×2H×2W` (= latent MLP + PatchExpand)**
```
x ─(latent-style MLP, f32 islands + INT8 fc1/fc2)─ y[C,H,W]
  ─[INT8 pe: C → 2C] ─(PixelShuffle f32: [2C,H,W] → [C/2,2H,2W]) ─(pe-norm LN,f32)
  → out[C/2, 2H, 2W]
```
The decoder/out0 input first passes a **skip handler**:
`[INT8 split2 conv] (concat x_inout ‖ x_res, 2C_in → C_out)` before SS2D.

**(d) Output stages** (ss2d entirely F32; only MLP/pe INT8)
- `output0`: skip-handler → **F32 SS2D** (no block norms) → INT8 MLP → INT8 pe → shuffle → upsample.
- `output1`: **F32 SS2D** (with norms) → INT8 MLP → INT8 pe → shuffle → upsample.
- `output3`: `(pre_conv 4C→C, f32)` → **F32 SS2D** (no grouped loop) → INT8 `fc1`/`fc2`
  (fc1 is the `C_in==1` per-tensor case) → residual. **No pe, no upsample**; runs at full
  `[1,512,512]` — this is what sizes the activation buffer (§4).

### 3.4 Full forward (per stream)
```
STFT → PatchEmbed → Enc0 → Enc1 → Enc2 → Latent
     → Dec0 → Dec1 → Dec2 → Out0 → Out1 → Out3 → (+spectrogram residual) → freq-replace → iSTFT
```
After PatchEmbed and each Enc/Dec stage the streams interact (`mag`/`pha` add). The
encoder outputs and the input spectrogram are saved into the **global skip buffer** for the
decoder/output skip connections and the final residual.

---

## 4. Memory model

All activation buffers are **caller-allocated once** and reused across every stage (sized
to the worst-case stage). There are **two of each** — one per stream (mag, pha).

### 4.1 Buffer table (per stream)

| Buffer (`*_mag` / `*_pha`) | Elements | dtype | Bytes (mag) | Sized by / purpose |
|----------------------------|---------:|-------|------------:|--------------------|
| `model_inout_buf`     | 1,049,088 | f32 | 4,196,352 | working tensor; = `2·C_out1·L_out1 + 512` (Output1 is largest) |
| `model_internal_res_buf` | 524,288 | f32 | 2,097,152 | per-block residual = `C_out1·L_out1` |
| `model_global_res_buf` | 753,664 (mag) / 491,520 (pha) | f32 | 3,014,656 / 1,966,080 | skip connections + spectrogram (§4.3) |
| `model_act_buf`       | **7,864,320** | f32 | **31,457,280** | SS2D/MLP scratch; **driven by Output3** (§4.2) |
| `model_low_freq_buf`  | 87,552 | f32 | 350,208 | low-freq copy = `171·512` |
| `hidden_state_buf`    | 256 | f32 | 1,024 | SSM recurrent state (Dec0 largest) |
| **`quant_buf`** (INT8) | **7,864,320** | **int8** | **7,864,320** | int8 quant inputs; `= ACT_BUF_ELEMENT` |
| **`acc_buf`** (INT8)  | **7,864,320** | **int32** | **31,457,280** | int32 conv accumulators; `= ACT_BUF_ELEMENT` |

> The `pha` global-residual buffer is smaller by `C_in·L_in = 262,144` (only the `mag`
> stream stores the spectrogram residual slot used by the final add).

### 4.2 `act_buf` layout (why `ACT_BUF_ELEMENT = 30·L`)
Per scan-group the SS2D needs two regions, concatenated into one slice
`ACT_BUF_SLICE = (4·R2N + 9·D_inner)·L`:
- **act_buf_1** = `4·R2N·L + 4·D_inner·L` — the 4-direction `x_proj` outputs (`R2N` ch/dir)
  plus the 4-direction dw-conv `x`.
- **act_buf_2** = `5·D_inner·L` — `in_proj` `{x,z}` (`2·D_inner`), cross-merged `x`, `z`, etc.

For **Output3** (`R2N=3`, `D_inner=2`, `L = 512·512 = 262,144`):
`(4·3 + 9·2)·262,144 = 30·262,144 = 7,864,320` → this is the global `ACT_BUF_ELEMENT`.
(This matches the f32 scheme's `act_buf_1 + act_buf_2 = 20·L + 10·L = 30·L`.) The parallel
encoder/decoder stages slice the same buffer per thread; none exceeds Output3's single
sequential working set, so 7,864,320 bounds them all.

### 4.3 Global skip buffer = **753,664** elements (matches the f32 drawio "TOTAL")
Concatenated skip/residual slots:

| Slot | Shape | Elements |
|------|-------|---------:|
| spectrogram (final residual) | `[1,512,512]`  | 262,144 |
| patch-embed skip (→ out0)    | `[16,128,128]` | 262,144 |
| enc0 skip (→ dec2)           | `[32,64,64]`   | 131,072 |
| enc1 skip (→ dec1)           | `[64,32,32]`   |  65,536 |
| enc2 skip (→ dec0)           | `[128,16,16]`  |  32,768 |
| **total**                    |                | **753,664** |

### 4.4 Per-stage SS2D working set (per scan group)

| Stage | C | H=W | L | R2N | D_inner | `ACT_BUF_SLICE` | hidden/dir |
|-------|--:|----:|--:|----:|--------:|----------------:|-----------:|
| enc0 | 16 | 128 | 16,384 | 3 | 2 | 491,520 | 8 |
| enc1 | 32 | 64  | 4,096  | 3 | 4 | 196,608 | 16 |
| enc2 | 64 | 32  | 1,024  | 3 | 8 | 86,016  | 32 |
| latent | 128 | 16 | 256  | 3 | 16 | 39,936 | 64 |
| dec0 | 128 | 16 | 256    | 3 | 32 | 76,800  | 128 |
| dec1 | 64 | 32  | 1,024  | 3 | 16 | 159,744 | 64 |
| dec2 | 32 | 64  | 4,096  | 3 | 8 | 344,064 | 32 |
| out0 | 16 | 128 | 16,384 | 3 | 8 | 1,376,256 | 32 |
| out1 | 8  | 256 | 65,536 | 3 | 4 | 3,145,728 | 16 |
| **out3** | **1** | **512** | **262,144** | 3 | 2 | **7,864,320** | 8 |

### 4.5 Totals

| Group | mag | pha | both streams |
|-------|----:|----:|-------------:|
| F32 work buffers | 41,116,672 B (39.2 MiB) | 40,068,096 B (38.2 MiB) | **81,184,768 B (77.4 MiB)** |
| INT8 scratch (`quant`+`acc`) | 39,321,600 B (37.5 MiB) | 39,321,600 B (37.5 MiB) | **78,643,200 B (75.0 MiB)** |
| **Activation total** | 80,438,272 B | 79,389,696 B | **159,827,968 B (152.4 MiB)** |

- **INT8 scratch overhead = `+5 bytes / activation element`** (`1` for the int8 quant input
  + `4` for the int32 accumulator), per stream. Because this is **mixed** precision, the f32
  islands stay live *alongside* the int8 scratch — the int8 path **adds** memory, it does not
  replace the f32 working set. Net ≈ **2×** the f32-only activation footprint.
- **Weights:** the exported `data/weight_sq_int8_v0/` is ≈ **3.3 MB** on disk (vs ≈ 3.5 MB
  f32). Storage barely shrinks because only the conv/linear weights become int8 while all
  the f32 islands (SSM `A`/`Ds`/`delta_bias`, `x_proj`/`dt_proj`, norms, sumpool, `fc_res`,
  the f32 output-stage ss2d) remain f32. INT8 here buys **compute** (int8 MACs), not storage.

---

## 5. Hardware notes / requirements

1. **INT8 MAC with INT32 accumulate** is mandatory for every quantized layer (no int8
   saturation mid-accumulation). Bias is pre-folded into int32, so the datapath only needs
   `acc += w_i8 · x_i8` then `+ bias_i32`.
2. **Per-channel scaling.** `act_scale` indexes the **input** channel (length `C_in`),
   `deq_scale` indexes the **output** channel (length `C_out`). Both symmetric (zp = 0), so
   no zero-point offset terms in the MAC. Quant is `round(x / act_scale)` clamped to int8;
   dequant is `acc · deq_scale`.
3. **Scratch sizing is fixed at `ACT_BUF_ELEMENT = 7,864,320`** for both `quant_buf` (int8)
   and `acc_buf` (int32) per stream — set by Output3 at full `[1,512,512]`. A tiled/streaming
   accelerator can size its int8/int32 SRAM to the **per-stage** working sets in §4.4
   instead (much smaller for the deep stages), since each layer is processed independently.
4. **Two streams, embarrassingly parallel** except the per-stage interaction adds. Each
   stream owns its buffer set; an accelerator can run them on two engines and only
   synchronize at the interaction points.
5. **Determinism:** the kernel choice never branches at runtime — every quantized layer
   always uses the per-channel quant kernel (the `C_in==1` layers are just the degenerate
   length-1 case), so the schedule is fully static.
6. **F32 islands are unavoidable in this checkpoint:** the SSM selective scan, both grouped
   conv1ds, all norms, and the entire output-stage SS2D are float. A pure-int8 datapath is
   not sufficient; the device needs an f32 (or higher-precision fixed) path for those.

---

## 6. Heterogeneous mapping — CPU (F32) + INT8 accelerator

The mixed precision is not just a numeric choice; it is a **partition of the model across two
engines**. Each op in §1/§3 carries an `engine` tag (also emitted by the profiler, §7):

| Engine        | Ops                                                                 | Why it lives there |
|---------------|---------------------------------------------------------------------|--------------------|
| **CPU-F32**   | the whole **SS2D scan core** (selective_scan, cross-scan/merge, `x_proj`/`dt_proj` grouped conv1d, out-norm, gated `x*z`), all **LayerNorms**, **GeLU/ReLU**, **residual/interaction adds**, MLP **sum-pool**/`fc_res`, **pixel-shuffle**, and **STFT/iSTFT** | big, contiguous, mostly **sequential / data-dependent** f32 work |
| **INT8-ACCEL**| `conv1`/`conv2` (patch-embed), SS2D `in_proj`/`dw_conv2d`/`out_proj`, MLP `fc1`/`fc2`, patch-expand `pe`, decoder/output `skip_conv` | regular **GEMM/conv-shaped** int8×int8→int32 work |
| **boundary**  | `quant` (f32→int8, **CPU→ACCEL**) and `dequant` (int32→f32, **ACCEL→CPU**) | elementwise scale at the DMA edge of the accelerator |

### 6.1 Why the SS2D / big F32 kernels stay **on the CPU**
- The **selective scan is a sequential state-space recurrence** along `L` (carry the f32
  hidden state `[4·D·N]` frame-to-frame). It is *not* a matrix multiply — it is a
  data-dependent scan with per-step f32 accumulation. It maps poorly onto a systolic/MAC
  array and would lose accuracy if quantized (the checkpoint deliberately keeps it f32).
- It is the model's **largest single contiguous f32 working set** — up to the full
  `act_buf` slice (peak **32 MB** per op, at Output1/Output3, see §7). Keeping it on the
  host CPU/vector unit avoids streaming tens of MB across the accelerator boundary.
- The same applies to LayerNorm (reductions), cross-scan/merge (gather/scatter), gating, the
  residual/interaction adds, and the FFTs: irregular or memory-bound f32 — cheap on a CPU,
  awkward on an int8 MAC array. **→ run SS2D and friends fully on the CPU.**

### 6.2 Why the INT8 ops go to the **accelerator**
- `in_proj`/`out_proj`/`fc1`/`fc2`/`pe`/`skip` are **pointwise convolutions = dense GEMMs**
  (`[C_out × C_in] · [C_in × L]`); `conv1`/`conv2`/`dw_conv2d` are small structured convs.
  All are **regular, statically-shaped int8×int8→int32 MAC** — the ideal workload for a
  systolic array / NPU / DSP MAC engine (high throughput, low power vs f32).
- They carry **the bulk of the multiply-accumulate count** but a **small weight footprint**
  (int8). Bias is folded to int32 and scales are per-channel (no zero-point), so the inner
  loop is a plain `acc += w_i8·x_i8` with one int32 add — trivial control.
- They are **independent per layer** (no recurrence), so they tile/stream cleanly: an
  accelerator can size its SRAM to the per-layer working set (§4.4), not the whole model.
  **→ offload every INT8 op to the accelerator; bridge with quant/dequant at its edge.**

### 6.3 Net picture
```
   audio ─STFT(CPU)─┐
                    ▼
   ┌──────────────────────────────  per stage  ──────────────────────────────┐
   │  CPU (F32):  norms · SS2D scan core · cross-scan/merge · gating ·        │
   │              residuals/interactions · sum-pool · pixel-shuffle           │
   │     ▲ dequant (int32→f32)                         quant (f32→int8) ▼      │
   │  ACCEL (INT8):  in_proj · dw_conv2d · out_proj · fc1 · fc2 · pe ·        │
   │                 patch_embed conv · skip_conv   (int8×int8 → int32)        │
   └──────────────────────────────────────────────────────────────────────────┘
                    ▼
              iSTFT(CPU) ─ audio
```
So the deployable design is a **heterogeneous CPU + INT8-accelerator system**: the CPU owns
the contiguous/sequential F32 math (SS2D, norms, FFT, residuals); the INT8 accelerator owns
the GEMM-shaped projections/convs/MLP; `quant`/`dequant` are the hand-off at the boundary.

---

## 7. Memory profiler (`MODE=MEM_PROFILE`)

A fourth build variant emits a **per-op memory report**. It runs the real `model/` kernels
(one magnitude stream, sequential — the phase stream is identical) and, before each
micro-op, logs its footprint. The profile is **data-independent** (a pure function of the
static shapes), so one segment is the whole report.

```bash
make VERSION=SQ_INT8_V0 MODE=MEM_PROFILE
./upvm_asr_sq_int8_v0_mem_profile          # writes results/sq_int8_v0/memory_profile.csv
```

`memory_profile.csv` — one row per micro-op, 14 columns:

| column | meaning |
|--------|---------|
| `idx, stage, op` | sequence index, block (`enc0`…`out3`/`patch_embed`/`pre`/`stft`/`istft`), op name |
| `engine` | `CPU-F32` / `INT8-ACCEL` / `CPU->ACCEL` (quant) / `ACCEL->CPU` (dequant) |
| `repeats` | times the op runs in the block's group loop (per-invocation footprint is reported) |
| `in_dtype, in_elems, in_bytes` | input activation: dtype, element count, bytes |
| `weight_bytes` | weight footprint (int8 W + int32 bias + f32 scales for int8 layers) |
| `out_dtype, out_elems, out_bytes` | output activation |
| `op_bytes, op_MB` | the op's working set = in + weight + out |

Trailing `SUMMARY` rows give the peak single-op working set and per-engine sums. From the
current run (**377 ops**):

| metric | value |
|--------|------:|
| peak single-op working set | **32.0 MB** (a full-resolution Output1/Output3 F32 op) |
| Σ op working set on **CPU-F32** (213 ops) | 378.0 MB |
| Σ op working set on **INT8-ACCEL** (52 ops) | 38.0 MB |
| Σ op working set on **quant/dequant boundary** (108 ops) | 96.6 MB |

> These Σ are **per-op working sets summed over the whole forward** (total data the engine
> touches / streams), **not** the resident footprint — buffers are reused, so they
> double-count. The *resident* allocation is §4 (≈ 152 MiB both streams). Use the Σ for
> bandwidth/utilisation between CPU and accelerator, the **peak single-op** (32 MB) for the
> minimum on-chip buffer of a streaming accelerator, and the **per-stage** tables (§4.4) to
> size SRAM per layer.

Implementation: `sources/sq_int8_v0/model_memory/` (`model.c` orchestrator + per-op log
helpers, `mem_profile.c` CSV logger) and `headers/sq_int8_v0/model_memory.h`.

---

*Source of truth: `headers/sq_int8_v0/model.h` (shapes/buffers), `sources/sq_int8_v0/model/`
(block ops), `common/model_buf_init_f32_i8.c` (allocations), `sources/sq_int8_v0/model_memory/`
(profiler). Cross-checked against the f32 reference scheme in
`resources/VSS_Block_f32.{png,drawio}`.*
