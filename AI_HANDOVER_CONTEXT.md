<!-- ============================================================================
     AI HANDOVER CONTEXT  —  UPVM-ASR-FOR-C
     RIGID SHARED BATON between Claude Code (Anthropic) and Codex (OpenAI GPT).
     One project, two agents, alternating when one runs out of usage.

     >>> INCOMING AGENT: read this WHOLE file before touching anything. <<<
     >>> OUTGOING AGENT: update the marked sections, then sign §1 + tick §10. <<<

     DO NOT restructure this file. Only edit CONTENT inside sections, following
     each section's UPDATE RULE. Keep it terse, factual, current.
     ============================================================================ -->

# AI HANDOVER CONTEXT — UPVM-ASR-FOR-C

**Baton holder right now:** Codex (GPT)  ·  **Last handover:** 2026-06-26 by Claude Code
**Single source of truth for the project itself = `README.md`.** This file is only the
*live working state* + *handover protocol*. When in doubt, trust `README.md` and the code.

---

## 0. PROTOCOL  ·  UPDATE RULE: read-only (do not edit; both agents obey)

1. **Read order on arrival:** this file → `git status` / `git log --oneline -5` → `README.md` §1
   (versions) + the §7.x for the active version → the section of code you will touch.
2. **Sections are typed.** Each has an UPDATE RULE:
   - `OVERWRITE` = replace its content with the now-current truth every handover.
   - `APPEND-ONLY` = add a new dated entry on top; never edit/delete past entries.
   - `read-only` = the template/protocol; do not change.
3. **Before you hand over** (running low on usage, or task done): update §1, §4, §5;
   add a §6 DONE entry; add any new §8 landmine; refresh §9; tick the §10 checklist.
4. **One baton.** Set "Baton holder right now" (top) to the agent you are handing to.
   Never both edit at once. If you are unsure who holds it, assume you do NOT and ask the user.
5. **Facts only.** No speculation in §2/§3/§4. Mark guesses explicitly as "ASSUMPTION:".
6. **Never silently change the user's intent.** If you deviate from a planned approach,
   record why in §6 and (if it bites later) §8.
7. **Honor the hard rules in §3.** They are non-negotiable environment constraints.
8. Keep paths machine-relative where possible (see §3 mount caveat).

**Entry templates (copy exactly):**
```
§1 header:  From <agent> → To <agent> | <YYYY-MM-DD> | reason: <out-of-usage|task-done|blocked>
§6 log:     ### <YYYY-MM-DD> · <agent> · <one-line title>
            - did: …   verified: …   left: …
§8 landmine: - [<area>] <what bites> → <how to avoid>  (added <YYYY-MM-DD> <agent>)
```

---

## 1. HANDOVER HEADER  ·  UPDATE RULE: OVERWRITE every handover

- **From → To:** Claude Code → Codex | 2026-06-26 | reason: task-done (v10 delivered; awaiting full build on an FFTW-dev host)
- **Branch / commit:** `main` (working tree dirty, NOT committed) · last commit `d383bee Add ops fusion for SSM`
- **One-line state:** `sq_int8_v10` (AVX-VNNI vectorized int8 kernels, bit-exact to v9) is written + unit-verified; full binary link is blocked on this box (no FFTW dev).
- **Incoming agent's first move:** see §5 (the baton task).

---

## 2. PROJECT SNAPSHOT  ·  UPDATE RULE: OVERWRITE when it changes

- **What:** C host app for **UPVM-ASR** speech super-resolution (16 kHz → 48 kHz). Dual-stream
  (magnitude / phase) **VMamba/SS2D** model over STFT frames; C port mirrors a PyTorch reference.
  Supports f32 and **W8A8 SmoothQuant int8**.
- **Version chain:** `NAIVES`, `NAIVES_MP`, `SQ_INT8_V0..V10`. Each adds one fusion/optimization;
  see `README.md` §1 table + §7.x. **V10 = V9 math, int8 kernels vectorized (AVX-VNNI), bit-exact.**
- **Build:** `VERSION=<UPPER> MODE=<NORMAL|STAGE_TIMING|DETAIL_STAGE_TIMING|MEM_PROFILE> make`
  (Makefile globs `sources/<ver>/**.c` + `sources/helpers` + `main/<ver>/main.c`; needs FFTW — see §3).
- **Run:** `./upvm_asr_<ver>[_<mode>]` → writes `data/test_audio/generated/<ver>/` + `results/<ver>/`.
- **Layout:** `headers/<ver>/`, `sources/<ver>/{common,kernels,model,model_timing,model_memory}/`,
  `main/<ver>/`, `data/weight_<ver>/`, `model_pytorch/weight_extractor_sq_int8_<ver>.py`.
- **Timing tool:** `sources/helpers/timing_summarize.py` (edit `INPUT_CSV_PATH`; mean±std of CSV cols).

---

## 3. ENVIRONMENT & HARD CONSTRAINTS  ·  UPDATE RULE: OVERWRITE when it changes

- **Dev box (where Claude runs):** Linux, GCC 11.4, host CPU has `avx2 avx512f avx512_vnni avx_vnni fma`.
  Mounted over **NAS/QNAP sshfs** → **recursive `grep -r` / `find /` are SLOW or hang**; prefer
  targeted paths, cap depth, avoid whole-tree scans.
- **CANNOT run/link the full binary here:** no `fftw3.h`, no `libfftw3f.so` dev symlink (only runtime
  `libfftw3f.so.3`). Kernel `.o` files compile fine (they include only `micro_kernels.h`). To full-build:
  install `libfftw3f-dev` OR provide a local prefix at `../fftw3/local` (Makefile auto-detects it).
- **Mount caveat:** same files appear at different absolute paths per machine
  (NAS box: `/F/AI_Train/.../NAS/Projects/AuAx/Model/UPVM-ASR-FOR-C`; phuong's laptop:
  `~/QNAP/Projects/AuAx/Model/UPVM-ASR-FOR-C`). Hardcoded absolute paths break across machines.
- **int8 deployment targets** (verified): **i7-12700K** = Alder Lake, has `avx2/fma/avx_vnni`, **NO AVX-512**;
  **Ryzen 9700X** = Zen 5, has AVX-512 + VNNI + `avx_vnni`; **ARM Cortex-A55** = later (NEON `sdot`, 128-bit).
  → **Common ISA = 256-bit AVX-VNNI.** **HARD RULE: never `-march=native`** (would SIGILL on the 12700K).
- **HARD RULE: never damage other versions.** Only touch the version you are working on + shared files
  (Makefile/README/.gitignore) additively. v0..v9 must stay byte-identical.
- **Latency facts (v9, inner_8, n=3659):** ~284 ms/segment; budget <150 ms (ITU-T). SS2D ≈ 42%,
  MLP ≈ 38%. `mag`/`pha` branches run concurrently → single-branch sum ≈ wall-clock. **SS2D selective
  scan is f32 and NOT yet vectorized** (the biggest remaining item; see README §11.2).

---

## 4. CURRENT STATE — WHERE WE ARE  ·  UPDATE RULE: OVERWRITE every handover

- **`sq_int8_v10` created and unit-verified.** It is a byte-copy of v9 with **4 int8 kernels vectorized
  with AVX-VNNI**, each keeping its signature + a scalar `#else` fallback, all **bit-exact to v9**:
  `pointwise_conv2d_int8`, `pointwise_conv2d_split2_int8`, `depthwise_conv2d_int8`, `depthwise_conv2d_int32`.
  (Method: `vpdpbusd` with on-stack transpose tile + `+128` offset / `128·Σw` compensation for the GEMMs;
  AVX2 widening for depthwise. Identity `Σ(x+128)·w − 128·Σw = Σx·w` ⇒ integer-exact.)
- **Makefile:** v10 added to OpenMP filter; SIMD flags `-mavx2 -mfma -mavxvnni` scoped to **ONLY the 4
  kernel objects** (`SIMD_KERNEL_OBJS`) so the rest of v10 keeps v9 codegen and the **f32 path is
  bit-identical** (no auto-vec/FMA drift). v9/naives flags unchanged (verified by dry-run).
- **Weights:** `data/weight_sq_int8_v10/` = byte-copy of v9 (804 files). v10 needs **no new hyperparameter**.
- **Extractor:** `weight_extractor_sq_int8_v10.py` `OUTPUT_ROOT` was wrongly pointing at the **v9** dir with
  `rmtree` (would have wiped v9) — **fixed** to `weight_sq_int8_v10`.
- **README:** v10 row in §1 + new **§7.10** + §11 status updated. **.gitignore:** now code-base-only
  (ignores `/data/weight_*/`, `*.wav`, results CSV/TXT/LOG; keeps `results/` tree via `.gitkeep`).
- **NOT done:** full binary build/run (needs FFTW dev — see §5); on-target latency numbers; SS2D f32 scan.

---

## 5. ACTIVE TASK — THE BATON  ·  UPDATE RULE: OVERWRITE every handover

**Goal for the incoming agent:** prove v10's latency win and confirm v10≡v9 output, then report.

1. On a host with `libfftw3f-dev` (or `../fftw3/local`): `VERSION=SQ_INT8_V10 MODE=DETAIL_STAGE_TIMING make`.
   If it fails to link, that's the FFTW issue in §3 — install dev libs or build local FFTW.
2. Run v10 and v9 on the same inputs; **confirm generated audio is identical (bit-exact)** and compare
   `results/sq_int8_v10` vs `results/sq_int8_v9` stage timing (use `sources/helpers/timing_summarize.py`).
3. Report: per-stage latency delta on the int8 GEMM/depthwise kernels (expected: faster; f32 scan unchanged).
4. **Acceptance:** v10 SNR == v9 SNR (bit-exact), int8 kernels measurably faster, no other version touched.

**Do NOT:** add `-march=native`; modify v0..v9; vectorize the SS2D f32 scan in v10 (that's a *future*
version, not v10); change any kernel API/buffer.

---

## 6. DONE LOG  ·  UPDATE RULE: APPEND-ONLY, newest first

### 2026-06-26 · Claude Code · Built sq_int8_v10 (AVX-VNNI int8 kernels)
- did: copied v9→v10 (headers/sources/main/weights); vectorized 4 int8 kernels with AVX-VNNI +
  scalar fallback; scoped SIMD flags to those 4 objects in Makefile; fixed v10 extractor OUTPUT_ROOT;
  README §1 row + §7.10 + §11; .gitignore → code-base-only.
- verified: all 4 kernels **bit-exact vs scalar** over many shapes (incl. non-mult-of-8/4, L=1, large
  C_IN, k=3/5/7, split boundary, stride-2 sumpool); all 26 v10 kernel TUs compile clean `-Wall -Wextra
  -mavxvnni`; Makefile dry-run shows flags only on the 4 kernels; v9/naives unchanged.
- left: full build+run (FFTW dev missing here) → §5.

### 2026-06-26 · Claude Code · README §11 + .gitignore + handover scaffolding
- did: README §11 (CPU vectorization roadmap + SS2D-scan + FPGA notes); first `.gitignore` pass;
  authored this handover file.
- verified: gitignore behavior in throwaway git repos.

---

## 7. OPEN QUESTIONS / DECISIONS PENDING  ·  UPDATE RULE: OVERWRITE (resolve or carry forward)

- None blocking. (Resolved this session: v10 = bit-exact vectorization, reuse v9 weights, VNNI with
  scalar fallback, core kernels only — all confirmed by the user.)
- Future (not v10): vectorize the **SS2D selective scan** (f32, ~42% of time) — channel-major SIMD +
  vectorized `exp`; see README §11.2. Decide float-SIMD vs eventual fixed-point before an FPGA port (§11.3).

---

## 8. LANDMINES / GOTCHAS  ·  UPDATE RULE: APPEND-ONLY (never silently remove)

- [build] **Never `-march=native`** on an AVX-512 box → SIGILL on the i7-12700K. Use `-mavx2 -mfma -mavxvnni`.  (2026-06-26 Claude)
- [build] v10 SIMD flags are **per-object** (only the 4 kernels); whole-tree `-mavx2/-mfma` would drift the **f32** kernels vs v9 (auto-vec/FMA), breaking bit-exactness.  (2026-06-26 Claude)
- [weights] Each `weight_extractor_sq_int8_vN.py` does `rmtree(OUTPUT_ROOT)` then writes — **OUTPUT_ROOT must point at its OWN version dir**, never another's. (v10's was wrong, fixed.)  (2026-06-26 Claude)
- [env] NAS/sshfs: recursive `grep -r`/`find /` **hang/timeout** — scope to exact paths, cap depth.  (2026-06-26 Claude)
- [env] No FFTW dev on the dev box → can compile kernel `.o` but **cannot link the full binary** here.  (2026-06-26 Claude)
- [paths] Same repo mounts at different absolute paths per machine (`/F/AI_Train/.../NAS/...` vs `~/QNAP/...`); hardcoded absolute paths break cross-machine (bit Claude on `timing_summarize.py`).  (2026-06-26 Claude)
- [tooling] `sources/helpers/timing_summarize.py` **crashes (`NoneType.strip`)** on a truncated/partial last CSV row (e.g. a run still writing) — point it at a *finished* CSV.  (2026-06-26 Claude)
- [vcs] `.gitignore` is **code-base-only**: weights, `*.wav`, and results CSV/TXT/LOG are ignored; `results/` keeps only its folder tree via `.gitkeep`. Don't expect data files in git.  (2026-06-26 Claude)

---

## 9. VERIFICATION STATUS  ·  UPDATE RULE: OVERWRITE every handover

| Item | Status | How |
|------|--------|-----|
| v10 kernels compile (`-mavxvnni`) | ✅ | all 26 v10 kernel TUs, `-Wall -Wextra`, 0 warnings |
| v10 kernels bit-exact vs scalar | ✅ | randomized harness, many shapes (scratchpad) |
| Makefile flags scoped to 4 kernels | ✅ | `make -n` dry-run inspection |
| v0..v9 untouched | ✅ | no `immintrin` in v9 kernels; flags unchanged |
| Full v10 binary builds | ⛔ blocked | no FFTW dev on this box (§3) |
| v10 run == v9 output (SNR) | ⏳ pending | needs full build → §5 |
| v10 on-target latency vs v9 | ⏳ pending | run on 12700K / 9700X → §5 |

---

## 10. OUTGOING HANDOVER CHECKLIST  ·  UPDATE RULE: re-tick every handover

- [x] §1 header overwritten (from/to/date/reason).
- [x] §4 current state reflects reality.
- [x] §5 baton task is concrete + has acceptance criteria + "do NOT" list.
- [x] §6 DONE entry appended.
- [x] §8 new landmines appended (if any).
- [x] §9 verification table refreshed.
- [x] "Baton holder right now" (top) set to the receiving agent.
- [x] Working tree state noted (committed? dirty? which files).
