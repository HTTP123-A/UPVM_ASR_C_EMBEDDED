#!/usr/bin/env python3
"""Weight extractor for the sq_int8_v8 INT8-F32 mixed-precision C model.

v8 = v7 weights PLUS a shared piece-wise-linear (PWL) table approximating rsqrt() for the fused
int32 PatchEmbed LayerNorm (sources/sq_int8_v8/kernels/layernorm_int32_pwl_rsqrt.c). That norm reads
the int8 conv's int32 accumulator directly (the per-channel dequant is fused in) and replaces the
f32 1/sqrtf(var+eps) with a range-reduced LUT: frexp -> mantissa in [1,4) + even-exponent shift k,
then rsqrt(mantissa) ~= slope[i]*m + intercept[i], and * 2^-k. Range reduction makes ONE small table
cover every variance magnitude, so a single tuned segment count (the "max rule") serves all
LayerNorms. Knots are placed adaptively (denser where curvature is high) and each segment is a
min-max (Chebyshev) linear fit; tuning (`_suggest_segments`) picks the fewest segments meeting a
relative-error target. The table (knot[PWL_RSQRT_SEGMENTS+1], slope[PWL_RSQRT_SEGMENTS],
intercept[PWL_RSQRT_SEGMENTS], all f32) is written into BOTH embed stages; it is data-independent so
the bytes are identical everywhere. v8 reuses the entire v7 pipeline verbatim (every base + requant
artifact byte-identical to v7) and only ADDS the 6 PWL files (3 members x 2 embed streams).

v7 = v6 weights PLUS, for every fused SSM stage, an INTEGER fixed-point requant pair for the
SSM `in_proj -> dw_conv2d` (x-branch) REQUANT — the same dyadic-multiplier trick v6 applied to
the MLP `fc1 -> fc2` requant, now also applied to the SSM so that requant runs with no float:

    {mlp,ssm}_requant_scale[c]  (f32)   ~=   {mlp,ssm}_requant_mul[c] / 2^{mlp,ssm}_requant_shift[c]

i.e. the gemmlowp / TFLite "quantized multiplier" (dyadic) trick — a real per-channel
multiplier M is approximated by an integer multiplier `mul` and a power-of-two right
shift `shift`, so the C kernel computes (see
sources/sq_int8_v7/kernels/requant_int32_to_int8_perchannel_fixed.c):

    q[c,l] = clamp( ( (int64)acc[c,l]*mul[c] + (1<<(shift[c]-1)) ) >> shift[c],  -128, 127 )

Hyperparameter tuning: for each channel we SEARCH every shift in [1, 30] (finite), set
mul = round(M * 2^shift), reject mul outside (0, 2^31), and keep the shift that minimizes
the mismatch vs the exact f32 requant over a representative band of int32 accumulators
(tie-break: smaller LSB error, then larger shift = more multiplier precision). This is the
canonical "largest shift that keeps mul < 2^31" (gemmlowp's frexp choice) but verified by
the actual integer-vs-float error rather than assumed.

Reuses weight_extractor_sq_int8_v5 verbatim for every base + mlp_requant_scale +
ssm_requant_scale artifact, and reproduces v6's mlp_requant_mul/shift bit-for-bit (so all of
those are byte-identical to v6); v7 only ADDS the 14 ssm_requant_mul.bin + 14
ssm_requant_shift.bin (int32) files.

Source checkpoint: model/gen_ptq_int8pack.pth
Output:            ../data/weight_sq_int8_v8/<stage_dir>/<member>.bin (+ meta/manifest)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch

import weight_extractor_sq_int8_v0 as v0
import weight_extractor_sq_int8_v4 as v4
import weight_extractor_sq_int8_v5 as v5

HERE = Path(__file__).resolve().parent
PACK_PATH = HERE / "model" / "gen_ptq_int8pack.pth"
OUTPUT_ROOT = HERE.parent / "data" / "weight_sq_int8_v8"

SHIFT_MIN, SHIFT_MAX = 1, 30          # all candidate right-shifts to search (mul must stay < 2^31)
SEARCH_SAMPLES = 20000

# ===================== v8: PWL rsqrt table for the fused int32 PatchEmbed LayerNorm =====================
PWL_RSQRT_SEGMENTS = 10        # FROZEN segment count; MUST equal #define PWL_RSQRT_SEGMENTS in
                               # headers/sq_int8_v8/datatypes.h (the load-time file-size check enforces it).
RSQRT_TOL          = 1.5e-3    # tuning target: fewest segments with worst-case rsqrt rel-err <= this.


def _build_pwl_rsqrt_table(seg: int):
    """Adaptive knots on [1,4) (density ~ |f''|^(1/2) ~ x^(-5/4)) + per-segment MIN-MAX (Chebyshev)
    linear fit of rsqrt. Deterministic and data-independent. Returns (knot[seg+1], slope[seg],
    intercept[seg]) as float32 (the C kernel evaluates slope*m + intercept on the reduced mantissa)."""
    z = 1.0 - 4.0 ** -0.25
    i = np.arange(seg + 1, dtype=np.float64)
    knot = (1.0 - (i / seg) * z) ** -4.0
    knot[0], knot[-1] = 1.0, 4.0                          # exact endpoints
    f = lambda x: x ** -0.5
    slope = np.empty(seg, dtype=np.float64)
    icpt  = np.empty(seg, dtype=np.float64)
    for s in range(seg):
        a, b = knot[s], knot[s + 1]
        k  = (f(b) - f(a)) / (b - a)                      # secant slope (<0)
        xs = (-2.0 * k) ** (-2.0 / 3.0)                   # tangent point: f'(x*)=k, f'(x)=-0.5 x^(-3/2)
        gap = (f(a) + k * (xs - a)) - f(xs)               # secant-minus-f at x* (>=0 for convex rsqrt)
        slope[s] = k
        icpt[s]  = (f(a) - k * a) - 0.5 * gap             # shift secant down half the gap -> min-max
    return (np.ascontiguousarray(knot,  dtype=np.float32),
            np.ascontiguousarray(slope, dtype=np.float32),
            np.ascontiguousarray(icpt,  dtype=np.float32))


def _pwl_rsqrt_eval(t, knot, slope, icpt):
    """Range-reduced PWL rsqrt, bit-mirroring layernorm_int32_pwl_rsqrt.c (np.frexp == C frexpf)."""
    t = np.asarray(t, dtype=np.float32)
    mant, e2 = np.frexp(t)                                # t = mant*2^e2, mant in [0.5,1)
    odd = (e2 & 1).astype(bool)
    mp  = np.where(odd, mant * 2.0, mant * 4.0).astype(np.float32)   # m' in [1,4)
    k   = np.where(odd, (e2 - 1) >> 1, (e2 - 2) >> 1)
    idx = np.zeros(t.shape, dtype=np.int64)
    for s in range(1, slope.size):                        # largest s with mp >= knot[s]
        idx[mp >= knot[s]] = s
    r = slope[idx] * mp + icpt[idx]
    return (r * (2.0 ** (-k.astype(np.float64)))).astype(np.float32)


def _pwl_max_rel_err(seg: int) -> float:
    knot, slope, icpt = _build_pwl_rsqrt_table(seg)
    mm = np.linspace(1.0, 4.0, 400000).astype(np.float32)
    return float(np.max(np.abs(_pwl_rsqrt_eval(mm, knot, slope, icpt) - mm ** -0.5) / (mm ** -0.5)))


def _suggest_segments(tol: float, lo: int = 2, hi: int = 64) -> int:
    """Tuning: the fewest segments whose worst-case rsqrt rel-err over [1,4) is <= tol (the 'max rule')."""
    for seg in range(lo, hi + 1):
        if _pwl_max_rel_err(seg) <= tol:
            return seg
    return hi


def _validate_pwl(knot, slope, icpt, seg: int) -> None:
    """rsqrt rel-err over a huge range (range reduction => range-independent) + end-to-end LayerNorm SNR."""
    rng = np.random.default_rng(0)
    eps = 1e-5
    tt  = np.exp(rng.uniform(np.log(1e-10), np.log(1e10), 500000)).astype(np.float32)
    rel = np.abs(_pwl_rsqrt_eval(tt, knot, slope, icpt) - tt ** -0.5) / (tt ** -0.5)
    C, L = 16, 4096
    x = (rng.standard_normal((C, L)) * rng.uniform(0.01, 50.0, (C, 1))).astype(np.float32)
    g = rng.uniform(0.5, 1.5, C).astype(np.float32)
    b = rng.uniform(-0.3, 0.3, C).astype(np.float32)
    mean, var = x.mean(0), x.var(0)
    ref  = ((x - mean) * (1.0 / np.sqrt(var + eps)))                     * g[:, None] + b[:, None]
    appx = ((x - mean) *  _pwl_rsqrt_eval(var + eps, knot, slope, icpt)) * g[:, None] + b[:, None]
    snr = 10.0 * np.log10(np.sum(ref ** 2) / np.sum((ref - appx) ** 2))
    print(f"  PWL rsqrt check: segments={seg} wide_max_rel={rel.max():.3e} LayerNorm_SNR={snr:.2f} dB")
    assert rel.max() <= 2.0e-3, "PWL rsqrt rel error too high - retune PWL_RSQRT_SEGMENTS"


def _quantize_multiplier_searched(M: float):
    """Approximate real multiplier M (>0) by (mul, shift) with mul/2^shift, searching every
    shift in [SHIFT_MIN, SHIFT_MAX]. Returns (mul:int, shift:int, mism:int, maxlsb:int, n:int)."""
    if not (M > 0.0 and np.isfinite(M)):
        raise ValueError(f"non-positive/non-finite requant multiplier: {M}")
    span = max(1.0, 140.0 / M)                                   # cover the unsaturated band + margin
    acc = np.unique(np.rint(np.linspace(-span, span, SEARCH_SAMPLES)).astype(np.int64))
    qf = np.clip(np.rint(acc.astype(np.float64) * M), -128, 127).astype(np.int32)   # exact f32 requant
    best = None
    for s in range(SHIFT_MIN, SHIFT_MAX + 1):
        mul = int(round(M * (2.0 ** s)))
        if mul <= 0 or mul >= 2 ** 31:
            continue
        bias = 1 << (s - 1)
        qi = np.clip((acc * mul + bias) >> s, -128, 127).astype(np.int32)
        mism = int(np.count_nonzero(qf != qi))
        maxlsb = int(np.abs(qf - qi).max()) if acc.size else 0
        key = (mism, maxlsb, -s)                                 # fewest mismatches, smallest LSB, largest shift
        if best is None or key < best[0]:
            best = (key, mul, s, mism, maxlsb)
    if best is None:
        raise ValueError(f"no valid (mul,shift) found for M={M}")
    _, mul, s, mism, maxlsb = best
    return mul, s, mism, maxlsb, int(acc.size)


def _build_requant_fixed(extra_in, member_base):
    """From f32 requant_scale records derive per-channel integer (mul, shift) via the searched
    dyadic multiplier. member_base is 'mlp_requant' (v6 MLP) or 'ssm_requant' (v7 SSM x-branch).
    Returns (extra_field_records, triples) where triples = (stage, scale, mul, shift)."""
    extra, triples = [], []
    for field_name, scale, _src, _note in extra_in:
        stage = field_name.split(".", 1)[0]
        scale = np.asarray(scale, dtype=np.float32).reshape(-1)
        mul = np.empty(scale.size, dtype=np.int32)
        shift = np.empty(scale.size, dtype=np.int32)
        for c, M in enumerate(scale):
            mul[c], shift[c], _, _, _ = _quantize_multiplier_searched(float(M))
        extra.append((f"{stage}.{member_base}_mul", np.ascontiguousarray(mul),
                      f"derived: search round({member_base}_scale * 2^shift)", "fixed-point requant multiplier"))
        extra.append((f"{stage}.{member_base}_shift", np.ascontiguousarray(shift),
                      f"derived: best right-shift for {member_base}_scale", "fixed-point requant shift"))
        triples.append((stage, scale, mul, shift))
    return extra, triples


def _validate_fixed(triples, n_per_chan=4096):
    """Per-stage: integer (mul,shift) requant vs the exact f32 requant on random int32 accs."""
    rng = np.random.default_rng(0)
    tot = mism = worst = 0
    for stage, scale, mul, shift in triples:
        C = scale.size
        span = 140.0 / np.maximum(scale.astype(np.float64), 1e-30)
        acc = (rng.uniform(-1, 1, (C, n_per_chan)) * span[:, None]).astype(np.int64)
        qf = np.clip(np.rint(acc.astype(np.float64) * scale.astype(np.float64)[:, None]), -128, 127).astype(np.int32)
        bias = (1 << (shift.astype(np.int64) - 1))[:, None]
        qi = np.clip((acc * mul.astype(np.int64)[:, None] + bias) >> shift.astype(np.int64)[:, None],
                     -128, 127).astype(np.int32)
        m = qf != qi
        tot += acc.size
        mism += int(m.sum())
        if m.any():
            worst = max(worst, int(np.abs(qf[m].astype(int) - qi[m].astype(int)).max()))
    print(f"  fixed-point requant check: stages={len(triples)} elems={tot} mismatches={mism} "
          f"rate={mism/max(tot,1):.2e} max|LSB diff|={worst}")
    assert worst <= 1, "integer requant diverges by more than 1 LSB - check shift search"


def main() -> None:
    pack = torch.load(PACK_PATH, map_location="cpu", weights_only=False)
    qp = pack["qpacks"]
    nq = pack["non_quant_state_dict"]
    v0.PACK_PATH = PACK_PATH

    consumed: set = set()
    fields = v0._build_fields(nq, qp, consumed)
    leftover = sorted(set(qp) - consumed)
    expected_dead = sorted(n for n in qp if n.endswith("mlp.dim_reduce"))
    assert leftover == expected_dead, f"unexpected unconsumed qpacks: {set(leftover) ^ set(expected_dead)}"

    mlp_extra, mlp_triples = v4._build_requant(fields)        # v4 fc1->fc2 requant scale (unchanged)
    ssm_extra, ssm_triples = v5._build_ssm_requant(fields)    # v5 in_proj->dw_conv requant scale (unchanged)
    mlp_fix_extra, mlp_fix_triples = _build_requant_fixed(mlp_extra, "mlp_requant")  # v6 integer MLP requant
    ssm_fix_extra, ssm_fix_triples = _build_requant_fixed(ssm_extra, "ssm_requant")  # v7 integer SSM requant (x)

    # v8: shared PWL rsqrt table for the fused int32 PatchEmbed LayerNorm (data-independent -> the same
    # bytes go into both embed streams). Tuning picks the fewest segments meeting RSQRT_TOL ("max rule").
    suggested = _suggest_segments(RSQRT_TOL)
    print(f"  PWL tuning: fewest segments for rel-err<={RSQRT_TOL:.1e} = {suggested}; using PWL_RSQRT_SEGMENTS={PWL_RSQRT_SEGMENTS}")
    assert PWL_RSQRT_SEGMENTS >= suggested, (
        f"PWL_RSQRT_SEGMENTS={PWL_RSQRT_SEGMENTS} < tuned minimum {suggested}; raise it here AND the "
        f"#define in headers/sq_int8_v8/datatypes.h")
    pwl_knot, pwl_slope, pwl_icpt = _build_pwl_rsqrt_table(PWL_RSQRT_SEGMENTS)
    pwl_extra = []
    for _stage in ("embed_weight_mag", "embed_weight_pha"):
        pwl_extra.append((f"{_stage}.pwl_rsqrt_knot",      pwl_knot,  "derived: adaptive knots on [1,4)",     "v8 PWL rsqrt breakpoints"))
        pwl_extra.append((f"{_stage}.pwl_rsqrt_slope",     pwl_slope, "derived: min-max linear fit of rsqrt", "v8 PWL rsqrt slope"))
        pwl_extra.append((f"{_stage}.pwl_rsqrt_intercept", pwl_icpt,  "derived: min-max linear fit of rsqrt", "v8 PWL rsqrt intercept"))

    fields = fields + mlp_extra + ssm_extra + mlp_fix_extra + ssm_fix_extra + pwl_extra

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    v0._write_stage_exports(fields, OUTPUT_ROOT)

    print(f"Exported {len(fields)} files into {OUTPUT_ROOT}")
    print(f"  added {len(mlp_extra)} mlp_requant_scale.bin (v4)  +  {len(ssm_extra)} ssm_requant_scale.bin (v5)")
    print(f"  added v6 int32: mlp_requant_mul x{len(mlp_fix_triples)} + mlp_requant_shift x{len(mlp_fix_triples)}")
    print(f"  added v7 int32: ssm_requant_mul x{len(ssm_fix_triples)} + ssm_requant_shift x{len(ssm_fix_triples)}")
    print(f"  added v8 PWL f32: {len(pwl_extra)} files ({PWL_RSQRT_SEGMENTS} segments x knot/slope/intercept x 2 embed streams)")
    v4._validate_requant(mlp_triples)
    v4._validate_requant(ssm_triples)
    _validate_fixed(mlp_fix_triples)
    _validate_fixed(ssm_fix_triples)
    _validate_pwl(pwl_knot, pwl_slope, pwl_icpt, PWL_RSQRT_SEGMENTS)
    print("validation passed")


if __name__ == "__main__":
    main()
