#!/usr/bin/env python3
"""Weight extractor for the sq_int8_v7 INT8-F32 mixed-precision C model.

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
Output:            ../data/weight_sq_int8_v7/<stage_dir>/<member>.bin (+ meta/manifest)
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
OUTPUT_ROOT = HERE.parent / "data" / "weight_sq_int8_v7"

SHIFT_MIN, SHIFT_MAX = 1, 30          # all candidate right-shifts to search (mul must stay < 2^31)
SEARCH_SAMPLES = 20000


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
    fields = fields + mlp_extra + ssm_extra + mlp_fix_extra + ssm_fix_extra

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    v0._write_stage_exports(fields, OUTPUT_ROOT)

    print(f"Exported {len(fields)} files into {OUTPUT_ROOT}")
    print(f"  added {len(mlp_extra)} mlp_requant_scale.bin (v4)  +  {len(ssm_extra)} ssm_requant_scale.bin (v5)")
    print(f"  added v6 int32: mlp_requant_mul x{len(mlp_fix_triples)} + mlp_requant_shift x{len(mlp_fix_triples)}")
    print(f"  added v7 int32: ssm_requant_mul x{len(ssm_fix_triples)} + ssm_requant_shift x{len(ssm_fix_triples)}")
    v4._validate_requant(mlp_triples)
    v4._validate_requant(ssm_triples)
    _validate_fixed(mlp_fix_triples)
    _validate_fixed(ssm_fix_triples)
    print("validation passed")


if __name__ == "__main__":
    main()
