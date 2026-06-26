#!/usr/bin/env python3
"""Weight extractor for the sq_int8_v5 INT8-F32 mixed-precision C model.

v5 = v4 weights PLUS one extra per-stage tensor that powers the fused SSM
in_proj -> dw_conv2d REQUANT on the x branch. In v0..v4 the SS2D ran:
  in_proj: quant -> int8 conv -> dequant (int32->f32)
  dw_conv: quant (f32->int8) -> int8 conv -> ...
v5 collapses that dequant + the following quant into ONE int32->int8 op:

    ssm_requant_scale[c] = ssm_in_projection.dequant_scale[c] / ssm_dw_conv2d.quant_scale[c]
                           for c in [0, SUB_D_INNER)            (f32, [SUB_D_INNER])

The SS2D in-projection emits [x | z] = 2*SUB_D_INNER channels; only the first
SUB_D_INNER (x) feed the depthwise conv, so only those are requantized. The z
branch keeps the float dequant (it is consumed by the f32 gate / scan core).
ssm_in_projection.dequant_scale therefore has 2*SUB_D_INNER entries and is sliced
to its first SUB_D_INNER; ssm_dw_conv2d.quant_scale already has SUB_D_INNER entries.

Rationale (see sources/sq_int8_v4/kernels/requant_int32_to_int8_perchannel.c):
  unfused: f[c,l] = (float)acc[c,l] * deq[c];  q = clamp(lrintf(f / act_next[c]), -128, 127)
  fused:   q[c,l] = clamp(lrintf((float)acc[c,l] * requant_scale[c]), -128, 127)
Channel c is shared (in_proj x-out-channel == dw_conv in-channel), so the
per-out-channel dequant scale and the per-in-channel next-layer quant scale
align 1:1. NOT bit-identical (removes one f32 rounding) -> differs only at
rounding ties (empirically < 1e-5 of values, always by +-1 LSB).

v5 also carries v4's fused fc1->fc2 mlp_requant_scale unchanged, so this script
reuses weight_extractor_sq_int8_v4's MLP requant builder/validator verbatim and
only adds the SSM requant files. Base artifacts come from
weight_extractor_sq_int8_v0 (so every non-requant .bin is byte-identical to
v0/v4).

Source checkpoint: model/gen_ptq_int8pack.pth
Output:            ../data/weight_sq_int8_v5/<stage_dir>/<member>.bin (+ meta/manifest)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch

import weight_extractor_sq_int8_v0 as v0
import weight_extractor_sq_int8_v4 as v4

HERE = Path(__file__).resolve().parent
PACK_PATH = HERE / "model" / "gen_ptq_int8pack.pth"
OUTPUT_ROOT = HERE.parent / "data" / "weight_sq_int8_v5"

# SSM in_proj->dw_conv2d boundary is fused only in the int8 PVSS blocks
# (same stages as the MLP requant; the output stages keep an f32 ss2d).
SSM_REQUANT_STAGE_PREFIXES = ("pvss_ds_weight_enc", "pvss_latent_weight", "pvss_us_weight_dec")


def _is_ssm_requant_stage(stage: str) -> bool:
    return any(stage.startswith(p) for p in SSM_REQUANT_STAGE_PREFIXES)


def _build_ssm_requant(fields):
    """For every fused SSM stage, derive ssm_requant_scale = in_proj_deq[:D] / dw_conv_act.
    Returns (extra_field_records, triples) where triples carries (stage, deq_x, act,
    requant) for the numerical self-check."""
    by_key = {}
    for field_name, arr, _src, _note in fields:
        stage, member = field_name.split(".", 1)
        by_key[(stage, member)] = arr

    stages = sorted({fn.split(".", 1)[0] for fn, *_ in fields
                     if _is_ssm_requant_stage(fn.split(".", 1)[0])})
    extra, triples = [], []
    for stage in stages:
        deq = by_key.get((stage, "ssm_in_projection_w_dequant_scale"))
        act = by_key.get((stage, "ssm_dw_conv2d_w_quant_scale"))
        if deq is None or act is None:
            raise KeyError(f"{stage}: missing ssm_in_projection deq / ssm_dw_conv2d act for requant")

        deq = np.ascontiguousarray(np.asarray(deq, dtype=np.float32).reshape(-1))
        act = np.asarray(act, dtype=np.float32).reshape(-1)
        d_inner = act.size                          # SUB_D_INNER (x-branch width)
        if act.size == 1:                           # per-tensor next-layer quant -> broadcast
            act = np.full(deq.size // 2, float(act[0]), dtype=np.float32)
            d_inner = act.size
        if deq.size != 2 * d_inner:
            raise ValueError(f"{stage}: in_proj deq[{deq.size}] != 2*dw_conv act[{d_inner}] (expect [x|z])")
        deq_x = deq[:d_inner]                        # x-branch dequant scales (z keeps the float dequant)
        if not (np.all(deq_x > 0) and np.all(act > 0)):
            raise ValueError(f"{stage}: non-positive scale breaks symmetric requant")

        # one correctly-rounded f32 ratio per channel (compute in f64, store f32)
        requant = np.ascontiguousarray((deq_x.astype(np.float64) / act.astype(np.float64)).astype(np.float32))
        if not np.all(np.isfinite(requant)):
            raise ValueError(f"{stage}: non-finite ssm_requant_scale")
        extra.append((f"{stage}.ssm_requant_scale", requant,
                      "derived: ssm_in_projection.w_dequant_scale[:SUB_D_INNER] / ssm_dw_conv2d.w_quant_scale",
                      "in_proj deq(x) / dw_conv act, fused SSM REQUANT"))
        triples.append((stage, deq_x, act, requant))
    return extra, triples


def main() -> None:
    pack = torch.load(PACK_PATH, map_location="cpu", weights_only=False)
    qp = pack["qpacks"]
    nq = pack["non_quant_state_dict"]

    # make v0's manifest/summary record THIS pack (it reads v0.PACK_PATH at write time)
    v0.PACK_PATH = PACK_PATH

    consumed: set = set()
    fields = v0._build_fields(nq, qp, consumed)

    leftover = sorted(set(qp) - consumed)
    expected_dead = sorted(n for n in qp if n.endswith("mlp.dim_reduce"))
    assert leftover == expected_dead, f"unexpected unconsumed qpacks: {set(leftover) ^ set(expected_dead)}"

    mlp_extra, mlp_triples = v4._build_requant(fields)      # v4 fc1->fc2 requant (unchanged)
    ssm_extra, ssm_triples = _build_ssm_requant(fields)     # v5 in_proj->dw_conv requant (new)
    fields = fields + mlp_extra + ssm_extra

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    v0._write_stage_exports(fields, OUTPUT_ROOT)

    print(f"Exported {len(fields)} files into {OUTPUT_ROOT}")
    print(f"  added {len(mlp_extra)} mlp_requant_scale.bin (v4 fc1->fc2: enc x6, latent x2, dec x6)")
    print(f"  added {len(ssm_extra)} ssm_requant_scale.bin (v5 in_proj->dw_conv: enc x6, latent x2, dec x6)")
    _validate = v4._validate_requant
    print(" MLP requant:", end=" "); _validate(mlp_triples)
    print(" SSM requant:", end=" "); _validate(ssm_triples)
    print("validation passed")


if __name__ == "__main__":
    main()
