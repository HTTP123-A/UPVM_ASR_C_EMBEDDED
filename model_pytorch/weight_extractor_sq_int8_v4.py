#!/usr/bin/env python3
"""Weight extractor for the sq_int8_v4 INT8-F32 mixed-precision C model.

v4 = v3 weights (== v0 bytes) PLUS one extra per-stage tensor that powers the
fused fc1->fc2 REQUANT (dequant + quant collapsed into one int32->int8 op):

    mlp_requant_scale[c] = mlp_fc1.dequant_scale[c] / mlp_fc2.quant_scale[c]   (f32, [MLP_HIDDEN_DIM])

Rationale (see sources/sq_int8_v4/kernels/requant_int32_to_int8_perchannel.c):
  unfused: f[c,l] = (float)acc[c,l] * deq[c];  q = clamp(lrintf(f / act_next[c]), -128, 127)
  fused:   q[c,l] = clamp(lrintf((float)acc[c,l] * requant_scale[c]), -128, 127)
Channel c is shared (fc1 out-channel == fc2 in-channel), so the per-out-channel
dequant scale and the per-in-channel next-layer quant scale align 1:1.

Only the stages whose MLP fc1->fc2 boundary is fused get the extra file:
  encoder  pvss_ds_weight_enc{0,1,2}_{mag,pha}   (fc1: relu -> sumpool -> requant)
  latent   pvss_latent_weight_{mag,pha}          (fc1: relu -> requant)
  decoder  pvss_us_weight_{dec0,dec1,dec2}_{mag,pha}
The output stages (vss_output*) are deliberately NOT fused in v4 and get no file.

This script reuses weight_extractor_sq_int8_v0.py for ALL base artifacts (so the
two exports are byte-identical except for the added mlp_requant_scale.bin files).

Source checkpoint: model/gen_ptq_int8pack.pth
Output:            ../data/weight_sq_int8_v4/<stage_dir>/<member>.bin (+ meta/manifest)
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch

import weight_extractor_sq_int8_v0 as v0

HERE = Path(__file__).resolve().parent
PACK_PATH = HERE / "model" / "gen_ptq_int8pack.pth"
OUTPUT_ROOT = HERE.parent / "data" / "weight_sq_int8_v4"

# stages whose fc1->fc2 boundary is fused into REQUANT (output stages excluded)
REQUANT_STAGE_PREFIXES = ("pvss_ds_weight_enc", "pvss_latent_weight", "pvss_us_weight_dec")


def _is_requant_stage(stage: str) -> bool:
    return any(stage.startswith(p) for p in REQUANT_STAGE_PREFIXES)


def _build_requant(fields):
    """For every fused stage, derive mlp_requant_scale = fc1_deq / fc2_act.
    Returns (extra_field_records, triples) where triples carries (stage, deq, act,
    requant) for the numerical self-check."""
    by_key = {}
    for field_name, arr, _src, _note in fields:
        stage, member = field_name.split(".", 1)
        by_key[(stage, member)] = arr

    stages = sorted({fn.split(".", 1)[0] for fn, *_ in fields if _is_requant_stage(fn.split(".", 1)[0])})
    extra, triples = [], []
    for stage in stages:
        deq = by_key.get((stage, "mlp_fc1_w_dequant_scale"))
        act = by_key.get((stage, "mlp_fc2_w_quant_scale"))
        if deq is None or act is None:
            raise KeyError(f"{stage}: missing mlp_fc1 dequant / mlp_fc2 quant scale for requant")

        deq = np.ascontiguousarray(np.asarray(deq, dtype=np.float32).reshape(-1))
        act = np.asarray(act, dtype=np.float32).reshape(-1)
        if act.size == 1:                       # per-tensor next-layer quant -> broadcast
            act = np.full_like(deq, float(act[0]))
        if deq.size != act.size:
            raise ValueError(f"{stage}: fc1 deq[{deq.size}] vs fc2 act[{act.size}] length mismatch")
        if not (np.all(deq > 0) and np.all(act > 0)):
            raise ValueError(f"{stage}: non-positive scale breaks symmetric requant")

        # one correctly-rounded f32 ratio per channel (compute in f64, store f32)
        requant = np.ascontiguousarray((deq.astype(np.float64) / act.astype(np.float64)).astype(np.float32))
        if not np.all(np.isfinite(requant)):
            raise ValueError(f"{stage}: non-finite requant_scale")
        extra.append((f"{stage}.mlp_requant_scale", requant,
                      "derived: mlp_fc1.w_dequant_scale / mlp_fc2.w_quant_scale", "fc1 deq / fc2 act, fused REQUANT"))
        triples.append((stage, deq, act, requant))
    return extra, triples


def _validate_requant(triples, n_per_chan=4096):
    """C-exact check per stage: fused REQUANT vs the v0..v3 dequant->quant path on
    random int32 accumulators, using each stage's REAL deq/act scales. lrintf is
    round-half-to-even (np.rint); both paths share the same clamp. They must agree
    except at rounding ties (a few 1e-6 of values, always by exactly 1 LSB)."""
    rng = np.random.default_rng(0)

    def clamp_round(x):                          # lrintf + clamp[-128,127]
        return np.clip(np.rint(x).astype(np.int64), -128, 127).astype(np.int32)

    tot = mism = worst_diff = 0
    for stage, deq, act, requant in triples:
        C = deq.size
        ratio = deq.astype(np.float64) / act.astype(np.float64)
        span = 140.0 / np.maximum(ratio, 1e-30)               # land mostly in the unsaturated band
        acc = (rng.uniform(-1, 1, (C, n_per_chan)) * span[:, None]).astype(np.int64).astype(np.int32)

        # v0..v3 unfused: dequant (acc*deq) then quant (lrintf(f * 1/act))
        f = acc.astype(np.float32) * deq[:, None]
        inv = (np.float32(1.0) / act).astype(np.float32)
        q_unf = clamp_round((f * inv[:, None]).astype(np.float32))
        # v4 fused requant
        q_fus = clamp_round((acc.astype(np.float32) * requant[:, None]).astype(np.float32))

        m = q_unf != q_fus
        tot += acc.size
        mism += int(m.sum())
        if m.any():
            worst_diff = max(worst_diff, int(np.abs(q_unf[m].astype(int) - q_fus[m].astype(int)).max()))
    print(f"  requant check: stages={len(triples)} elems={tot} mismatches={mism} "
          f"rate={mism/max(tot,1):.2e} max|LSB diff|={worst_diff}")
    assert worst_diff <= 1, "requant diverges by more than 1 LSB - check scale alignment"


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

    extra, triples = _build_requant(fields)
    fields = fields + extra

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    v0._write_stage_exports(fields, OUTPUT_ROOT)

    print(f"Exported {len(fields)} files into {OUTPUT_ROOT}")
    print(f"  added {len(extra)} mlp_requant_scale.bin (fused fc1->fc2: enc x6, latent x2, dec x6)")
    _validate_requant(triples)
    print("validation passed")


if __name__ == "__main__":
    main()
