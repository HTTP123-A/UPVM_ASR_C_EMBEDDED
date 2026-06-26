#!/usr/bin/env python3
"""Weight extractor for sq_int8_v10 — NUMERICALLY IDENTICAL TO v9.

v10 is a pure AVX-VNNI vectorization of v9's int8 kernels (bit-exact; see sources/sq_int8_v10
and README §7.10). It introduces NO new weights or hyperparameters: this script produces bytes
byte-identical to v9 and writes them into data/weight_sq_int8_v10/ only so the folder structure
matches the other versions. It reuses v9's calibration (calib_v9_s_ln.json) unchanged. (For the
build you may instead just copy data/weight_sq_int8_v9/ -> data/weight_sq_int8_v10/.)

--- v9 datapath (unchanged) -------------------------------------------------------------------
Weight extractor for sq_int8_v9: fully-INTEGER PatchEmbed LayerNorm (norm1 int8-out, norm2 f32-out)
+ per-channel int8 GELU LUT (fused dequant+GELU+quant). v9 = v8 weights byte-identical PLUS the
fixed-point LN tables and the GELU LUT. Math is the proven integer datapath (see Q-formats below);
validated bit-exact in /tmp/v9_intexact.py (norm2 ~64 dB, norm1 int8 ~50 dB floor, all int64-safe).

S_ln (norm1 int8 output scale, per stream) is CALIBRATED by model_pytorch/calibrate_v9.py
(forward hook on norm1 over real wavs) -> model_pytorch/calib_v9_s_ln.json. Run that first.

Datapath (matches sources/sq_int8_v9/kernels/layernorm_int_q.c):
  xq[c]=(acc[c]*deq_mul[c])>>deq_sh (Q.F); mean=Sxq>>log2C; d=xq-mean; V=Sd*d>>log2C; t=V+EPSP
  rsqrt(t)->(rm Q.QG, k);  n=(d*rm)>>(QG+k-QN) (Q.QN);  y=n*gamma_q (Q.(QN+QC));  yq=y>>(QN+QC-QY)
  norm1->int8: clamp((yq*amul + bfix[c] + (1<<(ash-1)))>>ash)
  norm2->f32 : (float)yq/2^QY + norm2_b[c]
  GELU[c][q] = clamp(round(GELU(q*S_ln)/conv2.act_scale[c])),  q in [-128,127]
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import torch

import weight_extractor_sq_int8_v0 as v0
import weight_extractor_sq_int8_v4 as v4
import weight_extractor_sq_int8_v5 as v5
import weight_extractor_sq_int8_v8 as v8

HERE = Path(__file__).resolve().parent
PACK_PATH = HERE / "model" / "gen_ptq_int8pack.pth"
OUTPUT_ROOT = HERE.parent / "data" / "weight_sq_int8_v10"  # v10 == v9 bytes; never touch the v9 dir
CALIB_JSON = HERE / "calib_v9_s_ln.json"

# fixed-point formats (MUST match headers/sq_int8_v9/datatypes.h)
F, QM, QC, QG, QN, QY, SEG = 12, 14, 20, 20, 20, 16, 10
EPS = 1e-5
EPSP = int(round(EPS * (1 << (2 * F))))          # 168
AMUL_MAXBITS = 14
STREAMS = {"embed_weight_mag": "mag", "embed_weight_pha": "pha"}
C_MID, C_OUT = 8, 16


def _get(fields, name):
    for n, arr, _s, _t in fields:
        if n == name:
            return np.asarray(arr)
    raise KeyError(name)


def _deq_fold(deq):
    M = deq.astype(np.float64) * (1 << F)
    sh = 0
    while M.max() * (1 << (sh + 1)) < (1 << 30):
        sh += 1
    mul = np.round(M * (1 << sh)).astype(np.int64)
    assert mul.max() < (1 << 31), "deq_mul overflow int32"
    return mul.astype(np.int32), int(sh)


def _int_rsqrt_lut():
    knot, sl, ic = v8._build_pwl_rsqrt_table(SEG)          # f32 knot[SEG+1], slope[SEG], icpt[SEG]
    knot_q = np.round(knot.astype(np.float64) * (1 << QM)).astype(np.int32)
    sl_q = np.round(sl.astype(np.float64) * (1 << QC)).astype(np.int32)
    ic_q = np.round(ic.astype(np.float64) * (1 << QC)).astype(np.int32)
    return knot_q, sl_q, ic_q


def _build_norm1(gamma, beta, deq, S_ln):
    deq_mul, deq_sh = _deq_fold(deq)
    gamma_q = np.round(gamma.astype(np.float64) * (1 << QC)).astype(np.int32)
    A = 1.0 / ((1 << QY) * S_ln)
    ash = 0
    while A * (1 << (ash + 1)) < (1 << AMUL_MAXBITS):
        ash += 1
    # shrink ash if needed so |bfix| stays < 2^30 (int32-safe)
    while ash > 1 and np.abs(np.round((beta.astype(np.float64) / S_ln) * (1 << ash))).max() >= (1 << 30):
        ash -= 1
    amul = int(round(A * (1 << ash)))
    bfix = np.round((beta.astype(np.float64) / S_ln) * (1 << ash)).astype(np.int32)
    assert 0 < amul < (1 << 31) and np.abs(bfix).max() < (1 << 31)
    return dict(deq_mul=deq_mul, deq_sh=deq_sh, gamma_q=gamma_q,
                amul=np.int32(amul), ash=np.int32(ash), bfix=bfix)


def _build_norm2(gamma, deq):
    deq_mul, deq_sh = _deq_fold(deq)
    gamma_q = np.round(gamma.astype(np.float64) * (1 << QC)).astype(np.int32)
    return dict(deq_mul=deq_mul, deq_sh=deq_sh, gamma_q=gamma_q)


def _build_gelu_lut(S_ln, conv2_act):
    q = np.arange(-128, 128, dtype=np.float64)
    gv = 0.5 * (q * S_ln) * (1.0 + np.vectorize(math.erf)((q * S_ln) / math.sqrt(2.0)))
    lut = np.empty((conv2_act.size, 256), dtype=np.int8)
    for c in range(conv2_act.size):
        lut[c] = np.clip(np.round(gv / float(conv2_act[c])), -128, 127).astype(np.int8)
    return lut


# ---- bit-exact reference of the C kernel (for self-validation) ----
def _ref_int_rsqrt(t, knot_q, sl_q, ic_q):
    t = int(t); p = t.bit_length() - 1; k = p >> 1
    sh = 2 * k - QM; m4q = (t >> sh) if sh >= 0 else (t << (-sh))
    s = 0
    while s + 1 < len(knot_q) and m4q >= int(knot_q[s + 1]):
        s += 1
    r = int(sl_q[s]) * m4q + (int(ic_q[s]) << QM)
    return (r >> (QC + QM - QG)), k


def _ref_ln(acc, p, knot_q, sl_q, ic_q, out_int8, S_ln=None, beta=None):
    C, L = acc.shape; log2C = int(round(math.log2(C)))
    out = np.empty((C, L), np.int8 if out_int8 else np.float32)
    dm, ds, gq = p["deq_mul"], int(p["deq_sh"]), p["gamma_q"]
    for l in range(L):
        xq = [(int(acc[c, l]) * int(dm[c])) >> ds for c in range(C)]
        Mn = sum(xq) >> log2C; d = [x - Mn for x in xq]
        V = (sum(di * di for di in d) >> log2C) + EPSP
        rm, k = _ref_int_rsqrt(V, knot_q, sl_q, ic_q); shn = QG + k - QN
        for c in range(C):
            dr = d[c] * rm; n = (dr >> shn) if shn >= 0 else (dr << (-shn))
            yq = (n * int(gq[c])) >> (QN + QC - QY)
            if out_int8:
                ash = int(p["ash"]); q = (yq * int(p["amul"]) + int(p["bfix"][c]) + (1 << (ash - 1))) >> ash
                out[c, l] = max(-128, min(127, q))
            else:
                out[c, l] = np.float32(yq / float(1 << QY) + float(beta[c]))
    return out


def _validate(fields, s_ln, knot_q, sl_q, ic_q):
    rng = np.random.default_rng(0)
    for stage, sk in STREAMS.items():
        g1 = _get(fields, f"{stage}.norm1_w"); b1 = _get(fields, f"{stage}.norm1_b")
        d1 = _get(fields, f"{stage}.conv1_w_dequant_scale")
        g2 = _get(fields, f"{stage}.norm2_w"); b2 = _get(fields, f"{stage}.norm2_b")
        d2 = _get(fields, f"{stage}.conv2_w_dequant_scale")
        p1 = _build_norm1(g1, b1, d1, s_ln[sk]); p2 = _build_norm2(g2, d2)
        for (C, deq, gam, bet, pp, oi, beta) in ((8, d1, g1, b1, p1, True, None),
                                                 (16, d2, g2, b2, p2, False, b2)):
            acc = (rng.standard_normal((C, 1500)) * (3e4 / max(float(deq.max()), 1e-9) * float(deq.mean()))).astype(np.int64)
            x = acc.astype(np.float64) * deq[:, None]; m = x.mean(0); v = x.var(0)
            ref = (x - m) / np.sqrt(v + EPS) * gam[:, None] + bet[:, None]
            o = _ref_ln(acc, pp, knot_q, sl_q, ic_q, oi, s_ln[sk], beta)
            if oi:
                r8 = np.clip(np.round(ref / s_ln[sk]), -128, 127)
                snr = 10 * np.log10((r8 ** 2).sum() / (((r8 - o) ** 2).sum() + 1e-30))
                print(f"  {sk} norm1 int8 SNR={snr:.2f} dB"); assert snr > 44
            else:
                snr = 10 * np.log10((ref ** 2).sum() / (((ref - o) ** 2).sum() + 1e-30))
                print(f"  {sk} norm2 f32  SNR={snr:.2f} dB"); assert snr > 58


def main():
    s_ln = json.loads(CALIB_JSON.read_text())
    pack = torch.load(PACK_PATH, map_location="cpu", weights_only=False)
    qp, nq = pack["qpacks"], pack["non_quant_state_dict"]; v0.PACK_PATH = PACK_PATH

    consumed: set = set()
    fields = v0._build_fields(nq, qp, consumed)
    mlp_extra, mlp_tr = v4._build_requant(fields)
    ssm_extra, ssm_tr = v5._build_ssm_requant(fields)
    mlp_fix, _ = v8._build_requant_fixed(mlp_extra, "mlp_requant")
    ssm_fix, _ = v8._build_requant_fixed(ssm_extra, "ssm_requant")
    pwl_knot, pwl_slope, pwl_icpt = v8._build_pwl_rsqrt_table(v8.PWL_RSQRT_SEGMENTS)
    pwl_extra = []
    for st in STREAMS:
        pwl_extra += [(f"{st}.pwl_rsqrt_knot", pwl_knot, "v8", "v8 f32 pwl (kept)"),
                      (f"{st}.pwl_rsqrt_slope", pwl_slope, "v8", "v8"),
                      (f"{st}.pwl_rsqrt_intercept", pwl_icpt, "v8", "v8")]
    base = fields + mlp_extra + ssm_extra + mlp_fix + ssm_fix + pwl_extra

    knot_q, sl_q, ic_q = _int_rsqrt_lut()
    v9_extra = []
    for stage, sk in STREAMS.items():
        g1 = _get(base, f"{stage}.norm1_w"); b1 = _get(base, f"{stage}.norm1_b")
        d1 = _get(base, f"{stage}.conv1_w_dequant_scale")
        g2 = _get(base, f"{stage}.norm2_w"); d2 = _get(base, f"{stage}.conv2_w_dequant_scale")
        c2a = _get(base, f"{stage}.conv2_w_quant_scale")
        p1 = _build_norm1(g1, b1, d1, s_ln[sk]); p2 = _build_norm2(g2, d2)
        lut = _build_gelu_lut(s_ln[sk], c2a)

        def rec(member, arr):
            v9_extra.append((f"{stage}.{member}", np.ascontiguousarray(arr), "v9", "v9"))
        rec("norm1_deq_mul", p1["deq_mul"]); rec("norm1_deq_sh", np.array([p1["deq_sh"]], np.int32))
        rec("norm1_gamma_q", p1["gamma_q"]); rec("norm1_amul", np.array([p1["amul"]], np.int32))
        rec("norm1_ash", np.array([p1["ash"]], np.int32)); rec("norm1_bfix", p1["bfix"])
        rec("norm2_deq_mul", p2["deq_mul"]); rec("norm2_deq_sh", np.array([p2["deq_sh"]], np.int32))
        rec("norm2_gamma_q", p2["gamma_q"])
        rec("rsqrt_knot_q", knot_q); rec("rsqrt_sl_q", sl_q); rec("rsqrt_ic_q", ic_q)
        rec("gelu_lut", lut.reshape(-1))

    fields_all = base + v9_extra
    import shutil
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    v0._write_stage_exports(fields_all, OUTPUT_ROOT)
    print(f"Exported {len(fields_all)} files into {OUTPUT_ROOT}  (+{len(v9_extra)} v9 int/lut files)")
    print(f"S_ln: {s_ln.get('mag')}, {s_ln.get('pha')}   EPSP={EPSP}")
    _validate(base, s_ln, knot_q, sl_q, ic_q)
    print("validation passed")


if __name__ == "__main__":
    main()
