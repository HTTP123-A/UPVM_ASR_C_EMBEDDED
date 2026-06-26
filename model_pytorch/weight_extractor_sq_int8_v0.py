#!/usr/bin/env python3
"""Weight extractor for the sq_int8_v0 INT8-F32 mixed-precision C model.

Source checkpoint: model/gen_ptq_int8pack.pth (W8A8 PTQ pack, ptq_int8_pack_v2).
Output:            data/weight_sq_int8_v0/<stage_dir>/<member>.bin (+ meta/manifest)

Mirrors weight_extractor.py (f32 version): same stage dirs (v3 datatypes.h names),
same member names, same layout transforms, same derived fields (A = -exp(A_logs)).

Float members  -> <member>.bin                  float32
INT8 members   -> <member>_w.bin                int8   (kernel-ready, fold applied)
                  <member>_w_quant_scale.bin    float32 [C_IN] or [1]  (activation quant)
                  <member>_w_dequant_scale.bin  float32 [C_OUT]        (int32 acc -> f32)
                  <member>_b_i32.bin            int32   [C_OUT]        (only if ckpt bias)

Fold rules (PROGRESS.md section 1b, verified against utils/ptq_w8a8.py):
  quant_scale[c] = sq_s[c] * a_scale[c] * a_scale_mul     (sq_s only when numel > 1)
  - depthwise / per-tensor a_scale: weights exported as stored;
      dequant_scale = w_scale * a_scale * a_scale_mul
  - groups==1 with per-channel a_scale: activation scale folded into the weights,
      weights re-quantized per out-channel; dequant_scale = new per-OC scale.
  bias_i32 = round(bias_f32 / dequant_scale[oc])
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from weight_extractor import _as_f32_np, _conv1d_to_2d, _conv1x1_to_2d, _dw2d_to_3d

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "UPVM_ASR_C_EMBEDDED" / "model_pytorch" / "model"
PACK_PATH = MODEL_DIR / "gen_ptq_int8pack.pth"
OUTPUT_ROOT = REPO_ROOT / "UPVM_ASR_C_EMBEDDED" / "data" / "weight_sq_int8_v0"

# (field_name "stage.member_file_stem", np array, source key, note)
FieldRecord = Tuple[str, np.ndarray, str, str]


# ----------------------------------------------------------------------------
# 1) qpack fold -> kernel-ready artifacts
# ----------------------------------------------------------------------------
def _fold_qpack(pack: dict) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Optional[np.ndarray], bool]:
    """Returns (w_int8 [ckpt layout], quant_scale, dequant_scale, bias_i32|None, refolded)."""
    w_q8 = pack["weight_int8"]
    w_scale = pack["w_scale"].float().reshape(-1)            # [O]
    a_scale = pack["a_scale"].float().reshape(-1)            # [Cin] or [1]
    mul = float(pack["a_scale_mul"])
    sq = pack.get("sq_s", None)
    sq = sq.float().reshape(-1) if (sq is not None and sq.numel() > 1) else None

    conv = pack.get("conv")
    groups = int(conv["groups"]) if conv else 1
    depthwise = conv is not None and groups != 1

    # effective activation quant scale used by the C quant kernel (one fused divide)
    quant_scale = a_scale * mul
    if sq is not None:
        quant_scale = quant_scale * sq                       # groups==1 only (ptq_w8a8.py)

    if depthwise or a_scale.numel() == 1:
        # exact path: no weight refold needed
        dequant_scale = w_scale * a_scale * mul if a_scale.numel() > 1 else w_scale * float(a_scale.item()) * mul
        w_export = w_q8
        refolded = False
    else:
        # groups==1, per-channel a_scale: fold act scale into the weights, requantize per OC
        view = [1] * w_q8.dim()
        view[0] = -1
        w_fp = w_q8.float() * w_scale.view(view)             # exact fake-quant weights
        view_in = [1] * w_q8.dim()
        view_in[1] = -1
        w_fold = w_fp * (a_scale * mul).view(view_in)        # absorb per-Cin act scale
        new_scale = w_fold.reshape(w_fold.shape[0], -1).abs().amax(dim=1) / 127.0
        new_scale = new_scale.clamp(min=1e-12)
        w_export = torch.clamp(torch.round(w_fold / new_scale.view(view)), -128, 127).to(torch.int8)
        dequant_scale = new_scale
        refolded = True

    bias = pack.get("bias", None)
    bias_i32 = None
    if bias is not None:
        bias_i32 = torch.round(bias.float() / dequant_scale).to(torch.int32).numpy()

    return (
        w_export,
        np.ascontiguousarray(quant_scale.numpy().astype(np.float32)),
        np.ascontiguousarray(dequant_scale.numpy().astype(np.float32)),
        bias_i32,
        refolded,
    )


# ----------------------------------------------------------------------------
# 2) record helpers
# ----------------------------------------------------------------------------
def _rec_f32(fields: List[FieldRecord], field: str, arr, src: str, note: str = "") -> None:
    fields.append((field, _as_f32_np(arr), src, note))


_W_TRANSFORM = {
    "conv1x1": _conv1x1_to_2d,   # [O,I,1,1] or [O,I] -> [O,I]
    "dw": _dw2d_to_3d,           # [C,1,k,k] -> [C,k,k]
    "conv4d": lambda w: w.contiguous(),  # regular conv keeps [O,I,k,k]
}


def _rec_qconv(fields: List[FieldRecord], qpacks: Dict[str, dict], consumed: set,
               field: str, ckpt_name: str, kind: str) -> None:
    pack = qpacks[ckpt_name]
    consumed.add(ckpt_name)
    w_i8, quant_scale, dequant_scale, bias_i32, refolded = _fold_qpack(pack)
    w_i8 = _W_TRANSFORM[kind](w_i8)
    w_np = np.ascontiguousarray(w_i8.numpy().astype(np.int8, copy=False))
    note = "act scale folded, requantized per OC" if refolded else "as stored (no refold)"
    fields.append((f"{field}_w", w_np, f"{ckpt_name}.weight_int8", note))
    fields.append((f"{field}_w_quant_scale", quant_scale, f"{ckpt_name}.a_scale",
                   "sq_s*a_scale*a_scale_mul" if quant_scale.size > 1 else "per-tensor a_scale*a_scale_mul"))
    fields.append((f"{field}_w_dequant_scale", dequant_scale, f"{ckpt_name}.w_scale",
                   "per-OC; int32 acc -> f32"))
    if bias_i32 is not None:
        fields.append((f"{field}_b_i32", np.ascontiguousarray(bias_i32), f"{ckpt_name}.bias",
                       "round(bias_f32 / dequant_scale)"))


def _rec_ss2d_f32_common(fields: List[FieldRecord], nq: Dict[str, torch.Tensor],
                         out: str, blk: str) -> None:
    """Float SSM internals shared by every block (x_proj / dt / A / Ds / out_norm)."""
    _rec_f32(fields, f"{out}.ssm_dw_conv1d_w", _conv1d_to_2d(nq[f"{blk}.ss2d.x_proj_conv.weight"]), f"{blk}.ss2d.x_proj_conv.weight")
    _rec_f32(fields, f"{out}.ssm_dt_projection_w", _conv1d_to_2d(nq[f"{blk}.ss2d.dt_proj_conv.weight"]), f"{blk}.ss2d.dt_proj_conv.weight")
    _rec_f32(fields, f"{out}.ssm_A", -torch.exp(nq[f"{blk}.ss2d.A_logs"].float()), f"{blk}.ss2d.A_logs", "A = -exp(A_logs)")
    _rec_f32(fields, f"{out}.ssm_Ds", nq[f"{blk}.ss2d.Ds"], f"{blk}.ss2d.Ds")
    _rec_f32(fields, f"{out}.ssm_delta_bias", nq[f"{blk}.ss2d.dt_projs_bias"].reshape(-1), f"{blk}.ss2d.dt_projs_bias")
    _rec_f32(fields, f"{out}.ssm_out_norm_w", nq[f"{blk}.ss2d.out_norm.weight"], f"{blk}.ss2d.out_norm.weight")
    _rec_f32(fields, f"{out}.ssm_out_norm_b", nq[f"{blk}.ss2d.out_norm.bias"], f"{blk}.ss2d.out_norm.bias")


def _rec_ss2d_f32_convs(fields: List[FieldRecord], nq: Dict[str, torch.Tensor],
                        out: str, blk: str) -> None:
    """Float ss2d convs (output stage only: in_proj / depthwise / out_proj are NOT quantized)."""
    _rec_f32(fields, f"{out}.ssm_in_projection_w", _conv1x1_to_2d(nq[f"{blk}.ss2d.in_proj.weight"]), f"{blk}.ss2d.in_proj.weight")
    _rec_f32(fields, f"{out}.ssm_dw_conv2d_w", _dw2d_to_3d(nq[f"{blk}.ss2d.conv2d.weight"]), f"{blk}.ss2d.conv2d.weight")
    _rec_f32(fields, f"{out}.ssm_dw_conv2d_b", nq[f"{blk}.ss2d.conv2d.bias"], f"{blk}.ss2d.conv2d.bias")
    _rec_f32(fields, f"{out}.ssm_out_projection_w", _conv1x1_to_2d(nq[f"{blk}.ss2d.out_proj.weight"]), f"{blk}.ss2d.out_proj.weight")


# ----------------------------------------------------------------------------
# 3) per-stage extractors (v3 datatypes.h)
# ----------------------------------------------------------------------------
def _extract_patch(nq, qp, consumed, pt: str, out: str, fields) -> None:
    _rec_qconv(fields, qp, consumed, f"{out}.conv1", f"{pt}.0", "conv4d")
    _rec_f32(fields, f"{out}.norm1_w", nq[f"{pt}.2.weight"], f"{pt}.2.weight")
    _rec_f32(fields, f"{out}.norm1_b", nq[f"{pt}.2.bias"], f"{pt}.2.bias")
    _rec_qconv(fields, qp, consumed, f"{out}.conv2", f"{pt}.5", "conv4d")
    _rec_f32(fields, f"{out}.norm2_w", nq[f"{pt}.7.weight"], f"{pt}.7.weight")
    _rec_f32(fields, f"{out}.norm2_b", nq[f"{pt}.7.bias"], f"{pt}.7.bias")


def _extract_pvss_ds(nq, qp, consumed, pt: str, out: str, fields) -> None:
    _rec_f32(fields, f"{out}.ssm_in_norm_w", nq[f"{pt}.norm.weight"], f"{pt}.norm.weight")
    _rec_f32(fields, f"{out}.ssm_in_norm_b", nq[f"{pt}.norm.bias"], f"{pt}.norm.bias")
    _rec_qconv(fields, qp, consumed, f"{out}.ssm_in_projection", f"{pt}.ss2d.in_proj", "conv1x1")
    _rec_qconv(fields, qp, consumed, f"{out}.ssm_dw_conv2d", f"{pt}.ss2d.conv2d", "dw")
    _rec_ss2d_f32_common(fields, nq, out, pt)
    _rec_qconv(fields, qp, consumed, f"{out}.ssm_out_projection", f"{pt}.ss2d.out_proj", "conv1x1")
    _rec_f32(fields, f"{out}.mlp_norm_w", nq[f"{pt}.norm2.weight"], f"{pt}.norm2.weight")
    _rec_f32(fields, f"{out}.mlp_norm_b", nq[f"{pt}.norm2.bias"], f"{pt}.norm2.bias")
    _rec_qconv(fields, qp, consumed, f"{out}.mlp_fc1", f"{pt}.mlp.fc1", "conv1x1")
    _rec_qconv(fields, qp, consumed, f"{out}.mlp_fc2", f"{pt}.mlp.fc2.mlp_dim_redu", "conv1x1")
    fc_res_w = _conv1x1_to_2d(nq[f"{pt}.skip_proj.skip_dim_redu.weight"])
    _rec_f32(fields, f"{out}.mlp_fc_res_w", fc_res_w, f"{pt}.skip_proj.skip_dim_redu.weight",
             "f32 in ckpt (skip_proj excluded from PTQ)")
    _rec_f32(fields, f"{out}.mlp_fc_res_b", np.zeros((fc_res_w.shape[0],), dtype=np.float32), "<generated>", "zero bias placeholder")
    _rec_f32(fields, f"{out}.mlp_sumpool_w", _dw2d_to_3d(nq[f"{pt}.mlp.fc2.mlp_sumpool.weight"]), f"{pt}.mlp.fc2.mlp_sumpool.weight")
    _rec_f32(fields, f"{out}.mlp_dim_reduce_w", np.zeros((1,), dtype=np.float32), "<unused>", "unused optional field")
    _rec_f32(fields, f"{out}.skip_reduce_w", np.zeros((1,), dtype=np.float32), "<unused>", "unused optional field")


def _extract_pvss_latent(nq, qp, consumed, pt: str, out: str, fields) -> None:
    _rec_f32(fields, f"{out}.ssm_in_norm_w", nq[f"{pt}.norm.weight"], f"{pt}.norm.weight")
    _rec_f32(fields, f"{out}.ssm_in_norm_b", nq[f"{pt}.norm.bias"], f"{pt}.norm.bias")
    _rec_qconv(fields, qp, consumed, f"{out}.ssm_in_projection", f"{pt}.ss2d.in_proj", "conv1x1")
    _rec_qconv(fields, qp, consumed, f"{out}.ssm_dw_conv2d", f"{pt}.ss2d.conv2d", "dw")
    _rec_ss2d_f32_common(fields, nq, out, pt)
    _rec_qconv(fields, qp, consumed, f"{out}.ssm_out_projection", f"{pt}.ss2d.out_proj", "conv1x1")
    _rec_f32(fields, f"{out}.mlp_norm_w", nq[f"{pt}.norm2.weight"], f"{pt}.norm2.weight")
    _rec_f32(fields, f"{out}.mlp_norm_b", nq[f"{pt}.norm2.bias"], f"{pt}.norm2.bias")
    _rec_qconv(fields, qp, consumed, f"{out}.mlp_fc1", f"{pt}.mlp.fc1", "conv1x1")
    _rec_qconv(fields, qp, consumed, f"{out}.mlp_fc2", f"{pt}.mlp.fc2", "conv1x1")


def _extract_pvss_us(nq, qp, consumed, pt: str, out: str, fields) -> None:
    blk = f"{pt}.blocks.0"
    _rec_f32(fields, f"{out}.ssm_in_norm_w", nq[f"{blk}.norm.weight"], f"{blk}.norm.weight")
    _rec_f32(fields, f"{out}.ssm_in_norm_b", nq[f"{blk}.norm.bias"], f"{blk}.norm.bias")
    _rec_qconv(fields, qp, consumed, f"{out}.ssm_in_projection", f"{blk}.ss2d.in_proj", "conv1x1")
    _rec_qconv(fields, qp, consumed, f"{out}.ssm_dw_conv2d", f"{blk}.ss2d.conv2d", "dw")
    _rec_ss2d_f32_common(fields, nq, out, blk)
    _rec_qconv(fields, qp, consumed, f"{out}.ssm_out_projection", f"{blk}.ss2d.out_proj", "conv1x1")
    _rec_f32(fields, f"{out}.mlp_norm_w", nq[f"{blk}.norm2.weight"], f"{blk}.norm2.weight")
    _rec_f32(fields, f"{out}.mlp_norm_b", nq[f"{blk}.norm2.bias"], f"{blk}.norm2.bias")
    _rec_qconv(fields, qp, consumed, f"{out}.mlp_fc1", f"{blk}.mlp.fc1", "conv1x1")
    _rec_qconv(fields, qp, consumed, f"{out}.mlp_fc2", f"{blk}.mlp.fc2", "conv1x1")
    _rec_qconv(fields, qp, consumed, f"{out}.pe", f"{pt}.sampler.expand", "conv1x1")
    _rec_f32(fields, f"{out}.pe_norm_w", nq[f"{pt}.sampler.norm.weight"], f"{pt}.sampler.norm.weight")
    _rec_f32(fields, f"{out}.pe_norm_b", nq[f"{pt}.sampler.norm.bias"], f"{pt}.sampler.norm.bias")


def _extract_vss_output01(nq, qp, consumed, pt: str, out: str, fields, with_norms: bool) -> None:
    blk = f"{pt}.blocks.0"
    if with_norms:  # only output_layer.1 has block norms in the checkpoint
        _rec_f32(fields, f"{out}.ssm_in_norm_w", nq[f"{blk}.norm.weight"], f"{blk}.norm.weight")
        _rec_f32(fields, f"{out}.ssm_in_norm_b", nq[f"{blk}.norm.bias"], f"{blk}.norm.bias")
    _rec_ss2d_f32_convs(fields, nq, out, blk)
    _rec_ss2d_f32_common(fields, nq, out, blk)
    if with_norms:
        _rec_f32(fields, f"{out}.mlp_norm_w", nq[f"{blk}.norm2.weight"], f"{blk}.norm2.weight")
        _rec_f32(fields, f"{out}.mlp_norm_b", nq[f"{blk}.norm2.bias"], f"{blk}.norm2.bias")
    _rec_qconv(fields, qp, consumed, f"{out}.mlp_fc1", f"{blk}.mlp.fc1", "conv1x1")
    _rec_qconv(fields, qp, consumed, f"{out}.mlp_fc2", f"{blk}.mlp.fc2", "conv1x1")
    _rec_qconv(fields, qp, consumed, f"{out}.pe", f"{pt}.sampler.expand", "conv1x1")
    _rec_f32(fields, f"{out}.pe_norm_w", nq[f"{pt}.sampler.norm.weight"], f"{pt}.sampler.norm.weight")
    _rec_f32(fields, f"{out}.pe_norm_b", nq[f"{pt}.sampler.norm.bias"], f"{pt}.sampler.norm.bias")


def _extract_vss_output3(nq, qp, consumed, stream: str, out: str, fields) -> None:
    pre = f"output_layer_{stream}.2"
    blk = f"output_layer_{stream}.3.blocks.0"
    _rec_f32(fields, f"{out}.pre_conv2d_w", _conv1x1_to_2d(nq[f"{pre}.weight"]), f"{pre}.weight")
    _rec_f32(fields, f"{out}.pre_conv2d_b", nq[f"{pre}.bias"], f"{pre}.bias")
    _rec_ss2d_f32_convs(fields, nq, out, blk)
    _rec_ss2d_f32_common(fields, nq, out, blk)
    _rec_qconv(fields, qp, consumed, f"{out}.mlp_fc1", f"{blk}.mlp.fc1", "conv1x1")
    _rec_qconv(fields, qp, consumed, f"{out}.mlp_fc2", f"{blk}.mlp.fc2", "conv1x1")


def _extract_skip(nq, qp, consumed, pt: str, out: str, fields) -> None:
    stream = "mag" if "mag" in pt else "phase"
    _rec_qconv(fields, qp, consumed, f"{out}.skip_conv_dec0", f"{pt}.1.skip_handler", "conv1x1")
    _rec_qconv(fields, qp, consumed, f"{out}.skip_conv_dec1", f"{pt}.2.skip_handler", "conv1x1")
    _rec_qconv(fields, qp, consumed, f"{out}.skip_conv_dec2", f"{pt}.3.skip_handler", "conv1x1")
    _rec_qconv(fields, qp, consumed, f"{out}.skip_conv_out0", f"output_layer_{stream}.0.skip_handler", "conv1x1")


def _build_fields(nq, qp, consumed) -> List[FieldRecord]:
    fields: List[FieldRecord] = []
    _extract_patch(nq, qp, consumed, "patch_embed_mag", "embed_weight_mag", fields)
    _extract_patch(nq, qp, consumed, "patch_embed_phase", "embed_weight_pha", fields)

    for i in (0, 1, 2):
        _extract_pvss_ds(nq, qp, consumed, f"layers_encoder_mag.{i}.blocks.0", f"pvss_ds_weight_enc{i}_mag", fields)
        _extract_pvss_ds(nq, qp, consumed, f"layers_encoder_phase.{i}.blocks.0", f"pvss_ds_weight_enc{i}_pha", fields)

    _extract_pvss_latent(nq, qp, consumed, "layers_encoder_mag.3.blocks.0", "pvss_latent_weight_mag", fields)
    _extract_pvss_latent(nq, qp, consumed, "layers_encoder_phase.3.blocks.0", "pvss_latent_weight_pha", fields)

    _extract_skip(nq, qp, consumed, "layers_decoder_mag", "pvss_us_skip_weight_mag", fields)
    _extract_skip(nq, qp, consumed, "layers_decoder_phase", "pvss_us_skip_weight_pha", fields)

    for i, dec in ((1, "dec0"), (2, "dec1"), (3, "dec2")):
        _extract_pvss_us(nq, qp, consumed, f"layers_decoder_mag.{i}", f"pvss_us_weight_{dec}_mag", fields)
        _extract_pvss_us(nq, qp, consumed, f"layers_decoder_phase.{i}", f"pvss_us_weight_{dec}_pha", fields)

    _extract_vss_output01(nq, qp, consumed, "output_layer_mag.0", "vss_output0_weight_mag", fields, with_norms=False)
    _extract_vss_output01(nq, qp, consumed, "output_layer_phase.0", "vss_output0_weight_pha", fields, with_norms=False)
    _extract_vss_output01(nq, qp, consumed, "output_layer_mag.1", "vss_output1_weight_mag", fields, with_norms=True)
    _extract_vss_output01(nq, qp, consumed, "output_layer_phase.1", "vss_output1_weight_pha", fields, with_norms=True)

    _extract_vss_output3(nq, qp, consumed, "mag", "vss_output3_weight_mag", fields)
    _extract_vss_output3(nq, qp, consumed, "phase", "vss_output3_weight_pha", fields)
    return fields


# ----------------------------------------------------------------------------
# 4) writer (same scheme as weight_extractor.py, dtype-aware)
# ----------------------------------------------------------------------------
def _write_stage_exports(fields: List[FieldRecord], output_root: Path) -> None:
    manifest_rows = []
    stages: Dict[str, List[dict]] = {}

    for field_name, arr, source_key, note in fields:
        stage_name, member_name = field_name.split(".", 1)
        stage_dir = output_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{member_name}.bin"

        arr.tofile(stage_dir / filename)
        row = {
            "stage_name": stage_name,
            "member_name": member_name,
            "filename": filename,
            "relative_path": f"{stage_name}/{filename}",
            "dtype": str(arr.dtype),
            "numel": int(arr.size),
            "shape": "x".join(str(d) for d in arr.shape),
            "source_key": source_key,
            "note": note,
        }
        stages.setdefault(stage_name, []).append(row)
        manifest_rows.append(row)

    for stage_name, rows in stages.items():
        with (output_root / stage_name / "meta.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["member_name", "filename", "dtype", "numel", "shape", "source_key", "note"])
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in writer.fieldnames})

    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage_name", "member_name", "filename", "relative_path", "dtype", "numel", "shape", "source_key", "note"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "pack_path": str(PACK_PATH),
        "output_root": str(output_root),
        "num_stages": len(stages),
        "num_files": len(manifest_rows),
        "stages": sorted(stages.keys()),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


# ----------------------------------------------------------------------------
# 5) numerical validation: exported artifacts vs PyTorch fake-quant reference
# ----------------------------------------------------------------------------
def _validate(qp: Dict[str, dict], names: List[str]) -> None:
    import torch.nn.functional as F

    print("\nvalidation: C int8 pipeline (exported artifacts) vs PyTorch fake-quant reference")
    for name in names:
        pack = qp[name]
        w_i8, quant_scale, dequant_scale, bias_i32, refolded = _fold_qpack(pack)
        conv = pack.get("conv")
        groups = int(conv["groups"]) if conv else 1
        w_ref = pack["weight_int8"].float() * pack["w_scale"].float().view([-1] + [1] * (pack["weight_int8"].dim() - 1))
        cin = pack["weight_int8"].shape[1] * groups
        qs = torch.from_numpy(quant_scale)
        x = torch.randn(1, cin, 6, 7, dtype=torch.float32) * (qs.view(1, -1, 1, 1) if qs.numel() > 1 else qs) * 64.0

        # --- reference: utils/ptq_w8a8.py frozen forward ---
        sq = pack.get("sq_s")
        x_div = x / sq.float().view(1, -1, 1, 1) if (sq is not None and sq.numel() > 1) else x
        s_eff = (pack["a_scale"].float() * float(pack["a_scale_mul"])).reshape(-1)
        xq_ref = torch.clamp(torch.round(x_div / s_eff.view(1, -1, 1, 1)), -128, 127)
        if conv is not None:
            w4 = w_ref if w_ref.dim() == 4 else w_ref[:, :, None, None]
            ref = F.conv2d(xq_ref * s_eff.view(1, -1, 1, 1), w4, pack.get("bias"),
                           stride=conv["stride"], padding=conv["padding"], groups=groups)
        else:
            w4 = w_ref[:, :, None, None]
            ref = F.conv2d(xq_ref * s_eff.view(1, -1, 1, 1), w4, pack.get("bias"))

        # --- C pipeline: quant -> int GEMM -> dequant (+bias_i32 in acc) ---
        xq_c = torch.clamp(torch.round(x / qs.view(1, -1, 1, 1)), -128, 127)
        assert torch.equal(xq_c, xq_ref), f"{name}: activation integers differ"
        w4c = w_i8.float() if w_i8.dim() == 4 else w_i8.float()[:, :, None, None]
        if conv is not None:
            acc = F.conv2d(xq_c, w4c, None, stride=conv["stride"], padding=conv["padding"], groups=groups)
        else:
            acc = F.conv2d(xq_c, w4c, None)
        if bias_i32 is not None:
            acc = acc + torch.from_numpy(bias_i32).float().view(1, -1, 1, 1)
        out = acc * torch.from_numpy(dequant_scale).view(1, -1, 1, 1)

        rel = (out - ref).norm() / ref.norm().clamp(min=1e-12)
        limit = 0.05 if refolded else 0.01
        status = "OK " if rel.item() < limit else "FAIL"
        print(f"  [{status}] {name:55s} refolded={refolded!s:5s} rel_err={rel.item():.3e}")
        assert rel.item() < limit, f"{name}: rel error {rel.item()} exceeds {limit}"


def main() -> None:
    pack = torch.load(PACK_PATH, map_location="cpu", weights_only=False)
    qp: Dict[str, dict] = pack["qpacks"]
    nq: Dict[str, torch.Tensor] = pack["non_quant_state_dict"]

    consumed: set = set()
    fields = _build_fields(nq, qp, consumed)

    leftover = sorted(set(qp) - consumed)
    expected_dead = sorted(n for n in qp if n.endswith("mlp.dim_reduce"))
    assert leftover == expected_dead, f"unexpected unconsumed qpacks: {set(leftover) ^ set(expected_dead)}"

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_stage_exports(fields, OUTPUT_ROOT)

    n_i8 = sum(1 for f in fields if f[1].dtype == np.int8)
    n_i32 = sum(1 for f in fields if f[1].dtype == np.int32)
    n_f32 = sum(1 for f in fields if f[1].dtype == np.float32)
    print(f"Exported {len(fields)} files into {OUTPUT_ROOT}")
    print(f"  int8 weights: {n_i8} | int32 biases: {n_i32} | float32: {n_f32}")
    print(f"  consumed qpacks: {len(consumed)}/110; intentionally skipped (dead in C): {len(leftover)} -> {leftover[:2]} ...")

    _validate(qp, [
        "layers_encoder_mag.1.blocks.0.ss2d.conv2d",     # depthwise, per-channel, no refold
        "layers_encoder_mag.0.blocks.0.ss2d.in_proj",    # per-tensor (C_IN=1), no refold
        "layers_decoder_mag.1.blocks.0.ss2d.in_proj",    # pointwise, per-channel -> refold
        "layers_decoder_mag.1.blocks.0.mlp.fc1",         # linear with bias -> refold
        "layers_decoder_mag.1.skip_handler",             # split2-class 1x1, bias -> refold
        "patch_embed_mag.5",                             # regular conv k3 s2 -> refold
        "output_layer_mag.0.blocks.0.mlp.fc1",           # output-stage int8 (a_scale_mul=1.1 group)
    ])
    print("validation passed")


if __name__ == "__main__":
    main()
