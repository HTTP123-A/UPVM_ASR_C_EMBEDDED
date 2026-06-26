#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "UPVM_ASR_C_EMBEDDED" / "model_pytorch" / "model"
CHECKPOINT_PATH = MODEL_DIR / "checkpoint-best-G.pth"
OUTPUT_ROOT = REPO_ROOT / "UPVM_ASR_C_EMBEDDED" / "data" / "weight_f32"


FieldRecord = Tuple[str, np.ndarray, str, str]


def _install_checkpoint_stubs() -> None:
    if "yacs.config" in sys.modules:
        return

    yacs_mod = types.ModuleType("yacs")
    yacs_config_mod = types.ModuleType("yacs.config")

    class CfgNode(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

        def defrost(self):
            return None

        def freeze(self):
            return None

    yacs_config_mod.CfgNode = CfgNode
    yacs_mod.config = yacs_config_mod
    sys.modules["yacs"] = yacs_mod
    sys.modules["yacs.config"] = yacs_config_mod


def _strip_module(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def _as_f32_np(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().to("cpu").contiguous().numpy()
    else:
        arr = np.asarray(tensor)
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def _conv1x1_to_2d(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() == 4:
        return weight[:, :, 0, 0].contiguous()
    if weight.dim() == 2:
        return weight.contiguous()
    raise ValueError(f"Unsupported 1x1 weight dim: {tuple(weight.shape)}")


def _conv1d_to_2d(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() != 3 or weight.shape[-1] != 1:
        raise ValueError(f"Expected conv1d weight [O,I,1], got {tuple(weight.shape)}")
    return weight[:, :, 0].contiguous()


def _dw2d_to_3d(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() == 4 and weight.shape[1] == 1:
        return weight[:, 0, :, :].contiguous()
    if weight.dim() == 3:
        return weight.contiguous()
    raise ValueError(f"Unsupported depthwise weight dim: {tuple(weight.shape)}")


def _record_weight(fields: List[FieldRecord], field_name: str, arr, source_key: str, note: str = "") -> None:
    fields.append((field_name, _as_f32_np(arr), source_key, note))


def _extract_patch(sd: Dict[str, torch.Tensor], pt_prefix: str, out_prefix: str, fields: List[FieldRecord]) -> None:
    _record_weight(fields, f"{out_prefix}.conv1_w", sd[f"{pt_prefix}.0.weight"], f"{pt_prefix}.0.weight")
    _record_weight(fields, f"{out_prefix}.conv1_b", sd[f"{pt_prefix}.0.bias"], f"{pt_prefix}.0.bias")
    _record_weight(fields, f"{out_prefix}.norm1_w", sd[f"{pt_prefix}.2.weight"], f"{pt_prefix}.2.weight")
    _record_weight(fields, f"{out_prefix}.norm1_b", sd[f"{pt_prefix}.2.bias"], f"{pt_prefix}.2.bias")
    _record_weight(fields, f"{out_prefix}.conv2_w", sd[f"{pt_prefix}.5.weight"], f"{pt_prefix}.5.weight")
    _record_weight(fields, f"{out_prefix}.conv2_b", sd[f"{pt_prefix}.5.bias"], f"{pt_prefix}.5.bias")
    _record_weight(fields, f"{out_prefix}.norm2_w", sd[f"{pt_prefix}.7.weight"], f"{pt_prefix}.7.weight")
    _record_weight(fields, f"{out_prefix}.norm2_b", sd[f"{pt_prefix}.7.bias"], f"{pt_prefix}.7.bias")


def _extract_pvss_ds(sd: Dict[str, torch.Tensor], pt_prefix: str, out_prefix: str, fields: List[FieldRecord]) -> None:
    _record_weight(fields, f"{out_prefix}.ssm_in_norm_w", sd[f"{pt_prefix}.norm.weight"], f"{pt_prefix}.norm.weight")
    _record_weight(fields, f"{out_prefix}.ssm_in_norm_b", sd[f"{pt_prefix}.norm.bias"], f"{pt_prefix}.norm.bias")
    _record_weight(fields, f"{out_prefix}.ssm_in_projection_w", _conv1x1_to_2d(sd[f"{pt_prefix}.ss2d.in_proj.weight"]), f"{pt_prefix}.ss2d.in_proj.weight")
    _record_weight(fields, f"{out_prefix}.ssm_dw_conv2d_w", _dw2d_to_3d(sd[f"{pt_prefix}.ss2d.conv2d.weight"]), f"{pt_prefix}.ss2d.conv2d.weight")
    _record_weight(fields, f"{out_prefix}.ssm_dw_conv2d_b", sd[f"{pt_prefix}.ss2d.conv2d.bias"], f"{pt_prefix}.ss2d.conv2d.bias")
    _record_weight(fields, f"{out_prefix}.ssm_dw_conv1d_w", _conv1d_to_2d(sd[f"{pt_prefix}.ss2d.x_proj_conv.weight"]), f"{pt_prefix}.ss2d.x_proj_conv.weight")
    _record_weight(fields, f"{out_prefix}.ssm_dt_projection_w", _conv1d_to_2d(sd[f"{pt_prefix}.ss2d.dt_proj_conv.weight"]), f"{pt_prefix}.ss2d.dt_proj_conv.weight")
    _record_weight(fields, f"{out_prefix}.ssm_A", -torch.exp(sd[f"{pt_prefix}.ss2d.A_logs"].float()), f"{pt_prefix}.ss2d.A_logs", "A = -exp(A_logs)")
    _record_weight(fields, f"{out_prefix}.ssm_Ds", sd[f"{pt_prefix}.ss2d.Ds"], f"{pt_prefix}.ss2d.Ds")
    _record_weight(fields, f"{out_prefix}.ssm_delta_bias", sd[f"{pt_prefix}.ss2d.dt_projs_bias"].reshape(-1), f"{pt_prefix}.ss2d.dt_projs_bias")
    _record_weight(fields, f"{out_prefix}.ssm_out_norm_w", sd[f"{pt_prefix}.ss2d.out_norm.weight"], f"{pt_prefix}.ss2d.out_norm.weight")
    _record_weight(fields, f"{out_prefix}.ssm_out_norm_b", sd[f"{pt_prefix}.ss2d.out_norm.bias"], f"{pt_prefix}.ss2d.out_norm.bias")
    _record_weight(fields, f"{out_prefix}.ssm_out_projection_w", _conv1x1_to_2d(sd[f"{pt_prefix}.ss2d.out_proj.weight"]), f"{pt_prefix}.ss2d.out_proj.weight")
    _record_weight(fields, f"{out_prefix}.mlp_norm_w", sd[f"{pt_prefix}.norm2.weight"], f"{pt_prefix}.norm2.weight")
    _record_weight(fields, f"{out_prefix}.mlp_norm_b", sd[f"{pt_prefix}.norm2.bias"], f"{pt_prefix}.norm2.bias")
    _record_weight(fields, f"{out_prefix}.mlp_fc1_w", _conv1x1_to_2d(sd[f"{pt_prefix}.mlp.fc1.weight"]), f"{pt_prefix}.mlp.fc1.weight")
    _record_weight(fields, f"{out_prefix}.mlp_fc1_b", sd[f"{pt_prefix}.mlp.fc1.bias"], f"{pt_prefix}.mlp.fc1.bias")

    fc2_w = _conv1x1_to_2d(sd[f"{pt_prefix}.mlp.fc2.mlp_dim_redu.weight"])
    fc_res_w = _conv1x1_to_2d(sd[f"{pt_prefix}.skip_proj.skip_dim_redu.weight"])
    _record_weight(fields, f"{out_prefix}.mlp_fc2_w", fc2_w, f"{pt_prefix}.mlp.fc2.mlp_dim_redu.weight")
    _record_weight(fields, f"{out_prefix}.mlp_fc2_b", np.zeros((fc2_w.shape[0],), dtype=np.float32), "<generated>", "zero bias placeholder")
    _record_weight(fields, f"{out_prefix}.mlp_fc_res_w", fc_res_w, f"{pt_prefix}.skip_proj.skip_dim_redu.weight")
    _record_weight(fields, f"{out_prefix}.mlp_fc_res_b", np.zeros((fc_res_w.shape[0],), dtype=np.float32), "<generated>", "zero bias placeholder")
    _record_weight(fields, f"{out_prefix}.mlp_sumpool_w", _dw2d_to_3d(sd[f"{pt_prefix}.mlp.fc2.mlp_sumpool.weight"]), f"{pt_prefix}.mlp.fc2.mlp_sumpool.weight")
    _record_weight(fields, f"{out_prefix}.mlp_dim_reduce_w", np.zeros((1,), dtype=np.float32), "<unused>", "unused optional field")
    _record_weight(fields, f"{out_prefix}.skip_reduce_w", np.zeros((1,), dtype=np.float32), "<unused>", "unused optional field")


def _extract_pvss_latent(sd: Dict[str, torch.Tensor], pt_prefix: str, out_prefix: str, fields: List[FieldRecord]) -> None:
    _record_weight(fields, f"{out_prefix}.ssm_in_norm_w", sd[f"{pt_prefix}.norm.weight"], f"{pt_prefix}.norm.weight")
    _record_weight(fields, f"{out_prefix}.ssm_in_norm_b", sd[f"{pt_prefix}.norm.bias"], f"{pt_prefix}.norm.bias")
    _record_weight(fields, f"{out_prefix}.ssm_in_projection_w", _conv1x1_to_2d(sd[f"{pt_prefix}.ss2d.in_proj.weight"]), f"{pt_prefix}.ss2d.in_proj.weight")
    _record_weight(fields, f"{out_prefix}.ssm_dw_conv2d_w", _dw2d_to_3d(sd[f"{pt_prefix}.ss2d.conv2d.weight"]), f"{pt_prefix}.ss2d.conv2d.weight")
    _record_weight(fields, f"{out_prefix}.ssm_dw_conv2d_b", sd[f"{pt_prefix}.ss2d.conv2d.bias"], f"{pt_prefix}.ss2d.conv2d.bias")
    _record_weight(fields, f"{out_prefix}.ssm_dw_conv1d_w", _conv1d_to_2d(sd[f"{pt_prefix}.ss2d.x_proj_conv.weight"]), f"{pt_prefix}.ss2d.x_proj_conv.weight")
    _record_weight(fields, f"{out_prefix}.ssm_dt_projection_w", _conv1d_to_2d(sd[f"{pt_prefix}.ss2d.dt_proj_conv.weight"]), f"{pt_prefix}.ss2d.dt_proj_conv.weight")
    _record_weight(fields, f"{out_prefix}.ssm_A", -torch.exp(sd[f"{pt_prefix}.ss2d.A_logs"].float()), f"{pt_prefix}.ss2d.A_logs", "A = -exp(A_logs)")
    _record_weight(fields, f"{out_prefix}.ssm_Ds", sd[f"{pt_prefix}.ss2d.Ds"], f"{pt_prefix}.ss2d.Ds")
    _record_weight(fields, f"{out_prefix}.ssm_delta_bias", sd[f"{pt_prefix}.ss2d.dt_projs_bias"].reshape(-1), f"{pt_prefix}.ss2d.dt_projs_bias")
    _record_weight(fields, f"{out_prefix}.ssm_out_norm_w", sd[f"{pt_prefix}.ss2d.out_norm.weight"], f"{pt_prefix}.ss2d.out_norm.weight")
    _record_weight(fields, f"{out_prefix}.ssm_out_norm_b", sd[f"{pt_prefix}.ss2d.out_norm.bias"], f"{pt_prefix}.ss2d.out_norm.bias")
    _record_weight(fields, f"{out_prefix}.ssm_out_projection_w", _conv1x1_to_2d(sd[f"{pt_prefix}.ss2d.out_proj.weight"]), f"{pt_prefix}.ss2d.out_proj.weight")
    _record_weight(fields, f"{out_prefix}.mlp_norm_w", sd[f"{pt_prefix}.norm2.weight"], f"{pt_prefix}.norm2.weight")
    _record_weight(fields, f"{out_prefix}.mlp_norm_b", sd[f"{pt_prefix}.norm2.bias"], f"{pt_prefix}.norm2.bias")
    _record_weight(fields, f"{out_prefix}.mlp_fc1_w", _conv1x1_to_2d(sd[f"{pt_prefix}.mlp.fc1.weight"]), f"{pt_prefix}.mlp.fc1.weight")
    _record_weight(fields, f"{out_prefix}.mlp_fc1_b", sd[f"{pt_prefix}.mlp.fc1.bias"], f"{pt_prefix}.mlp.fc1.bias")
    _record_weight(fields, f"{out_prefix}.mlp_fc2_w", _conv1x1_to_2d(sd[f"{pt_prefix}.mlp.fc2.weight"]), f"{pt_prefix}.mlp.fc2.weight")
    _record_weight(fields, f"{out_prefix}.mlp_fc2_b", sd[f"{pt_prefix}.mlp.fc2.bias"], f"{pt_prefix}.mlp.fc2.bias")


def _extract_pvss_us(sd: Dict[str, torch.Tensor], pt_prefix: str, out_prefix: str, fields: List[FieldRecord]) -> None:
    dim = int(sd[f"{pt_prefix}.blocks.0.ss2d.in_proj.weight"].shape[1])
    norm_w_key = f"{pt_prefix}.blocks.0.norm.weight"
    norm_b_key = f"{pt_prefix}.blocks.0.norm.bias"
    norm2_w_key = f"{pt_prefix}.blocks.0.norm2.weight"
    norm2_b_key = f"{pt_prefix}.blocks.0.norm2.bias"

    if norm_w_key in sd and norm_b_key in sd:
        _record_weight(fields, f"{out_prefix}.ssm_in_norm_w", sd[norm_w_key], norm_w_key)
        _record_weight(fields, f"{out_prefix}.ssm_in_norm_b", sd[norm_b_key], norm_b_key)
    else:
        _record_weight(fields, f"{out_prefix}.ssm_in_norm_w", np.ones((dim,), dtype=np.float32), "<identity>", "PyTorch block uses Identity norm")
        _record_weight(fields, f"{out_prefix}.ssm_in_norm_b", np.zeros((dim,), dtype=np.float32), "<identity>", "PyTorch block uses Identity norm")

    _record_weight(fields, f"{out_prefix}.ssm_in_projection_w", _conv1x1_to_2d(sd[f"{pt_prefix}.blocks.0.ss2d.in_proj.weight"]), f"{pt_prefix}.blocks.0.ss2d.in_proj.weight")
    _record_weight(fields, f"{out_prefix}.ssm_dw_conv2d_w", _dw2d_to_3d(sd[f"{pt_prefix}.blocks.0.ss2d.conv2d.weight"]), f"{pt_prefix}.blocks.0.ss2d.conv2d.weight")
    _record_weight(fields, f"{out_prefix}.ssm_dw_conv2d_b", sd[f"{pt_prefix}.blocks.0.ss2d.conv2d.bias"], f"{pt_prefix}.blocks.0.ss2d.conv2d.bias")
    _record_weight(fields, f"{out_prefix}.ssm_dw_conv1d_w", _conv1d_to_2d(sd[f"{pt_prefix}.blocks.0.ss2d.x_proj_conv.weight"]), f"{pt_prefix}.blocks.0.ss2d.x_proj_conv.weight")
    _record_weight(fields, f"{out_prefix}.ssm_dt_projection_w", _conv1d_to_2d(sd[f"{pt_prefix}.blocks.0.ss2d.dt_proj_conv.weight"]), f"{pt_prefix}.blocks.0.ss2d.dt_proj_conv.weight")
    _record_weight(fields, f"{out_prefix}.ssm_A", -torch.exp(sd[f"{pt_prefix}.blocks.0.ss2d.A_logs"].float()), f"{pt_prefix}.blocks.0.ss2d.A_logs", "A = -exp(A_logs)")
    _record_weight(fields, f"{out_prefix}.ssm_Ds", sd[f"{pt_prefix}.blocks.0.ss2d.Ds"], f"{pt_prefix}.blocks.0.ss2d.Ds")
    _record_weight(fields, f"{out_prefix}.ssm_delta_bias", sd[f"{pt_prefix}.blocks.0.ss2d.dt_projs_bias"].reshape(-1), f"{pt_prefix}.blocks.0.ss2d.dt_projs_bias")
    _record_weight(fields, f"{out_prefix}.ssm_out_norm_w", sd[f"{pt_prefix}.blocks.0.ss2d.out_norm.weight"], f"{pt_prefix}.blocks.0.ss2d.out_norm.weight")
    _record_weight(fields, f"{out_prefix}.ssm_out_norm_b", sd[f"{pt_prefix}.blocks.0.ss2d.out_norm.bias"], f"{pt_prefix}.blocks.0.ss2d.out_norm.bias")
    _record_weight(fields, f"{out_prefix}.ssm_out_projection_w", _conv1x1_to_2d(sd[f"{pt_prefix}.blocks.0.ss2d.out_proj.weight"]), f"{pt_prefix}.blocks.0.ss2d.out_proj.weight")

    if norm2_w_key in sd and norm2_b_key in sd:
        _record_weight(fields, f"{out_prefix}.mlp_norm_w", sd[norm2_w_key], norm2_w_key)
        _record_weight(fields, f"{out_prefix}.mlp_norm_b", sd[norm2_b_key], norm2_b_key)
    else:
        _record_weight(fields, f"{out_prefix}.mlp_norm_w", np.ones((dim,), dtype=np.float32), "<identity>", "PyTorch block uses Identity norm")
        _record_weight(fields, f"{out_prefix}.mlp_norm_b", np.zeros((dim,), dtype=np.float32), "<identity>", "PyTorch block uses Identity norm")

    _record_weight(fields, f"{out_prefix}.mlp_fc1_w", _conv1x1_to_2d(sd[f"{pt_prefix}.blocks.0.mlp.fc1.weight"]), f"{pt_prefix}.blocks.0.mlp.fc1.weight")
    _record_weight(fields, f"{out_prefix}.mlp_fc1_b", sd[f"{pt_prefix}.blocks.0.mlp.fc1.bias"], f"{pt_prefix}.blocks.0.mlp.fc1.bias")
    _record_weight(fields, f"{out_prefix}.mlp_fc2_w", _conv1x1_to_2d(sd[f"{pt_prefix}.blocks.0.mlp.fc2.weight"]), f"{pt_prefix}.blocks.0.mlp.fc2.weight")
    _record_weight(fields, f"{out_prefix}.mlp_fc2_b", sd[f"{pt_prefix}.blocks.0.mlp.fc2.bias"], f"{pt_prefix}.blocks.0.mlp.fc2.bias")
    _record_weight(fields, f"{out_prefix}.pe_w", _conv1x1_to_2d(sd[f"{pt_prefix}.sampler.expand.weight"]), f"{pt_prefix}.sampler.expand.weight")
    _record_weight(fields, f"{out_prefix}.pe_norm_w", sd[f"{pt_prefix}.sampler.norm.weight"], f"{pt_prefix}.sampler.norm.weight")
    _record_weight(fields, f"{out_prefix}.pe_norm_b", sd[f"{pt_prefix}.sampler.norm.bias"], f"{pt_prefix}.sampler.norm.bias")


def _extract_vss_output(sd: Dict[str, torch.Tensor], stream: str, out_prefix: str, fields: List[FieldRecord]) -> None:
    pre = f"output_layer_{stream}.2"
    blk = f"output_layer_{stream}.3.blocks.0"

    _record_weight(fields, f"{out_prefix}.pre_conv2d_w", _conv1x1_to_2d(sd[f"{pre}.weight"]), f"{pre}.weight")
    _record_weight(fields, f"{out_prefix}.pre_conv2d_b", sd[f"{pre}.bias"], f"{pre}.bias")
    _record_weight(fields, f"{out_prefix}.ssm_in_norm_w", np.ones((1,), dtype=np.float32), "<identity>", "unused placeholder")
    _record_weight(fields, f"{out_prefix}.ssm_in_norm_b", np.zeros((1,), dtype=np.float32), "<identity>", "unused placeholder")
    _record_weight(fields, f"{out_prefix}.ssm_in_projection_w", _conv1x1_to_2d(sd[f"{blk}.ss2d.in_proj.weight"]), f"{blk}.ss2d.in_proj.weight")
    _record_weight(fields, f"{out_prefix}.ssm_dw_conv2d_w", _dw2d_to_3d(sd[f"{blk}.ss2d.conv2d.weight"]), f"{blk}.ss2d.conv2d.weight")
    _record_weight(fields, f"{out_prefix}.ssm_dw_conv2d_b", sd[f"{blk}.ss2d.conv2d.bias"], f"{blk}.ss2d.conv2d.bias")
    _record_weight(fields, f"{out_prefix}.ssm_dw_conv1d_w", _conv1d_to_2d(sd[f"{blk}.ss2d.x_proj_conv.weight"]), f"{blk}.ss2d.x_proj_conv.weight")
    _record_weight(fields, f"{out_prefix}.ssm_dt_projection_w", _conv1d_to_2d(sd[f"{blk}.ss2d.dt_proj_conv.weight"]), f"{blk}.ss2d.dt_proj_conv.weight")
    _record_weight(fields, f"{out_prefix}.ssm_A", -torch.exp(sd[f"{blk}.ss2d.A_logs"].float()), f"{blk}.ss2d.A_logs", "A = -exp(A_logs)")
    _record_weight(fields, f"{out_prefix}.ssm_Ds", sd[f"{blk}.ss2d.Ds"], f"{blk}.ss2d.Ds")
    _record_weight(fields, f"{out_prefix}.ssm_delta_bias", sd[f"{blk}.ss2d.dt_projs_bias"].reshape(-1), f"{blk}.ss2d.dt_projs_bias")
    _record_weight(fields, f"{out_prefix}.ssm_out_norm_w", sd[f"{blk}.ss2d.out_norm.weight"], f"{blk}.ss2d.out_norm.weight")
    _record_weight(fields, f"{out_prefix}.ssm_out_norm_b", sd[f"{blk}.ss2d.out_norm.bias"], f"{blk}.ss2d.out_norm.bias")
    _record_weight(fields, f"{out_prefix}.ssm_out_projection_w", _conv1x1_to_2d(sd[f"{blk}.ss2d.out_proj.weight"]), f"{blk}.ss2d.out_proj.weight")
    _record_weight(fields, f"{out_prefix}.mlp_norm_w", np.ones((1,), dtype=np.float32), "<identity>", "unused placeholder")
    _record_weight(fields, f"{out_prefix}.mlp_norm_b", np.zeros((1,), dtype=np.float32), "<identity>", "unused placeholder")
    _record_weight(fields, f"{out_prefix}.mlp_fc1_w", _conv1x1_to_2d(sd[f"{blk}.mlp.fc1.weight"]), f"{blk}.mlp.fc1.weight")
    _record_weight(fields, f"{out_prefix}.mlp_fc1_b", sd[f"{blk}.mlp.fc1.bias"], f"{blk}.mlp.fc1.bias")
    _record_weight(fields, f"{out_prefix}.mlp_fc2_w", _conv1x1_to_2d(sd[f"{blk}.mlp.fc2.weight"]), f"{blk}.mlp.fc2.weight")
    _record_weight(fields, f"{out_prefix}.mlp_fc2_b", sd[f"{blk}.mlp.fc2.bias"], f"{blk}.mlp.fc2.bias")


def _extract_skip(sd: Dict[str, torch.Tensor], pt_prefix: str, out_prefix: str, fields: List[FieldRecord]) -> None:
    stream = "mag" if "mag" in pt_prefix else "phase"
    _record_weight(fields, f"{out_prefix}.skip_conv_dec0_w", _conv1x1_to_2d(sd[f"{pt_prefix}.1.skip_handler.weight"]), f"{pt_prefix}.1.skip_handler.weight")
    _record_weight(fields, f"{out_prefix}.skip_conv_dec0_b", sd[f"{pt_prefix}.1.skip_handler.bias"], f"{pt_prefix}.1.skip_handler.bias")
    _record_weight(fields, f"{out_prefix}.skip_conv_dec1_w", _conv1x1_to_2d(sd[f"{pt_prefix}.2.skip_handler.weight"]), f"{pt_prefix}.2.skip_handler.weight")
    _record_weight(fields, f"{out_prefix}.skip_conv_dec1_b", sd[f"{pt_prefix}.2.skip_handler.bias"], f"{pt_prefix}.2.skip_handler.bias")
    _record_weight(fields, f"{out_prefix}.skip_conv_dec2_w", _conv1x1_to_2d(sd[f"{pt_prefix}.3.skip_handler.weight"]), f"{pt_prefix}.3.skip_handler.weight")
    _record_weight(fields, f"{out_prefix}.skip_conv_dec2_b", sd[f"{pt_prefix}.3.skip_handler.bias"], f"{pt_prefix}.3.skip_handler.bias")
    _record_weight(fields, f"{out_prefix}.skip_conv_out0_w", _conv1x1_to_2d(sd[f"output_layer_{stream}.0.skip_handler.weight"]), f"output_layer_{stream}.0.skip_handler.weight")
    _record_weight(fields, f"{out_prefix}.skip_conv_out0_b", sd[f"output_layer_{stream}.0.skip_handler.bias"], f"output_layer_{stream}.0.skip_handler.bias")


def _build_fields(state_dict: Dict[str, torch.Tensor]) -> List[FieldRecord]:
    fields: List[FieldRecord] = []
    _extract_patch(state_dict, "patch_embed_mag", "embed_weight_mag", fields)
    _extract_patch(state_dict, "patch_embed_phase", "embed_weight_pha", fields)

    _extract_pvss_ds(state_dict, "layers_encoder_mag.0.blocks.0", "pvss_ds_weight_enc0_mag", fields)
    _extract_pvss_ds(state_dict, "layers_encoder_phase.0.blocks.0", "pvss_ds_weight_enc0_pha", fields)
    _extract_pvss_ds(state_dict, "layers_encoder_mag.1.blocks.0", "pvss_ds_weight_enc1_mag", fields)
    _extract_pvss_ds(state_dict, "layers_encoder_phase.1.blocks.0", "pvss_ds_weight_enc1_pha", fields)
    _extract_pvss_ds(state_dict, "layers_encoder_mag.2.blocks.0", "pvss_ds_weight_enc2_mag", fields)
    _extract_pvss_ds(state_dict, "layers_encoder_phase.2.blocks.0", "pvss_ds_weight_enc2_pha", fields)

    _extract_pvss_latent(state_dict, "layers_encoder_mag.3.blocks.0", "pvss_latent_weight_mag", fields)
    _extract_pvss_latent(state_dict, "layers_encoder_phase.3.blocks.0", "pvss_latent_weight_pha", fields)

    _extract_skip(state_dict, "layers_decoder_mag", "pvss_us_skip_weight_mag", fields)
    _extract_skip(state_dict, "layers_decoder_phase", "pvss_us_skip_weight_pha", fields)

    _extract_pvss_us(state_dict, "layers_decoder_mag.1", "pvss_us_weight_dec0_mag", fields)
    _extract_pvss_us(state_dict, "layers_decoder_phase.1", "pvss_us_weight_dec0_pha", fields)
    _extract_pvss_us(state_dict, "layers_decoder_mag.2", "pvss_us_weight_dec1_mag", fields)
    _extract_pvss_us(state_dict, "layers_decoder_phase.2", "pvss_us_weight_dec1_pha", fields)
    _extract_pvss_us(state_dict, "layers_decoder_mag.3", "pvss_us_weight_dec2_mag", fields)
    _extract_pvss_us(state_dict, "layers_decoder_phase.3", "pvss_us_weight_dec2_pha", fields)
    _extract_pvss_us(state_dict, "output_layer_mag.0", "pvss_us_weight_out0_mag", fields)
    _extract_pvss_us(state_dict, "output_layer_phase.0", "pvss_us_weight_out0_pha", fields)
    _extract_pvss_us(state_dict, "output_layer_mag.1", "pvss_us_weight_out1_mag", fields)
    _extract_pvss_us(state_dict, "output_layer_phase.1", "pvss_us_weight_out1_pha", fields)

    _extract_vss_output(state_dict, "mag", "vss_output3_weight_mag", fields)
    _extract_vss_output(state_dict, "phase", "vss_output3_weight_pha", fields)
    return fields


def _write_stage_exports(fields: List[FieldRecord], output_root: Path) -> None:
    manifest_rows = []
    stages: Dict[str, List[dict]] = {}

    for field_name, arr, source_key, note in fields:
        stage_name, member_name = field_name.split(".", 1)
        stage_dir = output_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{member_name}.bin"
        relpath = f"{stage_name}/{filename}"

        arr.tofile(stage_dir / filename)
        row = {
            "stage_name": stage_name,
            "member_name": member_name,
            "filename": filename,
            "relative_path": relpath,
            "dtype": "float32",
            "numel": int(arr.size),
            "shape": "x".join(str(dim) for dim in arr.shape),
            "source_key": source_key,
            "note": note,
        }
        stages.setdefault(stage_name, []).append(row)
        manifest_rows.append(row)

    for stage_name, rows in stages.items():
        meta_path = output_root / stage_name / "meta.csv"
        with meta_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["member_name", "filename", "dtype", "numel", "shape", "source_key", "note"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "member_name": row["member_name"],
                        "filename": row["filename"],
                        "dtype": row["dtype"],
                        "numel": row["numel"],
                        "shape": row["shape"],
                        "source_key": row["source_key"],
                        "note": row["note"],
                    }
                )

    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage_name", "member_name", "filename", "relative_path", "dtype", "numel", "shape", "source_key", "note"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "output_root": str(output_root),
        "num_stages": len(stages),
        "num_fields": len(manifest_rows),
        "stages": sorted(stages.keys()),
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    _install_checkpoint_stubs()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = _strip_module(checkpoint["state_dict"])
    fields = _build_fields(state_dict)

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    _write_stage_exports(fields, OUTPUT_ROOT)
    print(f"Exported {len(fields)} float32 tensors into {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
