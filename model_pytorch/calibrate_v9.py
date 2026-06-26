#!/usr/bin/env python3
"""v9 calibration: measure the PatchEmbed norm1 OUTPUT range -> S_ln (per stream), for the int8
output of the v9 integer LayerNorm. norm1 is conv1 -> Permute -> LayerNorm, i.e. the very first
op block (upstream of every Mamba/VSS layer), so we only reconstruct conv1+norm1 from the float
checkpoint and feed REAL spectrograms -- no model class / mamba_ssm needed.

Pipeline replicated exactly from the Revision MambaUNet:
  wav -> wav2spectro(n_fft=512,hop=64,win=256, log2 mag) -> drop first freq bin -> per-sample
  normalize (x-mean)/(1e-5+std) over (C,F,T) -> conv1(k3,s2,p1) -> permute(0,2,3,1) -> LayerNorm.

Output: model_pytorch/calib_v9_s_ln.json  {"mag": S_ln_mag, "pha": S_ln_pha, ...meta}
Run from the C-port model_pytorch/ in a torch env (e.g. conda test_amb).
"""
import json, sys, glob, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REVISION = Path("/F/AI_Train/M11309813/Personal_AI_Model_Training/myenv/Revision/UPVM-ASR-0")
CKPT = HERE / "model" / "checkpoint-best-G.pth"
DOWNSAMPLE = HERE.parent / "data" / "test_audio" / "downsample"
OUT_JSON = HERE / "calib_v9_s_ln.json"

N_WAVS   = 300         # calibration subset (deterministic: first N sorted)
PCTL     = 99.99       # robust max (avoid single-outlier scale blow-up)
N_FFT, HOP, WIN = 512, 64, 256

sys.path.insert(0, str(REVISION))
from utils.stft import wav2spectro          # exact STFT used by the model

def load_wav(path):
    import torchaudio
    w, sr = torchaudio.load(path)            # [channel, length]
    if w.dim() == 2:
        w = w.mean(0, keepdim=True)          # mono
    return w.unsqueeze(0)                    # [1,1,length]

def build_conv_norm(sd, prefix, c_out):
    conv = nn.Conv2d(1, c_out, kernel_size=3, stride=2, padding=1, bias=True)
    conv.weight.data = sd[f"{prefix}.0.weight"].float()
    conv.bias.data   = sd[f"{prefix}.0.bias"].float()
    norm = nn.LayerNorm(c_out)
    norm.weight.data = sd[f"{prefix}.2.weight"].float()   # index .2 = norm1 (after conv .0, permute .1)
    norm.bias.data   = sd[f"{prefix}.2.bias"].float()
    return conv.eval(), norm.eval()

@torch.no_grad()
def norm1_out(conv, norm, spectro):
    # spectro: [1,1,F,T] (mag or phase), drop first freq bin, per-sample normalize, conv, permute, LN
    x = spectro[..., 1:, :]
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std  = x.std(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) / (1e-5 + std)
    x = conv(x)                       # [1,8,H,W]
    x = x.permute(0, 2, 3, 1)         # [1,H,W,8]
    return norm(x)                    # norm1 output

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "generator" in sd:   sd = sd["generator"]
    if isinstance(sd, dict) and "state_dict" in sd:  sd = sd["state_dict"]

    nets = {
        "mag": (build_conv_norm(sd, "patch_embed_mag",   8), "mag"),
        "pha": (build_conv_norm(sd, "patch_embed_phase", 8), "phase"),
    }
    for k in nets:
        (c, n), _ = nets[k]; c.to(dev); n.to(dev)

    wavs = sorted(glob.glob(str(DOWNSAMPLE / "*.wav")))[:N_WAVS]
    assert wavs, f"no wavs in {DOWNSAMPLE}"
    print(f"calibrating S_ln on {len(wavs)} wavs (dev={dev}) ...")

    absmax = {"mag": [], "pha": []}      # collect abs values (downsampled) for percentile
    for i, wpath in enumerate(wavs):
        w = load_wav(wpath).to(dev)
        mag, phase = wav2spectro(w, N_FFT, HOP, WIN, "log2")   # [1,1,257,T] each
        for key, spec in (("mag", mag), ("pha", phase)):
            (conv, norm), _ = nets[key]
            o = norm1_out(conv, norm, spec).abs().reshape(-1)
            # subsample to keep memory bounded yet representative
            if o.numel() > 50000:
                idx = torch.randint(0, o.numel(), (50000,), device=o.device)
                o = o[idx]
            absmax[key].append(o.cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(wavs)}")

    result = {"_meta": {"n_wavs": len(wavs), "percentile": PCTL, "n_fft": N_FFT,
                        "hop": HOP, "win": WIN, "note": "S_ln = pctl(|norm1_out|)/127, per stream"}}
    for key in ("mag", "pha"):
        a = np.concatenate(absmax[key])
        rng = float(np.percentile(a, PCTL))
        result[key] = rng / 127.0
        result[f"_{key}_range_pctl"] = rng
        result[f"_{key}_rawmax"] = float(a.max())
        print(f"  {key}: range(p{PCTL})={rng:.5f}  rawmax={a.max():.5f}  S_ln={rng/127.0:.6e}")

    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT_JSON}")

if __name__ == "__main__":
    main()
