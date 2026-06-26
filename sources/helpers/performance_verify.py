from __future__ import annotations

import csv
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.io import wavfile

# ========================= PATH CONFIGURATION =========================
EMBEDDED_ROOT = Path(__file__).resolve().parents[2]

# VERSION="naives_mp_8"
# The above code is defining a Python variable `VERSION` with the value "naives_mp_8".
VERSION="sq_int8_v9"
# Folder containing generated WAVs to evaluate.
GENERATED_AUDIO_DIR = EMBEDDED_ROOT / f"data/test_audio/generated/{VERSION}"
# Folder containing clean/original WAVs.
ORIGINAL_AUDIO_DIR = EMBEDDED_ROOT / "data/test_audio/original"
# Folder used to define protocol file list (matches tester set).
DEGRADED_AUDIO_DIR = EMBEDDED_ROOT / "data/test_audio/downsample"

OUTPUT_REPORT_PATH = EMBEDDED_ROOT / f"results/{VERSION}/performance_verify.txt"
OUTPUT_PER_FILE_CSV = EMBEDDED_ROOT / f"results/{VERSION}/performance_verify_per_file.csv"

# Always use embedded metric module so this script is self-contained on C side.
METRIC_PY_PATH = EMBEDDED_ROOT / "model_pytorch/metric.py"

# ========================= PROTOCOL CONFIGURATION =========================
TAG = "16000_48000"
INPUT_SR = int(TAG.split("_")[0])
TARGET_SR = int(TAG.split("_")[1])

# Highcut rule must follow data loader (config.DATA.STFT.N_FFT), not metric STFT n_fft.
DATA_STFT_N_FFT = 1024
HIGHCUT_IN_STFT = int((1 + DATA_STFT_N_FFT // 2) * (INPUT_SR / TARGET_SR))

# Metric STFT settings (match model/metric.py defaults in full PyTorch pipeline).
METRIC_STFT_N_FFT = 2048
METRIC_STFT_HOP = 512

# Pairing mode:
# - "direct_name": generated filename equals original filename (e.g. p360_001.wav).
# - "tester_suffix": generated uses *_up.wav and reference uses *_orig.wav if present,
#   otherwise falls back to ORIGINAL_AUDIO_DIR/<stem>.wav.
PAIRING_MODE = "direct_name"

# Evaluation protocol:
# - "tester_eval": pad to EVAL_SEGMENT_LENGTH before metric (match tester style).
# - "trimmed": align/trim only.
# - "both": compute both.
EVAL_PROTOCOL = "tester_eval"
EVAL_SEGMENT_LENGTH = 122640
EVAL_PAD_MODE = "zero"

# Optional debug cap (set to None for full dataset).
MAX_FILES: int | None = None


def _import_metrics(metric_path: Path):
    if not metric_path.is_file():
        raise FileNotFoundError(f"Metric file not found: {metric_path}")

    spec = importlib.util.spec_from_file_location("metric_module", str(metric_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load metric module spec from {metric_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.snr, module.lsd, module.lsd_hf, module.lsd_lf


def _to_float32_audio(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8:
        return (data.astype(np.float32) - 128.0) / 128.0
    if data.dtype == np.float32:
        return data
    if data.dtype == np.float64:
        return data.astype(np.float32)
    data_f32 = data.astype(np.float32)
    max_abs = float(np.max(np.abs(data_f32))) if data_f32.size > 0 else 0.0
    if max_abs > 0.0:
        data_f32 = data_f32 / max_abs
    return data_f32


def _load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(str(path))
    if data.ndim == 2:
        data = data.mean(axis=1)
    audio = _to_float32_audio(data)
    return int(sample_rate), audio


def _align_like_official(pred_wave: np.ndarray, target_wave: np.ndarray) -> np.ndarray:
    if pred_wave.shape[0] < target_wave.shape[0]:
        padded = np.zeros((target_wave.shape[0],), dtype=np.float32)
        padded[: pred_wave.shape[0]] = pred_wave
        return padded
    if pred_wave.shape[0] > target_wave.shape[0]:
        return pred_wave[: target_wave.shape[0]]
    return pred_wave


def _pad_eval_length(audio: np.ndarray, segment_length: int, pad_mode: str) -> np.ndarray:
    if segment_length <= 0:
        raise ValueError(f"EVAL_SEGMENT_LENGTH must be > 0, got {segment_length}")

    rem = audio.shape[0] % segment_length
    if rem == 0:
        return audio

    pad_len = segment_length - rem
    if pad_mode == "zero":
        return np.pad(audio, (0, pad_len), mode="constant", constant_values=0.0)

    raise ValueError(f"Unsupported EVAL_PAD_MODE='{pad_mode}'")


def _prepare_pair(pred: np.ndarray, target: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    pred_aligned = _align_like_official(pred, target)

    if mode == "trimmed":
        return pred_aligned, target

    if mode == "tester_eval":
        pred_eval = _pad_eval_length(pred_aligned, EVAL_SEGMENT_LENGTH, EVAL_PAD_MODE)
        target_eval = _pad_eval_length(target, EVAL_SEGMENT_LENGTH, EVAL_PAD_MODE)
        return pred_eval, target_eval

    raise ValueError(f"Unsupported mode '{mode}'")


def _compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    snr_fn: Callable,
    lsd_fn: Callable,
    lsd_hf_fn: Callable,
    lsd_lf_fn: Callable,
) -> tuple[float, float, float, float]:
    output_bt = torch.from_numpy(pred).to(torch.float32).unsqueeze(0)
    target_bt = torch.from_numpy(target).to(torch.float32).unsqueeze(0)
    hf_tensor = torch.tensor([HIGHCUT_IN_STFT], dtype=torch.int64)

    metric_kwargs = {
        "hf": hf_tensor,
        "sr": TARGET_SR,
        "n_fft": METRIC_STFT_N_FFT,
        "hop_length": METRIC_STFT_HOP,
    }
    snr_val = float(snr_fn(output_bt, target_bt, **metric_kwargs))
    lsd_val = float(lsd_fn(output_bt, target_bt, **metric_kwargs))
    lsd_hf_val = float(lsd_hf_fn(output_bt, target_bt, **metric_kwargs))
    lsd_lf_val = float(lsd_lf_fn(output_bt, target_bt, **metric_kwargs))
    return snr_val, lsd_val, lsd_hf_val, lsd_lf_val


def _eval_one_mode(
    pred: np.ndarray,
    target: np.ndarray,
    mode: str,
    snr_fn: Callable,
    lsd_fn: Callable,
    lsd_hf_fn: Callable,
    lsd_lf_fn: Callable,
) -> tuple[float, float, float, float]:
    pred_ready, tgt_ready = _prepare_pair(pred, target, mode)
    return _compute_metrics(pred_ready, tgt_ready, snr_fn, lsd_fn, lsd_hf_fn, lsd_lf_fn)


def _fmt_stat(values: list[float]) -> str:
    if not values:
        return "n=0"
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"n={arr.size}, mean={arr.mean():.6f}, std={arr.std(ddof=0):.6f}, "
        f"min={arr.min():.6f}, max={arr.max():.6f}, median={np.median(arr):.6f}"
    )


def _load_protocol_filelist() -> tuple[list[str], str]:
    degraded_files = sorted(p.name for p in DEGRADED_AUDIO_DIR.glob("*.wav"))
    if degraded_files:
        return degraded_files, f"degraded_dir={DEGRADED_AUDIO_DIR}"

    generated_files = sorted(p.name for p in GENERATED_AUDIO_DIR.glob("*.wav"))
    return generated_files, f"generated_dir={GENERATED_AUDIO_DIR}"


def _resolve_pair(filename: str) -> tuple[Path, Path]:
    if PAIRING_MODE == "direct_name":
        return GENERATED_AUDIO_DIR / filename, ORIGINAL_AUDIO_DIR / filename

    if PAIRING_MODE == "tester_suffix":
        stem = Path(filename).stem
        gen_path = GENERATED_AUDIO_DIR / f"{stem}_up.wav"
        same_dir_orig = GENERATED_AUDIO_DIR / f"{stem}_orig.wav"
        if same_dir_orig.exists():
            return gen_path, same_dir_orig
        return gen_path, ORIGINAL_AUDIO_DIR / filename

    raise ValueError(f"Unsupported PAIRING_MODE='{PAIRING_MODE}'")


def main() -> None:
    if EVAL_PROTOCOL not in {"tester_eval", "trimmed", "both"}:
        raise ValueError(f"EVAL_PROTOCOL must be one of [tester_eval, trimmed, both], got '{EVAL_PROTOCOL}'")

    if PAIRING_MODE not in {"direct_name", "tester_suffix"}:
        raise ValueError(f"PAIRING_MODE must be one of [direct_name, tester_suffix], got '{PAIRING_MODE}'")

    snr_fn, lsd_fn, lsd_hf_fn, lsd_lf_fn = _import_metrics(METRIC_PY_PATH)

    if not GENERATED_AUDIO_DIR.is_dir():
        raise FileNotFoundError(f"Generated directory not found: {GENERATED_AUDIO_DIR}")
    if not ORIGINAL_AUDIO_DIR.is_dir():
        raise FileNotFoundError(f"Original directory not found: {ORIGINAL_AUDIO_DIR}")

    protocol_filenames, protocol_source = _load_protocol_filelist()
    if MAX_FILES is not None:
        protocol_filenames = protocol_filenames[: int(MAX_FILES)]

    total_protocol_files = len(protocol_filenames)
    matched = 0
    missing_generated = 0
    missing_original = 0
    skipped_sr_mismatch = 0
    skipped_empty = 0

    stats: dict[str, list[float]] = {
        "snr_tester_eval": [], "lsd_tester_eval": [], "lsd_hf_tester_eval": [], "lsd_lf_tester_eval": [],
        "snr_trimmed": [], "lsd_trimmed": [], "lsd_hf_trimmed": [], "lsd_lf_trimmed": [],
    }
    per_file_rows: list[dict[str, float | str | int]] = []

    for filename in protocol_filenames:
        gen_path, ref_path = _resolve_pair(filename)

        if not gen_path.exists():
            missing_generated += 1
            continue
        if not ref_path.exists():
            missing_original += 1
            continue

        sr_gen, audio_gen = _load_wav_mono(gen_path)
        sr_ref, audio_ref = _load_wav_mono(ref_path)

        if sr_gen != sr_ref:
            skipped_sr_mismatch += 1
            continue
        if audio_ref.shape[0] == 0 or audio_gen.shape[0] == 0:
            skipped_empty += 1
            continue

        row: dict[str, float | str | int] = {
            "file_name": filename,
            "generated_file": gen_path.name,
            "reference_file": ref_path.name,
            "sample_rate": sr_ref,
            "highcut_in_stft": HIGHCUT_IN_STFT,
            "orig_len": int(audio_ref.shape[0]),
            "gen_len": int(audio_gen.shape[0]),
        }

        if EVAL_PROTOCOL in {"tester_eval", "both"}:
            snr, lsd, lsd_hf, lsd_lf = _eval_one_mode(
                audio_gen, audio_ref, "tester_eval", snr_fn, lsd_fn, lsd_hf_fn, lsd_lf_fn
            )
            row.update(
                {
                    "snr_tester_eval": snr,
                    "lsd_tester_eval": lsd,
                    "lsd_hf_tester_eval": lsd_hf,
                    "lsd_lf_tester_eval": lsd_lf,
                }
            )
            stats["snr_tester_eval"].append(snr)
            stats["lsd_tester_eval"].append(lsd)
            stats["lsd_hf_tester_eval"].append(lsd_hf)
            stats["lsd_lf_tester_eval"].append(lsd_lf)

        if EVAL_PROTOCOL in {"trimmed", "both"}:
            snr, lsd, lsd_hf, lsd_lf = _eval_one_mode(
                audio_gen, audio_ref, "trimmed", snr_fn, lsd_fn, lsd_hf_fn, lsd_lf_fn
            )
            row.update(
                {
                    "snr_trimmed": snr,
                    "lsd_trimmed": lsd,
                    "lsd_hf_trimmed": lsd_hf,
                    "lsd_lf_trimmed": lsd_lf,
                }
            )
            stats["snr_trimmed"].append(snr)
            stats["lsd_trimmed"].append(lsd)
            stats["lsd_hf_trimmed"].append(lsd_hf)
            stats["lsd_lf_trimmed"].append(lsd_lf)

        matched += 1
        per_file_rows.append(row)

    lines: list[str] = []
    lines.append("UPVM-ASR Generated Audio Performance Verification")
    lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"Tag                 : {TAG}")
    lines.append(f"Input SR / Target   : {INPUT_SR} / {TARGET_SR}")
    lines.append(f"Data STFT n_fft     : {DATA_STFT_N_FFT}")
    lines.append(f"Metric STFT n_fft   : {METRIC_STFT_N_FFT}")
    lines.append(f"Metric STFT hop     : {METRIC_STFT_HOP}")
    lines.append(f"Highcut in STFT     : {HIGHCUT_IN_STFT}")
    lines.append(f"Metric module       : {METRIC_PY_PATH}")
    lines.append(f"Pairing mode        : {PAIRING_MODE}")
    lines.append(f"Eval protocol       : {EVAL_PROTOCOL}")
    lines.append(f"Eval segment length : {EVAL_SEGMENT_LENGTH}")
    lines.append(f"Eval pad mode       : {EVAL_PAD_MODE}")
    lines.append("")
    lines.append(f"Protocol source     : {protocol_source}")
    lines.append(f"Original directory  : {ORIGINAL_AUDIO_DIR}")
    lines.append(f"Generated directory : {GENERATED_AUDIO_DIR}")
    lines.append("")
    lines.append("Dataset summary")
    lines.append(f"- protocol file count: {total_protocol_files}")
    lines.append(f"- matched pairs: {matched}")
    lines.append(f"- missing generated: {missing_generated}")
    lines.append(f"- missing reference: {missing_original}")
    lines.append(f"- skipped (sample-rate mismatch): {skipped_sr_mismatch}")
    lines.append(f"- skipped (empty audio): {skipped_empty}")
    lines.append("")

    def add_agg_block(mode_key: str) -> None:
        lines.append(f"Aggregate metrics (generated vs reference) [{mode_key}]")
        lines.append(f"- SNR (dB): {_fmt_stat(stats[f'snr_{mode_key}'])}")
        lines.append(f"- LSD      : {_fmt_stat(stats[f'lsd_{mode_key}'])}")
        lines.append(f"- LSD_HF   : {_fmt_stat(stats[f'lsd_hf_{mode_key}'])}")
        lines.append(f"- LSD_LF   : {_fmt_stat(stats[f'lsd_lf_{mode_key}'])}")
        lines.append("")

    if EVAL_PROTOCOL in {"tester_eval", "both"}:
        add_agg_block("tester_eval")
    if EVAL_PROTOCOL in {"trimmed", "both"}:
        add_agg_block("trimmed")

    if per_file_rows:
        if EVAL_PROTOCOL in {"tester_eval", "both"}:
            worst = sorted(
                [r for r in per_file_rows if "lsd_tester_eval" in r],
                key=lambda x: float(x["lsd_tester_eval"]),
                reverse=True,
            )
            lines.append("Worst 10 by LSD (tester_eval)")
            for row in worst[:10]:
                lines.append(
                    f"- {row['file_name']}: "
                    f"LSD={float(row['lsd_tester_eval']):.6f}, "
                    f"LSD_HF={float(row['lsd_hf_tester_eval']):.6f}, "
                    f"LSD_LF={float(row['lsd_lf_tester_eval']):.6f}, "
                    f"SNR={float(row['snr_tester_eval']):.6f}"
                )
            lines.append("")

        if EVAL_PROTOCOL in {"trimmed", "both"}:
            worst = sorted(
                [r for r in per_file_rows if "lsd_trimmed" in r],
                key=lambda x: float(x["lsd_trimmed"]),
                reverse=True,
            )
            lines.append("Worst 10 by LSD (trimmed)")
            for row in worst[:10]:
                lines.append(
                    f"- {row['file_name']}: "
                    f"LSD={float(row['lsd_trimmed']):.6f}, "
                    f"LSD_HF={float(row['lsd_hf_trimmed']):.6f}, "
                    f"LSD_LF={float(row['lsd_lf_trimmed']):.6f}, "
                    f"SNR={float(row['snr_trimmed']):.6f}"
                )
            lines.append("")

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    fieldnames: set[str] = set()
    for row in per_file_rows:
        fieldnames.update(row.keys())
    ordered = ["file_name", "generated_file", "reference_file", "sample_rate", "orig_len", "gen_len", "highcut_in_stft"]
    other = sorted(k for k in fieldnames if k not in ordered)

    with OUTPUT_PER_FILE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered + other)
        writer.writeheader()
        for row in per_file_rows:
            writer.writerow(row)

    print(f"Wrote verification report: {OUTPUT_REPORT_PATH}")
    print(f"Wrote per-file csv       : {OUTPUT_PER_FILE_CSV}")


if __name__ == "__main__":
    main()
