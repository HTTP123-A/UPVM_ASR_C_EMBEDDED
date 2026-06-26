import torch

try:
    from pystoi import stoi as stoi_score
except Exception:
    stoi_score = None

DEFAULT_STFT_N_FFT = 2048
DEFAULT_STFT_HOP_LENGTH = 512
LOG_EPS = 1e-8


def stft(audio, n_fft=DEFAULT_STFT_N_FFT, hop_length=DEFAULT_STFT_HOP_LENGTH):
    hann_window = torch.hann_window(n_fft).to(audio.device)
    stft_spec = torch.stft(
        audio, n_fft, hop_length, window=hann_window, return_complex=True
    )
    spec = torch.sqrt(stft_spec.real.pow(2) + stft_spec.imag.pow(2))
    return spec


def _resolve_stft_args(kwargs):
    n_fft = int(kwargs.get("n_fft", DEFAULT_STFT_N_FFT))
    hop_length = int(kwargs.get("hop_length", DEFAULT_STFT_HOP_LENGTH))
    return n_fft, hop_length


def snr(output, target, **kwargs):
    val = (
        20
        * torch.log10(
            torch.norm(target, dim=-1)
            / torch.norm(output - target, dim=-1).clamp(min=1e-8)
        )
    ).mean()
    return val.item()


def lsd(output, target, **kwargs):
    n_fft, hop_length = _resolve_stft_args(kwargs)
    sp = torch.log10(stft(output, n_fft=n_fft, hop_length=hop_length).square().clamp(LOG_EPS))
    st = torch.log10(stft(target, n_fft=n_fft, hop_length=hop_length).square().clamp(LOG_EPS))
    return (sp - st).square().mean(dim=1).sqrt().mean().item()


def lsd_hf(output, target, hf, **kwargs):
    n_fft, hop_length = _resolve_stft_args(kwargs)
    sp = torch.log10(stft(output, n_fft=n_fft, hop_length=hop_length).square().clamp(LOG_EPS))
    st = torch.log10(stft(target, n_fft=n_fft, hop_length=hop_length).square().clamp(LOG_EPS))
    val = []
    for i in range(output.size(0)):
        hf_i = hf[i].item()
        val.append(
            (
                (sp[i, hf_i:, :] - st[i, hf_i:, :])
                .square()
                .mean(dim=0)
                .sqrt()
                .mean()
                .item()
            )
        )
    return torch.tensor(val).mean().item()


def lsd_lf(output, target, hf, **kwargs):
    n_fft, hop_length = _resolve_stft_args(kwargs)
    sp = torch.log10(stft(output, n_fft=n_fft, hop_length=hop_length).square().clamp(LOG_EPS))
    st = torch.log10(stft(target, n_fft=n_fft, hop_length=hop_length).square().clamp(LOG_EPS))
    val = []
    for i in range(output.size(0)):
        hf_i = hf[i].item()
        val.append(
            (
                (sp[i, :hf_i, :] - st[i, :hf_i, :])
                .square()
                .mean(dim=0)
                .sqrt()
                .mean()
                .item()
            )
        )
    return torch.tensor(val).mean().item()


def stoi(output, target, **kwargs):
    if stoi_score is None:
        raise RuntimeError("pystoi is unavailable in this environment.")

    sr = kwargs.get("sr", 16000)
    scores = []
    for i in range(output.size(0)):
        clean = target[i].detach().cpu().numpy().astype(float).flatten()
        deg = output[i].detach().cpu().numpy().astype(float).flatten()
        s = stoi_score(clean, deg, sr, extended=False)
        scores.append(s)
    return float(sum(scores) / len(scores))
