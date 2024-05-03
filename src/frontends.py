from typing import Callable, List, Tuple, Union

import torch
import torchaudio

# values from FakeAVCeleb paper
SAMPLING_RATE = 16_000
win_length = 400  # int((25 / 1_000) * SAMPLING_RATE)
hop_length = 160  # int((10 / 1_000) * SAMPLING_RATE)

device = "cuda" if torch.cuda.is_available() else "cpu"

MFCC_FN = torchaudio.transforms.MFCC(
    sample_rate=SAMPLING_RATE,
    n_mfcc=80,
    melkwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
    },
).to(device)


LFCC_FN = torchaudio.transforms.LFCC(
    sample_rate=SAMPLING_RATE,
    n_lfcc=80,
    speckwargs={
        "n_fft": 512,
        "win_length": win_length,
        "hop_length": hop_length,
    },
).to(device)

MEL_SCALE_FN = torchaudio.transforms.MelScale(
    n_mels=80,
    n_stft=257,
    sample_rate=SAMPLING_RATE,
).to(device)


def get_frontend(
    frontends: List[str],
) -> Union[torchaudio.transforms.MFCC, torchaudio.transforms.LFCC, Callable,]:
    if "mfcc" in frontends:
        return MFCC_FN
    elif "lfcc" in frontends:
        return LFCC_FN
    elif "mel_spec" in frontends:
        return prepare_mel_scale_vector
    raise ValueError(f"{frontends} frontend is not supported!")


def prepare_mel_scale_vector(
    audio: torch.Tensor, win_length=400, hop_length=160
) -> torch.Tensor:
    stft_abs_mel, stft_abs_angle = prepare_stft_features(audio, win_length, hop_length)
    return torch.stack([stft_abs_mel, stft_abs_angle], dim=1)


def prepare_stft_features(
    audio: torch.Tensor, win_length: int = 400, hop_length: int = 160
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Run STFT
    stft_out = torch.stft(
        audio,
        n_fft=512,
        return_complex=True,
        hop_length=hop_length,
        win_length=win_length,
    )

    # Reduce dimensionality via use of mel filterbanks
    stft_real_mel = MEL_SCALE_FN(stft_out.real)
    stft_imag_mel = MEL_SCALE_FN(stft_out.imag)

    complex_tensor = torch.complex(stft_real_mel, stft_imag_mel)
    stft_abs_mel = complex_tensor.abs()
    stft_abs_angle = complex_tensor.angle()
    return stft_abs_mel, stft_abs_angle
