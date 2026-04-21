import numpy as np

from config import SAMPLE_RATE, N_FFT, GMM_HOP_LENGTH, GMM_N_MELS, MIMII_CLIP_SECS
from preprocessing.loader import load_audio
from preprocessing.mel_spectrogram import make_gmm_log_mel_spectrogram


def load_full_clip_log_mel(wav_path, n_mels=GMM_N_MELS, channel=None):
    """Load one clip and return the full-clip log-mel spectrogram for GMM.

    Parameters
    ----------
    wav_path : str
    n_mels : int
        Number of mel frequency bins.  Defaults to GMM_N_MELS (128).
        Pass 64 to match the reduced-resolution Chip B deployment path.
    channel : int or None
        Microphone channel index (0-7 for MIMII).  None mixes all channels
        to mono (legacy behaviour).
    """

    audio, _ = load_audio(wav_path, sampling_frequency=SAMPLE_RATE, mono=True, channel=channel)

    log_mel = make_gmm_log_mel_spectrogram(
        waveform=audio.astype(np.float32),
        chunk_seconds=MIMII_CLIP_SECS,
        sampling_frequency=SAMPLE_RATE,
        n_mels=n_mels,
        n_fft=N_FFT,
        hop_length=GMM_HOP_LENGTH,
        power=2.0,
        center=True,
    )

    return log_mel[0]  # drop the channel dimension added by make_gmm_log_mel_spectrogram