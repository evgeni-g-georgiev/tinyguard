"""WAV file → log-mel spectrogram for the GMM pipeline."""
import numpy as np

from config import SAMPLE_RATE, N_FFT, GMM_HOP_LENGTH, GMM_N_MELS, MIMII_CLIP_SECS
from preprocessing.loader import load_audio
from preprocessing.mel_spectrogram import make_gmm_log_mel_spectrogram


def load_full_clip_log_mel(wav_path, n_mels=GMM_N_MELS, channel=None):
    """Return the ``(n_mels, T)`` log-mel spectrogram for one clip.

    ``channel`` picks a single mic channel (0-7 for MIMII); ``None`` mixes to mono.
    """
    audio, _ = load_audio(wav_path, sampling_frequency=SAMPLE_RATE, mono=True, channel=channel)
    log_mel = make_gmm_log_mel_spectrogram(
        waveform=audio.astype(np.float32),
        chunk_seconds=MIMII_CLIP_SECS,
        sampling_frequency=SAMPLE_RATE,
        n_mels=n_mels,
        n_fft=N_FFT,
        hop_length=GMM_HOP_LENGTH,
    )
    return log_mel[0]
