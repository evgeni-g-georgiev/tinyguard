import numpy as np

from config import SAMPLE_RATE, N_FFT, GMM_HOP_LENGTH, GMM_N_MELS, MIMII_CLIP_SECS
from preprocessing.loader import load_audio
from preprocessing.mel_spectrogram import make_log_mel_spectrogram


def load_full_clip_log_mel(wav_path):
    """Load one clip and return the full-clip log-mel spectrogram for GMM."""

    audio, _ = load_audio(wav_path, sampling_frequency=SAMPLE_RATE, mono=True)

    log_mel =  make_log_mel_spectrogram(
        waveform=audio.astype(np.float32),
        chunk_seconds=MIMII_CLIP_SECS, # MIMII data by definition 10 seconds
        sampling_frequency=SAMPLE_RATE,
        n_mels=GMM_N_MELS,
        n_fft=N_FFT,
        hop_length=GMM_HOP_LENGTH,
        power=2.0, 
        center=True,
    )

    return log_mel[0] # we do not want the added channel dimension