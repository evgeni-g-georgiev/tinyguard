import numpy as np

from config import SAMPLE_RATE, chunk_seconds, N_MELS, N_FFT, HOP_LENGTH
from preprocessing.loader import load_audio, split_into_chunks
from preprocessing.mel_spectrogram import make_log_mel_spectrogram


def load_clip_log_mels(wav_path):
    """Load one clip, split it into fixed frames, and return stacked log-mels.

    Args:
        wav_path: Path to a WAV clip.

    Returns:
        A float32 array of shape (n_frames, 1, 64, 61), or None if the clip is
        shorter than one frame.
    """
    audio, _ = load_audio(wav_path, sampling_frequency=SAMPLE_RATE, mono=True)

    chunks = split_into_chunks(audio, sampling_frequency=SAMPLE_RATE, chunk_seconds=chunk_seconds)

    if len(chunks) == 0:
        return None

    mels = []

    for chunk in chunks:
        chunk = chunk.astype(np.float32)

        mel = make_log_mel_spectrogram(
            waveform=chunk,
            chunk_seconds=chunk_seconds,
            sampling_frequency=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )

        mels.append(mel)

    stacked_mels = np.stack(mels)
    return stacked_mels
