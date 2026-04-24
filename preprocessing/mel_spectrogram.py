"""Log-mel spectrogram builder used by the GMM pipeline.

The streaming construction here (rolling overlap buffer, no centre padding,
``log(energy + LOG_OFFSET)``) matches the on-device implementation frame for
frame so host-computed features are numerically identical to those produced
on the Arduino.
"""
import numpy as np
import librosa

from config import LOG_OFFSET


def make_gmm_log_mel_spectrogram(
    waveform,
    chunk_seconds,
    sampling_frequency,
    n_mels,
    n_fft,
    hop_length,
    power=2.0,
):
    """Return the ``(1, n_mels, T)`` float32 log-mel spectrogram for one clip.

    ``T = len(waveform) // hop_length`` (312 for 10-second clips at 16 kHz).
    The leading axis is a channel dimension kept for compatibility with
    callers that expect it.
    """
    expected_length = int(round(chunk_seconds * sampling_frequency))
    assert len(waveform) == expected_length

    mel_fb = librosa.filters.mel(sr=sampling_frequency, n_fft=n_fft, n_mels=n_mels)
    window = np.hanning(n_fft).astype(np.float32)

    n_frames  = len(waveform) // hop_length
    audio_buf = np.zeros(n_fft, dtype=np.float32)
    frames    = np.empty((n_mels, n_frames), dtype=np.float32)

    for i in range(n_frames):
        audio_buf[:n_fft - hop_length] = audio_buf[hop_length:]
        audio_buf[n_fft - hop_length:] = waveform[i * hop_length : (i + 1) * hop_length]

        spectrum   = np.fft.rfft(audio_buf * window)
        power_spec = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
        if power != 2.0:
            power_spec = power_spec ** (power / 2.0)
        frames[:, i] = np.log(mel_fb @ power_spec + LOG_OFFSET)

    return frames[np.newaxis].astype(np.float32)
