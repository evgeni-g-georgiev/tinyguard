import numpy as np
import librosa

from config import LOG_OFFSET

def make_log_mel_spectrogram(waveform, chunk_seconds, sampling_frequency, n_mels, n_fft, hop_length, power=2.0, center=True):
    """Convert a waveform chunk into a log-mel spectrogram.

    Args:
        waveform: 1D array containing one audio chunk.
        chunk_seconds: Length of the chunk in seconds.
        sampling_frequency: Sampling frequency in Hz.
        n_mels: Number of mel-frequency bins.
        n_fft: FFT window size.
        hop_length: Hop length in samples.
        power: Exponent for the magnitude spectrogram.

    Returns:
        A float32 log-mel spectrogram.
    """
    # Step 1: Check that the given sound chunk has the expected length
    expected_length = int(round(chunk_seconds * sampling_frequency))
    assert len(waveform) == expected_length

    # Step 2: Create a mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sampling_frequency,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
        center=center, # Padding added at the start and at the end. 
    )

    # Step 3: Convert the spectrogram into a log-mel spectrogram
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Step 4: Check output dimension
    expected_height = n_mels
    if center==True:
        pad = n_fft // 2
    else: 
        pad = 0
    expected_width = (expected_length + 2 * pad - n_fft) // hop_length + 1
    assert log_mel_spectrogram.shape == (expected_height, expected_width)

    # Step 5: Add a channel dimension
    log_mel_spectrogram = log_mel_spectrogram[np.newaxis, :, :]

    return log_mel_spectrogram.astype("float32")


def make_gmm_log_mel_spectrogram(
    waveform,
    chunk_seconds,
    sampling_frequency,
    n_mels,
    n_fft,
    hop_length,
    power=2.0,
    center=True,
):
    """Convert a waveform chunk into a GMM-style log-mel spectrogram.

    Args:
        waveform: 1D array containing one audio chunk or full clip.
        chunk_seconds: Length of the waveform in seconds.
        sampling_frequency: Sampling frequency in Hz.
        n_mels: Number of mel-frequency bins.
        n_fft: FFT window size.
        hop_length: Hop length in samples.
        power: Exponent for the magnitude spectrogram.

    Returns:
        A float32 log-mel spectrogram.
    """
    # Step 1: Check that the given waveform has the expected length
    expected_length = int(round(chunk_seconds * sampling_frequency))
    assert len(waveform) == expected_length

    # Step 2: Create a mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sampling_frequency,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
        center=center,
    )

    # Step 3: Convert to the GMM pipeline's log-mel definition
    log_mel_spectrogram = np.log(mel_spectrogram + LOG_OFFSET)

    # Step 4: Check output dimension
    expected_height = n_mels
    if center == True:
        pad = n_fft // 2
    else:
        pad = 0
    expected_width = (expected_length + 2 * pad - n_fft) // hop_length + 1
    assert log_mel_spectrogram.shape == (expected_height, expected_width)

    # Step 5: Add a channel dimension
    log_mel_spectrogram = log_mel_spectrogram[np.newaxis, :, :]

    return log_mel_spectrogram.astype("float32")
