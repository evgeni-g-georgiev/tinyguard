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
    center=True,   # kept for API compatibility; ignored — C++ uses no centre-pad
):
    """Convert a waveform into a GMM-style log-mel spectrogram.

    Replicates deployment/spectrogram.h step-for-step:
      - Zero-initialised overlap buffer of length n_fft (matches audio_buf[N_FFT]={0})
      - Hann window via np.hanning (symmetric form, identical to ArduinoFFT v2)
      - Power spectrum: real² + imag², cast to float32 (matches ArduinoFFT float32)
      - Mel filterbank: librosa.filters.mel (same coefficients as mel_filterbank.h)
      - log(mel_energy + LOG_OFFSET) stored per frame → shape (n_mels, T)
      - T = len(waveform) // hop_length = 312 for 10-second clips at 16 kHz

    Returns:
        float32 array of shape (1, n_mels, T).
    """
    expected_length = int(round(chunk_seconds * sampling_frequency))
    assert len(waveform) == expected_length

    mel_fb = librosa.filters.mel(sr=sampling_frequency, n_fft=n_fft, n_mels=n_mels)
    window = np.hanning(n_fft).astype(np.float32)

    n_frames  = len(waveform) // hop_length
    audio_buf = np.zeros(n_fft, dtype=np.float32)  # zero-init matches C++ static array
    frames    = np.empty((n_mels, n_frames), dtype=np.float32)

    for i in range(n_frames):
        # Slide buffer left, fill right — mirrors memmove in spectrogram_process_hop()
        audio_buf[:n_fft - hop_length] = audio_buf[hop_length:]
        audio_buf[n_fft - hop_length:] = waveform[i * hop_length : (i + 1) * hop_length]

        spectrum      = np.fft.rfft(audio_buf * window)
        power_spec    = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
        mel_energies  = mel_fb @ power_spec                   # (n_mels,)
        frames[:, i]  = np.log(mel_energies + LOG_OFFSET)

    assert frames.shape == (n_mels, n_frames)
    return frames[np.newaxis].astype(np.float32)              # (1, n_mels, T)
