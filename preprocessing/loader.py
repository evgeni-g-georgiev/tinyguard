"""WAV file loading."""
import librosa


def load_audio(file_path, sampling_frequency, mono=True, channel=None):
    """Load a WAV file as a 1-D waveform at ``sampling_frequency``.

    ``channel`` picks a single channel (0-indexed). ``mono`` is ignored when
    ``channel`` is given; otherwise ``mono=True`` mixes all channels.

    Returns ``(waveform, sr)``.
    """
    if channel is not None:
        waveform, sr = librosa.load(file_path, sr=sampling_frequency, mono=False)
        return waveform[channel], sr
    return librosa.load(file_path, sr=sampling_frequency, mono=mono)
