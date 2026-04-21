import librosa # For audio laoding


def load_audio(file_path, sampling_frequency, mono=True, channel=None):
    """Load a full .wav audio file.

    Args:
        file_path: Path to the .wav file.
        sampling_frequency: Target sampling frequency in Hz.
        mono: If True, mix all channels to mono (ignored when channel is set).
        channel: If set, return only this zero-indexed microphone channel as a
                 1-D waveform. MIMII files have 8 channels (0-7).

    Returns:
        waveform: 1D array containing the audio samples.
        sr: Sampling frequency of the loaded waveform.
    """
    if channel is not None:
        waveform, sr = librosa.load(file_path, sr=sampling_frequency, mono=False)
        return waveform[channel], sr
    waveform, sr = librosa.load(file_path, sr=sampling_frequency, mono=mono)
    return waveform, sr


def split_into_chunks(waveform, sampling_frequency, chunk_seconds):
    """Split a waveform into fixed-length chunks.

    Args:
        waveform: 1D array containing the audio samples.
        sampling_frequency: Sampling frequency in Hz.
        chunk_seconds: Length of each chunk in seconds.

    Returns:
        A list of 1D waveform chunks of equal length.
    """
    chunk_size = int(round(sampling_frequency * chunk_seconds))
    chunks = []

    for start in range(0, len(waveform), chunk_size):
        end = start + chunk_size
        chunk = waveform[start:end]

        if len(chunk) == chunk_size:
            chunks.append(chunk)

    assert len(chunks) == len(waveform) // chunk_size

    return chunks

def iter_audio_chunks(file_path, mono, sampling_frequency, chunk_seconds):
    """ Yield audio chunks one by one from a single audio file.

    This first loads the full audio file, then splits it into fixed-length
    chunks, and yields one chunk at a time.

    Args:
        file_path: Path to the .wav file.
        sampling_frequency: Target sampling frequency in Hz.
        chunk_seconds: Length of each chunk in seconds.

    Yields:
        A tuple (chunk, sr), where chunk is a 1D waveform chunk and sr is
        the sampling frequency.
    """
    waveform, sr = load_audio(file_path=file_path, sampling_frequency=sampling_frequency, mono=mono)

    chunks = split_into_chunks(waveform, sr, chunk_seconds)

    for chunk in chunks:
        yield chunk, sr
