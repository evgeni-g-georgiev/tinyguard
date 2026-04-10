import yaml
import librosa # For audio laoding


def load_audio(file_path, sampling_frequency, mono):
    """Load a full .wav audio file.

    Comments: This function uses librosa rather than soundfile 
    for loading audio because it allows us to: (1) resample the 
    audio to the target sampling frequency, even if the original 
    file was stored at a different sampling rate; (2) convert the
    audio to mono, even if the original file is stereo. We always load 
    at a fixed sampling frequency and in mono channel format.

    Args:
        file_path: Path to the .wav file.
        sampling_frequency: Target sampling frequency in Hz.
        mono: If True, convert the audio to mono during loading.

    Returns:
        waveform: 1D array containing the audio samples.
        sr: Sampling frequency of the loaded waveform.

    Note: If the audio duration is T seconds, then the waveform length is:
    len(waveform) = sr * T
    """
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

    Raises:                                                                           
        ValueError: If sampling_frequency * chunk_seconds is not an                   
        integer number of samples.
    """
    chunk_size_exact = sampling_frequency * chunk_seconds
    chunk_size = int(round(chunk_size_exact))

    if abs(chunk_size_exact - chunk_size) > 1e-9:
        raise ValueError(
            f"chunk_seconds ({chunk_seconds}) x sampling_frequency "                  
            f"({sampling_frequency}) = {chunk_size_exact}, which is not "
            f"a whole number of samples. Adjust one of them so the "                  
            f"product is an integer."
        )

    n_chunks = len(waveform) // chunk_size
    return [ 
        waveform[i * chunk_size: (i + 1) * chunk_size] for i in range(n_chunks)
    ]

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