import librosa
import numpy as np

SR     = 16000
N_FFT  = 1024
N_MELS = 64
N_BINS = N_FFT // 2 + 1   # 513

fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS).astype(np.float32)
# fb.shape == (64, 513)

with open("mel_filterbank.h", "w") as f:
    f.write("#pragma once\n")
    f.write(f"// librosa.filters.mel(sr={SR}, n_fft={N_FFT}, n_mels={N_MELS})\n")
    f.write(f"const float MEL_FB[{N_MELS}][{N_BINS}] = {{\n")
    for m in range(N_MELS):
        row = ", ".join(f"{v:.6e}f" for v in fb[m])
        f.write(f"  {{{row}}},\n")
    f.write("};\n")

print(f"Written mel_filterbank.h  ({N_MELS} x {N_BINS} = {N_MELS*N_BINS} floats)")
