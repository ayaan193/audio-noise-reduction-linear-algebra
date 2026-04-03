import os
import numpy as np
from scipy.io import wavfile

def load_clean_matrix(folder_path):
    signals = []

    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".wav")])

    for file in files:
        rate, data = wavfile.read(os.path.join(folder_path, file))

        data = data.astype(np.float32)

        # convert stereo to mono
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        signals.append(data)

    # make all same length
    min_len = min(len(s) for s in signals)
    signals = [s[:min_len] for s in signals]

    # stack as matrix (columns = signals)
    X = np.column_stack(signals)

    return X


def load_noisy(file_path, target_len):
    rate, data = wavfile.read(file_path)

    data = data.astype(np.float32)

    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # match length
    data = data[:target_len]

    return data, rate