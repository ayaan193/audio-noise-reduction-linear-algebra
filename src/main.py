import os
from scipy.io import wavfile
from load_audio import load_clean_matrix, load_noisy
from gram_schmidt import gram_schmidt
from projection import project

# -------- PATHS --------
clean_folder = "../data/clean/"
noisy_file = "../data/noisy/sp01_train_sn0.wav"
output_file = "../output/denoised.wav"

# -------- LOAD CLEAN DATA --------
print("Loading clean data...")
X = load_clean_matrix(clean_folder)
print("Matrix shape:", X.shape)

# -------- ORTHONORMAL BASIS --------
print("Applying Gram-Schmidt...")
Q = gram_schmidt(X)

# -------- LOAD NOISY SIGNAL --------
print("Loading noisy signal...")
noisy, rate = load_noisy(noisy_file, X.shape[0])

# -------- PROJECTION (DENOISING) --------
print("Applying projection...")
denoised = project(Q, noisy)

# -------- SAVE OUTPUT --------
denoised = denoised.astype("int16")
wavfile.write(output_file, rate, denoised)

print("Denoised file saved at:", output_file)