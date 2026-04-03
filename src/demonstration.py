import os
import numpy as np
from scipy.io import wavfile
from load_audio import load_clean_matrix, load_noisy
from gram_schmidt import gram_schmidt
from projection import project

# -------- PATH SETUP --------
base_dir = os.path.dirname(os.path.abspath(__file__))

clean_folder = os.path.join(base_dir, "../data/clean/")
noisy_file = os.path.join(base_dir, "../data/noisy/sp01_train_sn0.wav")
output_file = os.path.join(base_dir, "../output/denoised_demo.wav")

print("\n================ STEP 1: MATRIX REPRESENTATION ================\n")

# -------- LOAD CLEAN DATA --------
X = load_clean_matrix(clean_folder)

print("Matrix Shape (rows = samples, cols = signals):", X.shape)
print("\nSample of Matrix (first 5 rows, 3 columns):\n", X[:5, :3])


print("\n================ STEP 2: STRUCTURE OF SPACE ================\n")
print("Clean signals define the SIGNAL SUBSPACE.")
print("Noisy signal will be projected onto this space.\n")


print("\n================ STEP 3: REMOVE REDUNDANCY ================\n")
print("Gram-Schmidt will automatically remove linearly dependent components.\n")


print("\n================ STEP 4: ORTHOGONALIZATION ================\n")

Q = gram_schmidt(X)

print("Orthonormal Basis Shape:", Q.shape)

# Check orthogonality
identity_check = Q.T @ Q
print("\nQ^T Q (should be close to Identity):\n", identity_check[:5, :5])


print("\n================ STEP 5: LOAD NOISY SIGNAL ================\n")

noisy, rate = load_noisy(noisy_file, X.shape[0])

print("Noisy signal sample:\n", noisy[:10])


print("\n================ STEP 6: PROJECTION (NOISE REDUCTION) ================\n")

denoised = project(Q, noisy)

print("Denoised signal sample:\n", denoised[:10])


print("\n================ STEP 7: LEAST SQUARES INTERPRETATION ================\n")
print("Projection minimizes the error between noisy signal and subspace.")
print("Equivalent to solving a least squares problem.\n")


print("\n================ STEP 8: NOISE ANALYSIS ================\n")

noise = noisy - denoised

print("Noise sample:\n", noise[:10])
print("Noise energy (L2 norm):", np.linalg.norm(noise))


print("\n================ STEP 9: SAVE OUTPUT ================\n")

# Convert to int16 for saving
denoised_int = denoised.astype(np.int16)
wavfile.write(output_file, rate, denoised_int)

print("Denoised file saved at:", output_file)


print("\n================ FINAL OUTPUT ================\n")
print("Compare:")
print("1. Original noisy audio")
print("2. Denoised audio (output folder)")
