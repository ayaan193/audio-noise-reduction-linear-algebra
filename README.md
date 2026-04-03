# 🎧 Audio Noise Reduction using Linear Algebra

## 📌 Overview

This project demonstrates how linear algebra concepts can be used to remove noise from audio signals. Clean signals are used to construct a signal subspace, and noisy signals are projected onto this subspace to obtain a denoised output.

---

## 🧠 Concepts Used

* Matrix Representation
* Gram-Schmidt Orthogonalization
* Orthogonal Projection
* Least Squares Approximation

---

## ⚙️ How It Works

1. Convert clean audio signals into vectors
2. Form a matrix ( X ) where each column represents a signal
3. Apply Gram-Schmidt to obtain an orthonormal basis ( Q )
4. Project noisy signal onto this subspace
5. Obtain denoised output

---

## 🧮 Key Formula

```
x_hat = Q * Q^T * x
```

---

## 📁 Project Structure

```
LAA-miniproject/
├── src/
├── data/
├── output/
└── README.md
```

---

## ▶️ How to Run

```
cd src
python demonstration.py
```

---

## 📊 Output

* Generates a denoised audio file
* Shows step-by-step processing

---

## 🎯 Result

Noise is reduced by projecting the noisy signal onto the clean signal subspace.

---

## 👨‍💻 Author

Ayaan Ahmed
