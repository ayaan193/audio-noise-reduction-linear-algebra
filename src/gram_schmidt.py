import numpy as np

def gram_schmidt(X):
    n, m = X.shape
    Q = np.zeros((n, m))

    for j in range(m):
        v = X[:, j]

        for i in range(j):
            rij = np.dot(Q[:, i], v)
            v = v - rij * Q[:, i]

        norm = np.linalg.norm(v)

        if norm > 1e-10:  # avoid division by zero
            Q[:, j] = v / norm

    return Q