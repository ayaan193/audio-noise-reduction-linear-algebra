import numpy as np

def project(Q, y):
    # projection: y_hat = Q Q^T y
    return Q @ (Q.T @ y)