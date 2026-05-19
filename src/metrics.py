"""
Spectral analysis metrics.
All functions take numpy arrays and return numpy arrays or scalars.
"""

import numpy as np


def subspace_alignment(U1: np.ndarray, U2: np.ndarray, k: int) -> float:
    """
    Grassmann alignment between top-k left singular subspaces.
    Returns mean squared cosine similarity (0=orthogonal, 1=identical).
    """
    A = U1[:, :k]
    B = U2[:, :k]
    M = A.T @ B
    return float(np.sum(M ** 2) / k)


def principal_angles(U1: np.ndarray, U2: np.ndarray, k: int) -> np.ndarray:
    """
    Principal angles (in radians) between top-k subspaces of U1 and U2.
    Returns array of length k.
    """
    A = U1[:, :k]
    B = U2[:, :k]
    M = A.T @ B
    sigma = np.linalg.svd(M, compute_uv=False)
    sigma = np.clip(sigma, -1, 1)
    return np.arccos(sigma)


def spectral_entropy(S: np.ndarray) -> float:
    """Normalized entropy of the squared singular value distribution."""
    p = S ** 2
    p = p / p.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(len(S)))


def stable_rank(S: np.ndarray) -> float:
    """Ratio of squared Frobenius norm to squared spectral norm."""
    return float((S ** 2).sum() / S[0] ** 2)


def effective_rank(S: np.ndarray) -> float:
    """Roy & Vetterli effective rank: exp(spectral_entropy * log(n))."""
    p = S ** 2
    p = p / p.sum()
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log(p)))
    return float(np.exp(entropy))
