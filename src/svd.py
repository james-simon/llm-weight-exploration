"""
SVD computation and result persistence.
Results saved as .npz files under results/.
"""

from pathlib import Path
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


def compute_svd(matrix: np.ndarray, full_matrices: bool = False) -> dict:
    """
    Thin SVD of matrix. Returns dict with S, U, Vh.
    Set full_matrices=False (default) to get economy SVD.
    """
    U, S, Vh = np.linalg.svd(matrix, full_matrices=full_matrices)
    return {"U": U, "S": S, "Vh": Vh}


def compute_singular_values(matrix: np.ndarray) -> np.ndarray:
    """Just singular values — faster than full SVD."""
    return np.linalg.svd(matrix, compute_uv=False)


def save_results(name: str, data: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.npz"
    np.savez_compressed(path, **data)
    return path


def load_results(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No results found at {path}")
    f = np.load(path, allow_pickle=True)
    return {k: f[k] for k in f.files}


def results_exist(name: str) -> bool:
    return (RESULTS_DIR / f"{name}.npz").exists()
