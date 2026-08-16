import numpy as np


def compute_sinr(H: np.ndarray, W: np.ndarray, sigma2: float) -> np.ndarray:
    """Per-user SINR using the power-normalized precoder W.

    SINR_k = |h_k^H w_k|^2 / (sum_{j != k} |h_k^H w_j|^2 + sigma2)
    """
    G = H.conj().T @ W  # G[k, j] = h_k^H w_j
    gains = np.abs(G) ** 2
    signal = np.diag(gains)
    interference = gains.sum(axis=1) - signal
    return signal / (interference + sigma2)


def sum_rate(sinr: np.ndarray) -> float:
    """Sum spectral efficiency in bits/s/Hz: sum_k log2(1 + SINR_k)."""
    sinr = np.clip(sinr, 0.0, None)
    return float(np.sum(np.log2(1.0 + sinr)))
