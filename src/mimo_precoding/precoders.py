import numpy as np


def mrt_precoder(H: np.ndarray) -> np.ndarray:
    """Maximum Ratio Transmission: w_k = h_k, ignores inter-user interference."""
    return H


def regularized_precoder(H: np.ndarray, xi: float) -> np.ndarray:
    """W_raw = H (H^H H + xi * I_K)^-1.

    xi == 0.0 gives exact Zero Forcing (requires M > K): interference is
    forced to exactly zero. Larger xi trades interference cancellation for
    reduced noise/power penalty; as xi grows, direction approaches MRT.
    """
    K = H.shape[1]
    gram = H.conj().T @ H + xi * np.eye(K)
    return H @ np.linalg.inv(gram)


def mmse_precoder(H: np.ndarray, sigma2: float, P: float, K: int) -> np.ndarray:
    """MMSE precoder: the regularized precoder evaluated at the theoretically
    optimal regularization xi = K * sigma2 / P.
    """
    xi = K * sigma2 / P
    return regularized_precoder(H, xi)
