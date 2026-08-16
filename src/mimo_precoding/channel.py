import numpy as np


def generate_channel(M: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """Draw H in C^{M x K}, i.i.d. Rayleigh fading: each entry ~ CN(0, 1).

    Column k is the channel from the M base-station antennas to user k.
    """
    real = rng.standard_normal((M, K))
    imag = rng.standard_normal((M, K))
    return (real + 1j * imag) / np.sqrt(2)


def correlation_matrix(M: int, rho: float) -> np.ndarray:
    """Exponential (Kronecker-style) transmit-side spatial correlation matrix:
    R[i,j] = rho^|i-j|, rho in [0, 1). rho=0 gives the identity matrix
    (uncorrelated antennas), higher rho means antennas closer together in
    index see more similar fading.
    """
    idx = np.arange(M)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def generate_correlated_channel(
    M: int, K: int, rho: float, rng: np.random.Generator
) -> np.ndarray:
    """Kronecker-correlated Rayleigh fading: H = R^(1/2) @ H_iid.

    R is the BS-side transmit correlation matrix, shared across users (each
    user has a single antenna, so there is no receive-side correlation to
    model). R has unit diagonal, so per-antenna average power is unchanged
    from the i.i.d. case — only the spatial structure differs. rho=0 reduces
    exactly to `generate_channel`.
    """
    H_iid = generate_channel(M, K, rng)
    R = correlation_matrix(M, rho)
    eigvals, eigvecs = np.linalg.eigh(R)
    R_sqrt = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0, None))) @ eigvecs.T
    return R_sqrt @ H_iid
