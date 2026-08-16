import numpy as np

from mimo_precoding.channel import generate_channel
from mimo_precoding.precoders import mmse_precoder, mrt_precoder, regularized_precoder


def _off_diagonal_gains(H, W):
    G = H.conj().T @ W
    return G - np.diag(np.diag(G))


def test_zf_zeros_interference():
    rng = np.random.default_rng(1)
    M, K = 32, 4
    H = generate_channel(M, K, rng)
    W = regularized_precoder(H, xi=0.0)
    off_diag = _off_diagonal_gains(H, W)
    assert np.max(np.abs(off_diag)) < 1e-10


def test_mmse_matches_regularized_at_derived_xi():
    rng = np.random.default_rng(2)
    M, K = 32, 4
    H = generate_channel(M, K, rng)
    sigma2, P = 0.1, 1.0

    W_mmse = mmse_precoder(H, sigma2, P, K)
    W_reg = regularized_precoder(H, xi=K * sigma2 / P)

    np.testing.assert_allclose(W_mmse, W_reg)


def test_rzf_xi_zero_equals_zf():
    rng = np.random.default_rng(3)
    M, K = 32, 4
    H = generate_channel(M, K, rng)

    W_rzf = regularized_precoder(H, xi=0.0)
    W_zf_direct = H @ np.linalg.inv(H.conj().T @ H)

    np.testing.assert_allclose(W_rzf, W_zf_direct)


def test_rzf_large_xi_aligns_with_mrt_direction():
    rng = np.random.default_rng(4)
    M, K = 32, 4
    H = generate_channel(M, K, rng)

    W_rzf = regularized_precoder(H, xi=1e6)
    W_mrt = mrt_precoder(H)

    rzf_dirs = W_rzf / np.linalg.norm(W_rzf, axis=0, keepdims=True)
    mrt_dirs = W_mrt / np.linalg.norm(W_mrt, axis=0, keepdims=True)

    cos_sim = np.abs(np.sum(rzf_dirs.conj() * mrt_dirs, axis=0))
    assert np.all(cos_sim > 1 - 1e-6)
