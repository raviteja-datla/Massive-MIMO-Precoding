import numpy as np

from mimo_precoding.channel import generate_channel
from mimo_precoding.metrics import compute_sinr, sum_rate
from mimo_precoding.power import normalize_power
from mimo_precoding.precoders import mmse_precoder, mrt_precoder, regularized_precoder

PRECODERS = {
    "MRT": lambda H, sigma2, P, K: mrt_precoder(H),
    "ZF": lambda H, sigma2, P, K: regularized_precoder(H, xi=0.0),
    "RZF": lambda H, sigma2, P, K: regularized_precoder(H, xi=1.0),
    "MMSE": lambda H, sigma2, P, K: mmse_precoder(H, sigma2, P, K),
}


def test_sinr_nonnegative_and_finite():
    rng = np.random.default_rng(7)
    M, K = 32, 4
    H = generate_channel(M, K, rng)
    sigma2, P = 0.1, 1.0

    for build in PRECODERS.values():
        W = normalize_power(build(H, sigma2, P, K), P)
        sinr = compute_sinr(H, W, sigma2)
        assert np.all(sinr >= -1e-9)
        assert np.all(np.isfinite(sinr))


def test_sum_rate_nonnegative_and_finite():
    rng = np.random.default_rng(8)
    M, K = 32, 4
    H = generate_channel(M, K, rng)
    sigma2, P = 0.1, 1.0

    for build in PRECODERS.values():
        W = normalize_power(build(H, sigma2, P, K), P)
        sinr = compute_sinr(H, W, sigma2)
        rate = sum_rate(sinr)
        assert rate >= 0.0
        assert np.isfinite(rate)


def test_zf_sinr_equals_signal_over_noise():
    rng = np.random.default_rng(9)
    M, K = 32, 4
    H = generate_channel(M, K, rng)
    sigma2, P = 0.1, 1.0

    W = normalize_power(regularized_precoder(H, xi=0.0), P)
    sinr = compute_sinr(H, W, sigma2)

    G = H.conj().T @ W
    expected = np.abs(np.diag(G)) ** 2 / sigma2

    np.testing.assert_allclose(sinr, expected, rtol=1e-8)
