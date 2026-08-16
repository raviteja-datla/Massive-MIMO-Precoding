import numpy as np

from mimo_precoding.channel import generate_channel
from mimo_precoding.power import normalize_power
from mimo_precoding.precoders import mrt_precoder


def test_power_normalization_exact():
    rng = np.random.default_rng(5)
    M, K = 16, 4
    H = generate_channel(M, K, rng)
    W_raw = mrt_precoder(H)
    P = 2.5

    W = normalize_power(W_raw, P)

    assert abs(np.linalg.norm(W, "fro") ** 2 - P) < 1e-10


def test_transmit_power_constraint_monte_carlo():
    rng = np.random.default_rng(6)
    M, K = 16, 4
    H = generate_channel(M, K, rng)
    W = normalize_power(mrt_precoder(H), P=3.0)

    n_draws = 5000
    s = (rng.standard_normal((n_draws, K)) + 1j * rng.standard_normal((n_draws, K))) / np.sqrt(2)
    x = s @ W.T
    avg_power = np.mean(np.sum(np.abs(x) ** 2, axis=1))

    assert abs(avg_power - 3.0) < 0.1
