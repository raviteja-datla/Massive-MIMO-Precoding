import numpy as np

from mimo_precoding.modulation import qam16_demodulate, qam16_modulate, qpsk_demodulate, qpsk_modulate


def test_qpsk_round_trip_no_noise():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1000, 2))
    symbols = qpsk_modulate(bits)
    bits_hat = qpsk_demodulate(symbols)
    np.testing.assert_array_equal(bits, bits_hat)


def test_qpsk_unit_energy_constellation():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=(1000, 2))
    symbols = qpsk_modulate(bits)
    np.testing.assert_allclose(np.abs(symbols), 1.0)


def test_qam16_round_trip_no_noise():
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, size=(1000, 4))
    symbols = qam16_modulate(bits)
    bits_hat = qam16_demodulate(symbols)
    np.testing.assert_array_equal(bits, bits_hat)


def test_qam16_unit_average_energy():
    """Unlike QPSK, 16-QAM points don't share a common magnitude, so this
    checks the exact average over all 16 equally-likely constellation points
    rather than a Monte Carlo sample (whose average only converges to 1)."""
    bits = np.array([[b0, b1, b2, b3] for b0 in (0, 1) for b1 in (0, 1) for b2 in (0, 1) for b3 in (0, 1)])
    symbols = qam16_modulate(bits)
    avg_energy = np.mean(np.abs(symbols) ** 2)
    assert abs(avg_energy - 1.0) < 1e-9


def test_qam16_all_sixteen_points_reachable():
    bits = np.array([[b0, b1, b2, b3] for b0 in (0, 1) for b1 in (0, 1) for b2 in (0, 1) for b3 in (0, 1)])
    symbols = qam16_modulate(bits)
    assert len(np.unique(symbols)) == 16
