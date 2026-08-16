import numpy as np

from mimo_precoding.channel import correlation_matrix, generate_channel, generate_correlated_channel


def test_generate_channel_shape_and_variance():
    rng = np.random.default_rng(0)
    M, K = 32, 4
    n_draws = 2000
    samples = np.stack([generate_channel(M, K, rng) for _ in range(n_draws)])

    assert samples.shape[1:] == (M, K)
    assert np.abs(samples.mean()) < 0.05
    assert abs(samples.var() - 1.0) < 0.05


def test_correlation_matrix_identity_at_rho_zero():
    R = correlation_matrix(8, 0.0)
    np.testing.assert_allclose(R, np.eye(8))


def test_correlation_matrix_symmetric_with_unit_diagonal():
    R = correlation_matrix(8, 0.6)
    np.testing.assert_allclose(np.diag(R), np.ones(8))
    np.testing.assert_allclose(R, R.T)


def test_correlated_channel_matches_iid_at_rho_zero():
    H_iid = generate_channel(16, 4, np.random.default_rng(42))
    H_corr = generate_correlated_channel(16, 4, 0.0, np.random.default_rng(42))
    np.testing.assert_allclose(H_corr, H_iid, atol=1e-10)


def test_correlated_channel_preserves_per_antenna_power():
    rng = np.random.default_rng(43)
    M, K = 32, 4
    n_draws = 500
    samples = np.stack([generate_correlated_channel(M, K, 0.7, rng) for _ in range(n_draws)])
    assert abs(samples.var() - 1.0) < 0.05
