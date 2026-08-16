import numpy as np

from mimo_precoding.channel import generate_channel, generate_correlated_channel
from mimo_precoding.modulation import QAM16, QPSK
from mimo_precoding.simulate import run_ber_trial, run_trial


def test_ber_decreases_with_snr():
    M, K = 32, 4
    ber_low = run_ber_trial(M, K, snr_db=-5, precoder="MMSE", rng=np.random.default_rng(10), n_symbols=5000)
    ber_high = run_ber_trial(M, K, snr_db=20, precoder="MMSE", rng=np.random.default_rng(11), n_symbols=5000)
    assert ber_high < ber_low


def test_ber_bounded_in_unit_interval():
    ber = run_ber_trial(32, 4, snr_db=10, precoder="ZF", rng=np.random.default_rng(12), n_symbols=2000)
    assert 0.0 <= ber <= 1.0


def test_run_trial_accepts_correlated_channel_fn():
    def correlated(M, K, rng):
        return generate_correlated_channel(M, K, 0.5, rng)

    rate = run_trial(32, 4, snr_db=10, precoder="MMSE", rng=np.random.default_rng(20), channel_fn=correlated)
    assert rate >= 0.0 and np.isfinite(rate)


def test_run_trial_default_channel_fn_is_iid():
    rate_default = run_trial(32, 4, snr_db=10, precoder="MRT", rng=np.random.default_rng(30))
    rate_explicit = run_trial(
        32, 4, snr_db=10, precoder="MRT", rng=np.random.default_rng(30), channel_fn=generate_channel
    )
    assert rate_default == rate_explicit


def test_ber_default_modulation_is_qpsk():
    ber_default = run_ber_trial(32, 4, snr_db=10, precoder="ZF", rng=np.random.default_rng(40), n_symbols=2000)
    ber_explicit = run_ber_trial(
        32, 4, snr_db=10, precoder="ZF", rng=np.random.default_rng(40), n_symbols=2000, modulation=QPSK
    )
    assert ber_default == ber_explicit


def test_qam16_ber_worse_than_qpsk_at_same_snr():
    """16-QAM packs 4 bits/symbol into the same unit energy budget as QPSK's
    2 bits/symbol, so its constellation points sit closer together and it
    should show a higher bit-error rate at identical SNR."""
    M, K, snr_db = 32, 4, 10.0
    ber_qpsk = run_ber_trial(
        M, K, snr_db, "MMSE", rng=np.random.default_rng(41), n_symbols=5000, modulation=QPSK
    )
    ber_16qam = run_ber_trial(
        M, K, snr_db, "MMSE", rng=np.random.default_rng(42), n_symbols=5000, modulation=QAM16
    )
    assert ber_16qam > ber_qpsk


def test_mrt_beats_zf_at_tight_ratio_and_low_snr():
    """The one regime where MRT wins on sum-rate over ZF: a tight M/K ratio
    (ZF pays a heavy power penalty) combined with low SNR (noise, not
    interference, dominates, so cancelling interference isn't worth it)."""
    M, K, snr_db = 12, 8, -20.0
    rate_mrt = run_trial(M, K, snr_db, "MRT", rng=np.random.default_rng(50))
    rate_zf = run_trial(M, K, snr_db, "ZF", rng=np.random.default_rng(50))
    assert rate_mrt > rate_zf
