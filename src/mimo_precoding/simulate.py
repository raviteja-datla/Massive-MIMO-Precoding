from typing import Callable, Literal, Sequence

import numpy as np

from mimo_precoding.channel import generate_channel
from mimo_precoding.metrics import compute_sinr, sum_rate
from mimo_precoding.modulation import QPSK, Modulation
from mimo_precoding.power import normalize_power
from mimo_precoding.precoders import mmse_precoder, mrt_precoder, regularized_precoder

PrecoderName = Literal["MRT", "ZF", "RZF", "MMSE"]

# A channel model is any (M, K, rng) -> H callable. simulate.py stays agnostic
# to which one is used — `generate_channel` (i.i.d.) and
# `generate_correlated_channel` (Kronecker) from channel.py are both valid,
# kept as fully separate functions rather than merged behind a flag here.
ChannelFn = Callable[[int, int, np.random.Generator], np.ndarray]

# Transmit power is normalized to 1; SNR is controlled entirely through sigma2.
TX_POWER = 1.0


def _sigma2_from_snr_db(snr_db: float) -> float:
    return TX_POWER / (10 ** (snr_db / 10))


def build_precoder(
    name: PrecoderName,
    H: np.ndarray,
    sigma2: float,
    P: float,
    K: int,
    rzf_xi: float = 1.0,
) -> np.ndarray:
    """Dispatch to the raw (unnormalized) precoder matching `name`."""
    if name == "MRT":
        return mrt_precoder(H)
    if name == "ZF":
        return regularized_precoder(H, xi=0.0)
    if name == "RZF":
        return regularized_precoder(H, xi=rzf_xi)
    if name == "MMSE":
        return mmse_precoder(H, sigma2, P, K)
    raise ValueError(f"Unknown precoder: {name}")


def _rate_for_precoder(
    H: np.ndarray, snr_db: float, precoder: PrecoderName, rzf_xi: float = 1.0
) -> float:
    K = H.shape[1]
    sigma2 = _sigma2_from_snr_db(snr_db)
    W_raw = build_precoder(precoder, H, sigma2, TX_POWER, K, rzf_xi)
    W = normalize_power(W_raw, TX_POWER)
    sinr = compute_sinr(H, W, sigma2)
    return sum_rate(sinr)


def _ber_for_precoder(
    H: np.ndarray,
    snr_db: float,
    precoder: PrecoderName,
    bits: np.ndarray,
    noise: np.ndarray,
    rzf_xi: float = 1.0,
    modulation: Modulation = QPSK,
) -> float:
    """Bit-error rate for one channel realization and one fixed batch of
    (bits, noise), so multiple precoders can be compared under identical
    transmitted data and noise. Each user equalizes by its own effective
    channel gain h_k^H w_k (genie-aided, i.e. no channel-estimation error).
    """
    K = H.shape[1]
    sigma2 = _sigma2_from_snr_db(snr_db)
    W_raw = build_precoder(precoder, H, sigma2, TX_POWER, K, rzf_xi)
    W = normalize_power(W_raw, TX_POWER)

    s = modulation.modulate(bits)  # (n_symbols, K)
    X = s @ W.T  # (n_symbols, M), X[t] = W @ s[t]
    Y = X @ H.conj() + noise  # (n_symbols, K), Y[t,k] = h_k^H x[t] + n_k[t]

    effective_gain = np.diag(H.conj().T @ W)  # (K,), h_k^H w_k
    z = Y / effective_gain
    bits_hat = modulation.demodulate(z)

    return float(np.mean(bits_hat != bits))


def run_trial(
    M: int,
    K: int,
    snr_db: float,
    precoder: PrecoderName,
    rng: np.random.Generator,
    channel_fn: ChannelFn = generate_channel,
    rzf_xi: float = 1.0,
) -> float:
    """Draw one channel realization and return the resulting sum-rate."""
    H = channel_fn(M, K, rng)
    return _rate_for_precoder(H, snr_db, precoder, rzf_xi)


def run_ber_trial(
    M: int,
    K: int,
    snr_db: float,
    precoder: PrecoderName,
    rng: np.random.Generator,
    channel_fn: ChannelFn = generate_channel,
    n_symbols: int = 2000,
    rzf_xi: float = 1.0,
    modulation: Modulation = QPSK,
) -> float:
    """Draw one channel realization and n_symbols symbols per user; return
    the Monte Carlo bit-error rate under the given modulation scheme."""
    H = channel_fn(M, K, rng)
    sigma2 = _sigma2_from_snr_db(snr_db)
    bits = rng.integers(0, 2, size=(n_symbols, K, modulation.bits_per_symbol))
    noise = (
        rng.standard_normal((n_symbols, K)) + 1j * rng.standard_normal((n_symbols, K))
    ) * np.sqrt(sigma2 / 2)
    return _ber_for_precoder(H, snr_db, precoder, bits, noise, rzf_xi, modulation)


def sweep(
    param_name: str,
    param_values: Sequence,
    fixed: dict,
    precoders: Sequence[PrecoderName],
    n_trials: int,
    seed: int,
    channel_fn: ChannelFn = generate_channel,
) -> dict[str, np.ndarray]:
    """Monte Carlo average sum-rate per precoder, varying `param_name` over
    `param_values` while holding the rest of `fixed` constant.

    Within each trial, all precoders are evaluated on the *same* channel
    realization, so per-precoder curves are directly comparable (same noise
    in the comparison, not just the same expectation).
    """
    results = {p: np.zeros(len(param_values)) for p in precoders}
    for i, value in enumerate(param_values):
        params = {**fixed, param_name: value}
        M, K, snr_db = params["M"], params["K"], params["snr_db"]
        rzf_xi = params.get("rzf_xi", 1.0)
        rng = np.random.default_rng(seed + i)

        accum = {p: 0.0 for p in precoders}
        for _ in range(n_trials):
            H = channel_fn(M, K, rng)
            for p in precoders:
                accum[p] += _rate_for_precoder(H, snr_db, p, rzf_xi)

        for p in precoders:
            results[p][i] = accum[p] / n_trials

    return results


def ber_sweep(
    param_name: str,
    param_values: Sequence,
    fixed: dict,
    precoders: Sequence[PrecoderName],
    n_trials: int,
    n_symbols: int,
    seed: int,
    channel_fn: ChannelFn = generate_channel,
    modulation: Modulation = QPSK,
) -> dict[str, np.ndarray]:
    """Monte Carlo average BER per precoder, analogous to `sweep` but for
    bit-error rate. Each trial is one channel realization (coherence block)
    over which n_symbols symbols per user are transmitted; all precoders in a
    trial share the same channel, bits, and noise for a fair comparison.
    """
    results = {p: np.zeros(len(param_values)) for p in precoders}
    for i, value in enumerate(param_values):
        params = {**fixed, param_name: value}
        M, K, snr_db = params["M"], params["K"], params["snr_db"]
        rzf_xi = params.get("rzf_xi", 1.0)
        sigma2 = _sigma2_from_snr_db(snr_db)
        rng = np.random.default_rng(seed + i)

        accum = {p: 0.0 for p in precoders}
        for _ in range(n_trials):
            H = channel_fn(M, K, rng)
            bits = rng.integers(0, 2, size=(n_symbols, K, modulation.bits_per_symbol))
            noise = (
                rng.standard_normal((n_symbols, K)) + 1j * rng.standard_normal((n_symbols, K))
            ) * np.sqrt(sigma2 / 2)
            for p in precoders:
                accum[p] += _ber_for_precoder(H, snr_db, p, bits, noise, rzf_xi, modulation)

        for p in precoders:
            results[p][i] = accum[p] / n_trials

    return results
