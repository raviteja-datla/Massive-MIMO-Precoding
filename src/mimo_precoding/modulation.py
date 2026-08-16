from dataclasses import dataclass
from typing import Callable

import numpy as np


def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """Gray-coded QPSK: bits shape (..., 2) of {0,1} -> unit-energy complex
    symbols, one symbol per pair of bits. Constellation points are
    (+-1 +- 1j) / sqrt(2), each with |symbol| = 1.
    """
    real = 1.0 - 2.0 * bits[..., 0]
    imag = 1.0 - 2.0 * bits[..., 1]
    return (real + 1j * imag) / np.sqrt(2)


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Nearest-neighbor QPSK detection: sign of each quadrature component,
    inverse of `qpsk_modulate`. Returns bits shape (..., 2).
    """
    b0 = (np.real(symbols) < 0).astype(np.int64)
    b1 = (np.imag(symbols) < 0).astype(np.int64)
    return np.stack([b0, b1], axis=-1)


def _pam4_modulate(b_lo: np.ndarray, b_hi: np.ndarray) -> np.ndarray:
    """Gray-coded 4-level PAM: (b_lo,b_hi) -> amplitude in {-3,-1,+1,+3}.
    Mapping: (0,0)->-3 (0,1)->-1 (1,1)->+1 (1,0)->+3 (adjacent levels differ
    by exactly one bit).
    """
    sign = np.where(b_lo == 1, 1.0, -1.0)
    magnitude = np.where(b_hi == 1, 1.0, 3.0)
    return sign * magnitude


def _pam4_demodulate(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `_pam4_modulate`: nearest-level decision from the raw
    (unnormalized) amplitude. b_lo is the sign; b_hi flags the inner pair of
    levels (|amplitude| < 2)."""
    b_lo = (x >= 0).astype(np.int64)
    b_hi = (np.abs(x) < 2).astype(np.int64)
    return b_lo, b_hi


def qam16_modulate(bits: np.ndarray) -> np.ndarray:
    """Gray-coded 16-QAM: bits shape (..., 4) -> unit-average-energy complex
    symbols. bits[...,0:2] set the in-phase amplitude, bits[...,2:4] set the
    quadrature amplitude, each independently Gray-coded PAM-4 over
    {-3,-1,+1,+3}, jointly normalized (/sqrt(10)) so E[|symbol|^2] = 1.
    """
    I = _pam4_modulate(bits[..., 0], bits[..., 1])
    Q = _pam4_modulate(bits[..., 2], bits[..., 3])
    return (I + 1j * Q) / np.sqrt(10)


def qam16_demodulate(symbols: np.ndarray) -> np.ndarray:
    """Nearest-neighbor 16-QAM detection, inverse of `qam16_modulate`.
    Returns bits shape (..., 4)."""
    scaled = symbols * np.sqrt(10)
    i_lo, i_hi = _pam4_demodulate(np.real(scaled))
    q_lo, q_hi = _pam4_demodulate(np.imag(scaled))
    return np.stack([i_lo, i_hi, q_lo, q_hi], axis=-1)


@dataclass(frozen=True)
class Modulation:
    """A modulation scheme, bundled with its bit width so BER simulation code
    can stay agnostic to which scheme it's running."""

    name: str
    bits_per_symbol: int
    modulate: Callable[[np.ndarray], np.ndarray]
    demodulate: Callable[[np.ndarray], np.ndarray]


QPSK = Modulation("QPSK", 2, qpsk_modulate, qpsk_demodulate)
QAM16 = Modulation("16-QAM", 4, qam16_modulate, qam16_demodulate)
