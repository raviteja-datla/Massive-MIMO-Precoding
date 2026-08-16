"""Run the MRT/ZF/RZF/MMSE comparison sweeps under two separate channel
models — i.i.d. Rayleigh and Kronecker spatially-correlated fading — and
save each model's plots to its own subfolder of results/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mimo_precoding.channel import generate_channel, generate_correlated_channel
from mimo_precoding.modulation import QAM16
from mimo_precoding.simulate import ChannelFn, ber_sweep, sweep

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SEED = 0
N_TRIALS = 300
BER_N_TRIALS = 50
BER_N_SYMBOLS = 2000
ALL_PRECODERS = ["MRT", "ZF", "RZF", "MMSE"]
KRONECKER_RHO = 0.6

# Tight antenna/user ratio + very low SNR: the one regime where ZF's power
# penalty for exact interference cancellation outweighs MRT's simplicity.
CROSSOVER_M, CROSSOVER_K = 12, 8
CROSSOVER_SNR_VALUES = np.arange(-25, 6, 3)

# Large-scale "many users" regime; bigger matrices, so fewer trials to keep
# runtime reasonable.
LARGE_M = 256
LARGE_K_VALUES = [8, 32, 64, 96, 128, 160, 192, 224, 250]
LARGE_N_TRIALS = 100


def _kronecker_channel(M: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """Matches the ChannelFn signature; fixes rho=KRONECKER_RHO for this run."""
    return generate_correlated_channel(M, K, KRONECKER_RHO, rng)


def plot_sum_rate_vs_snr(results_dir: Path, channel_fn: ChannelFn, model_label: str) -> None:
    snr_values = np.arange(-10, 31, 5)
    results = sweep(
        "snr_db",
        snr_values,
        {"M": 64, "K": 8, "snr_db": 0},
        ALL_PRECODERS,
        N_TRIALS,
        SEED,
        channel_fn=channel_fn,
    )
    plt.figure()
    for name, rates in results.items():
        plt.plot(snr_values, rates, marker="o", label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Sum rate (bits/s/Hz)")
    plt.title(f"Sum rate vs SNR (M=64, K=8) — {model_label}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "sum_rate_vs_snr.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sum_rate_vs_M(results_dir: Path, channel_fn: ChannelFn, model_label: str) -> None:
    M_values = [8, 16, 32, 64, 128, 256]
    results = sweep(
        "M",
        M_values,
        {"M": 0, "K": 8, "snr_db": 10},
        ALL_PRECODERS,
        N_TRIALS,
        SEED,
        channel_fn=channel_fn,
    )
    plt.figure()
    for name, rates in results.items():
        plt.plot(M_values, rates, marker="o", label=name)
    plt.xlabel("Number of BS antennas M")
    plt.ylabel("Sum rate (bits/s/Hz)")
    plt.title(f"Sum rate vs M (K=8, SNR=10 dB) — {model_label}")
    plt.xscale("log", base=2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "sum_rate_vs_M.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sum_rate_vs_K(results_dir: Path, channel_fn: ChannelFn, model_label: str) -> None:
    M = 64
    K_values = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 63]
    results = sweep(
        "K",
        K_values,
        {"M": M, "K": 0, "snr_db": 10},
        ALL_PRECODERS,
        N_TRIALS,
        SEED,
        channel_fn=channel_fn,
    )
    plt.figure()
    for name, rates in results.items():
        plt.plot(K_values, rates, marker="o", label=name)
    plt.xlabel("Number of users K")
    plt.ylabel("Sum rate (bits/s/Hz)")
    plt.title(f"Sum rate vs K (M={M}, SNR=10 dB) — {model_label}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "sum_rate_vs_K.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_rzf_vs_xi(results_dir: Path, channel_fn: ChannelFn, model_label: str) -> None:
    M, K, snr_db = 64, 8, 10.0
    xi_values = np.logspace(-3, 3, 25)
    results = sweep(
        "rzf_xi",
        xi_values,
        {"M": M, "K": K, "snr_db": snr_db, "rzf_xi": 1.0},
        ["RZF"],
        N_TRIALS,
        SEED,
        channel_fn=channel_fn,
    )
    sigma2 = 1.0 / (10 ** (snr_db / 10))
    xi_mmse = K * sigma2 / 1.0

    plt.figure()
    plt.plot(xi_values, results["RZF"], marker="o", label="RZF")
    plt.axvline(xi_mmse, color="red", linestyle="--", label=f"MMSE-optimal ξ={xi_mmse:.3g}")
    plt.xscale("log")
    plt.xlabel("Regularization ξ")
    plt.ylabel("Sum rate (bits/s/Hz)")
    plt.title(f"RZF sum rate vs ξ (M={M}, K={K}, SNR={snr_db:.0f} dB) — {model_label}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "rzf_vs_xi.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_ber_vs_snr(results_dir: Path, channel_fn: ChannelFn, model_label: str) -> None:
    M, K = 64, 8
    snr_values = np.arange(-5, 21, 5)
    results = ber_sweep(
        "snr_db",
        snr_values,
        {"M": M, "K": K, "snr_db": 0},
        ALL_PRECODERS,
        n_trials=BER_N_TRIALS,
        n_symbols=BER_N_SYMBOLS,
        seed=SEED,
        channel_fn=channel_fn,
    )
    plt.figure()
    for name, ber in results.items():
        plt.semilogy(snr_values, np.clip(ber, 1e-6, None), marker="o", label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f"BER vs SNR, QPSK (M={M}, K={K}) — {model_label}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(results_dir / "ber_vs_snr.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_mrt_zf_crossover(results_dir: Path, channel_fn: ChannelFn, model_label: str) -> None:
    """MRT's one real advantage over ZF in sum-rate: at a tight antenna/user
    ratio and low SNR, ZF's power penalty for exact interference
    cancellation costs more than the interference it removes is worth. Never
    shows up against MMSE, which is provably >= MRT everywhere."""
    results = sweep(
        "snr_db",
        CROSSOVER_SNR_VALUES,
        {"M": CROSSOVER_M, "K": CROSSOVER_K, "snr_db": 0},
        ALL_PRECODERS,
        N_TRIALS,
        SEED,
        channel_fn=channel_fn,
    )
    plt.figure()
    for name, rates in results.items():
        plt.plot(CROSSOVER_SNR_VALUES, rates, marker="o", label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Sum rate (bits/s/Hz)")
    plt.title(
        f"MRT vs ZF crossover (M={CROSSOVER_M}, K={CROSSOVER_K}) — {model_label}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "mrt_zf_crossover.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_ber_vs_snr_16qam(results_dir: Path, channel_fn: ChannelFn, model_label: str) -> None:
    M, K = 64, 8
    snr_values = np.arange(0, 26, 5)
    results = ber_sweep(
        "snr_db",
        snr_values,
        {"M": M, "K": K, "snr_db": 0},
        ALL_PRECODERS,
        n_trials=BER_N_TRIALS,
        n_symbols=BER_N_SYMBOLS,
        seed=SEED,
        channel_fn=channel_fn,
        modulation=QAM16,
    )
    plt.figure()
    for name, ber in results.items():
        plt.semilogy(snr_values, np.clip(ber, 1e-6, None), marker="o", label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f"BER vs SNR, 16-QAM (M={M}, K={K}) — {model_label}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(results_dir / "ber_vs_snr_16qam.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sum_rate_vs_K_large(results_dir: Path, channel_fn: ChannelFn, model_label: str) -> None:
    """Same story as plot_sum_rate_vs_K but at a much larger scale (up to
    250 users), closer to how many users a real massive-MIMO cell serves."""
    results = sweep(
        "K",
        LARGE_K_VALUES,
        {"M": LARGE_M, "K": 0, "snr_db": 10},
        ALL_PRECODERS,
        LARGE_N_TRIALS,
        SEED,
        channel_fn=channel_fn,
    )
    plt.figure()
    for name, rates in results.items():
        plt.plot(LARGE_K_VALUES, rates, marker="o", label=name)
    plt.xlabel("Number of users K")
    plt.ylabel("Sum rate (bits/s/Hz)")
    plt.title(f"Sum rate vs K, large scale (M={LARGE_M}, SNR=10 dB) — {model_label}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "sum_rate_vs_K_large.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_all_plots(results_dir: Path, channel_fn: ChannelFn, model_label: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_sum_rate_vs_snr(results_dir, channel_fn, model_label)
    plot_sum_rate_vs_M(results_dir, channel_fn, model_label)
    plot_sum_rate_vs_K(results_dir, channel_fn, model_label)
    plot_rzf_vs_xi(results_dir, channel_fn, model_label)
    plot_ber_vs_snr(results_dir, channel_fn, model_label)
    plot_mrt_zf_crossover(results_dir, channel_fn, model_label)
    plot_ber_vs_snr_16qam(results_dir, channel_fn, model_label)
    plot_sum_rate_vs_K_large(results_dir, channel_fn, model_label)
    print(f"Saved 8 plots to {results_dir}")


def main() -> None:
    run_all_plots(RESULTS_DIR / "iid", generate_channel, "i.i.d. Rayleigh")
    run_all_plots(
        RESULTS_DIR / "correlated",
        _kronecker_channel,
        f"Kronecker ρ={KRONECKER_RHO}",
    )


if __name__ == "__main__":
    main()
