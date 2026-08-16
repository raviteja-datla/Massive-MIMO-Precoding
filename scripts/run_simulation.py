"""Run the MRT/ZF/RZF/MMSE comparison sweeps under two separate channel
models — i.i.d. Rayleigh and Kronecker spatially-correlated fading — and
save each model's plots to its own subfolder of results/.
"""

from pathlib import Path
from typing import Callable

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

# Fixed categorical colors, one per precoder, held constant across every plot
# (never matplotlib's auto-cycled defaults).
PRECODER_COLORS = {
    "MRT": "#2a78d6",   # blue
    "ZF": "#eb6834",    # orange
    "RZF": "#1baf7a",   # aqua
    "MMSE": "#eda100",  # yellow
}
CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": CHART_SURFACE,
            "axes.facecolor": CHART_SURFACE,
            "savefig.facecolor": CHART_SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
            "text.color": PRIMARY_INK,
            "axes.labelcolor": SECONDARY_INK,
            "axes.edgecolor": BASELINE,
            "axes.titlecolor": PRIMARY_INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.color": MUTED_INK,
            "ytick.color": MUTED_INK,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.labelcolor": SECONDARY_INK,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "lines.markeredgewidth": 0,
        }
    )


def _plot_lines(
    results_dir: Path,
    filename: str,
    series: dict[str, np.ndarray],
    x,
    xlabel: str,
    ylabel: str,
    title: str,
    xscale: str | None = None,
    yscale: str | None = None,
    extra: Callable[[], None] | None = None,
) -> None:
    plt.figure(figsize=(7, 4.5))
    for name, y in series.items():
        plt.plot(x, y, marker="o", label=name, color=PRECODER_COLORS.get(name, MUTED_INK))
    if extra is not None:
        extra()
    if xscale is not None:
        plt.xscale(xscale)
    if yscale is not None:
        plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both" if yscale == "log" else "major")
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


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
    _plot_lines(
        results_dir,
        "sum_rate_vs_snr.png",
        results,
        snr_values,
        "SNR (dB)",
        "Sum rate (bits/s/Hz)",
        f"Sum rate vs SNR (M=64, K=8) — {model_label}",
    )


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
    _plot_lines(
        results_dir,
        "sum_rate_vs_M.png",
        results,
        M_values,
        "Number of BS antennas M",
        "Sum rate (bits/s/Hz)",
        f"Sum rate vs M (K=8, SNR=10 dB) — {model_label}",
        xscale="log",
    )


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
    _plot_lines(
        results_dir,
        "sum_rate_vs_K.png",
        results,
        K_values,
        "Number of users K",
        "Sum rate (bits/s/Hz)",
        f"Sum rate vs K (M={M}, SNR=10 dB) — {model_label}",
    )


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

    def mark_mmse() -> None:
        plt.axvline(
            xi_mmse,
            color=PRECODER_COLORS["MMSE"],
            linestyle="--",
            linewidth=1.5,
            label=f"MMSE-optimal ξ={xi_mmse:.3g}",
        )

    _plot_lines(
        results_dir,
        "rzf_vs_xi.png",
        results,
        xi_values,
        "Regularization ξ",
        "Sum rate (bits/s/Hz)",
        f"RZF sum rate vs ξ (M={M}, K={K}, SNR={snr_db:.0f} dB) — {model_label}",
        xscale="log",
        extra=mark_mmse,
    )


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
    results = {name: np.clip(ber, 1e-6, None) for name, ber in results.items()}
    _plot_lines(
        results_dir,
        "ber_vs_snr.png",
        results,
        snr_values,
        "SNR (dB)",
        "BER",
        f"BER vs SNR, QPSK (M={M}, K={K}) — {model_label}",
        yscale="log",
    )


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
    _plot_lines(
        results_dir,
        "mrt_zf_crossover.png",
        results,
        CROSSOVER_SNR_VALUES,
        "SNR (dB)",
        "Sum rate (bits/s/Hz)",
        f"MRT vs ZF crossover (M={CROSSOVER_M}, K={CROSSOVER_K}) — {model_label}",
    )


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
    results = {name: np.clip(ber, 1e-6, None) for name, ber in results.items()}
    _plot_lines(
        results_dir,
        "ber_vs_snr_16qam.png",
        results,
        snr_values,
        "SNR (dB)",
        "BER",
        f"BER vs SNR, 16-QAM (M={M}, K={K}) — {model_label}",
        yscale="log",
    )


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
    _plot_lines(
        results_dir,
        "sum_rate_vs_K_large.png",
        results,
        LARGE_K_VALUES,
        "Number of users K",
        "Sum rate (bits/s/Hz)",
        f"Sum rate vs K, large scale (M={LARGE_M}, SNR=10 dB) — {model_label}",
    )


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
    _apply_style()
    run_all_plots(RESULTS_DIR / "iid", generate_channel, "i.i.d. Rayleigh")
    run_all_plots(
        RESULTS_DIR / "correlated",
        _kronecker_channel,
        f"Kronecker ρ={KRONECKER_RHO}",
    )


if __name__ == "__main__":
    main()
