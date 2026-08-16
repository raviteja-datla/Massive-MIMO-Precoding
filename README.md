# Massive MIMO Precoding

Simulation and comparison of downlink precoding techniques for Massive MIMO systems: **MRT**, **ZF**, **RZF**, and **MMSE**.

## The problem

Spectrum is finite — you can't create more of it, and squeezing more bits into the same frequency has hard limits. Massive MIMO breaks that ceiling by using space as an extra dimension: a base station with many antennas can serve multiple users on the same frequency at the same time. This only works if each user's signal is shaped correctly, since a base station transmitting to multiple users at once causes each user to receive leakage from everyone else's signal — that leakage is interference.

## What this project covers

**Precoding** is how the base station pre-shapes what it transmits from each antenna so that, at each user's receiver, the desired signal combines constructively and interference from other users is suppressed. Four techniques are implemented and compared here, from simplest to most sophisticated:

- **MRT** (Maximum Ratio Transmission) — points each beam straight at its user's channel, ignoring interference. Works well as antennas increasingly outnumber users, due to favorable propagation.
- **ZF** (Zero Forcing) — explicitly cancels interference between users, at a power cost when antennas don't heavily outnumber users.
- **RZF** (Regularized Zero Forcing) — a tunable middle ground between MRT and ZF.
- **MMSE** (Minimum Mean Square Error) — theoretically optimal, jointly minimizing the effect of noise and interference.

## System model

Single-cell downlink, `M` base-station antennas, `K` single-antenna users. All four precoders are built from one shared regularized-inverse function `H(HᴴH + ξI)⁻¹` — ZF is `ξ=0`, MMSE is `ξ = Kσ²/P` (the theoretically optimal value), and RZF is any other `ξ`, making MRT→ZF→RZF→MMSE a single tunable continuum rather than four unrelated formulas.

Two channel models are implemented, kept as fully separate functions (not blended behind a flag):

- **i.i.d. Rayleigh** — `H ∈ C^{M×K}`, every entry independent `CN(0,1)`.
- **Kronecker spatially correlated** — `H = R^{1/2} H_iid`, with an exponential BS-side correlation matrix `R[i,j] = ρ^|i-j|`. `ρ=0` reduces exactly to the i.i.d. model.

## Results

`scripts/run_simulation.py` runs the same 8-experiment suite under both channel models and writes each to its own subfolder — `results/iid/` and `results/correlated/` — so the two are directly comparable rather than merged into one sweep:

- **Sum-rate vs SNR** — MRT saturates at high SNR (interference-limited) while ZF/RZF/MMSE keep climbing; the saturation ceiling is *lower* under correlation, since correlated channels break the near-orthogonality MRT relies on.
- **Sum-rate vs M** — MRT closes the gap toward ZF/MMSE as antennas grow (favorable propagation); correlation slows this convergence.
- **Sum-rate vs K** — exact ZF collapses as `K→M` (near-singular channel Gram matrix), while RZF/MMSE's regularization holds up and MRT degrades gracefully.
- **Sum-rate vs K, large scale** — the same story at up to 250 users (M=256), closer to a real deployment's load.
- **RZF vs ξ** — sum-rate as a function of the regularization parameter, with the MMSE-optimal `ξ` marked.
- **MRT vs ZF crossover** — at a tight antenna/user ratio (M=12,K=8) and very low SNR, MRT actually beats ZF: ZF's power penalty for exact interference cancellation costs more than the interference is worth when noise, not interference, dominates. Never happens against MMSE, which is provably ≥ MRT everywhere.
- **BER vs SNR, QPSK** — Monte Carlo bit-error rate (genie-aided single-tap equalization per user); MRT floors at a much higher error rate than ZF/RZF/MMSE, and that floor is worse still under correlation.
- **BER vs SNR, 16-QAM** — same simulation with a denser constellation (4 bits/symbol vs QPSK's 2); every precoder's error floor is higher at the same SNR, since 16-QAM's points sit closer together for the same energy budget.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # run the test suite
python scripts/run_simulation.py   # saves plots to results/iid/ and results/correlated/ (gitignored)
```

## Status

✅ v1 complete: MRT/ZF/RZF/MMSE precoders, SINR/sum-rate metrics, Monte Carlo BER (QPSK and 16-QAM), i.i.d. and Kronecker-correlated channel models, Monte Carlo sweeps, and tests.

Planned extensions: imperfect/quantized CSI, multi-cell with pilot contamination, uplink combining.

## License

Licensed under the [MIT License](LICENSE).
