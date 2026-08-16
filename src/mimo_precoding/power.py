import numpy as np


def normalize_power(W_raw: np.ndarray, P: float) -> np.ndarray:
    """Scale W_raw so the total transmit power constraint E[||x||^2] = P holds
    given E[s s^H] = I_K, i.e. beta = sqrt(P) / ||W_raw||_F, W = beta * W_raw.

    Applied to the whole matrix at once (not per-column/per-user), since the
    constraint is defined on total transmit power.
    """
    beta = np.sqrt(P) / np.linalg.norm(W_raw, "fro")
    return beta * W_raw
