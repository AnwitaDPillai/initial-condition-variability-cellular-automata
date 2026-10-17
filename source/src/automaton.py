"""NumPy engine for elementary cellular automata (periodic boundary conditions)."""
from __future__ import annotations
import numpy as np


def evolve(rule: int, initial: np.ndarray, steps: int) -> np.ndarray:
    """Return an array of shape (steps + 1, width), including the initial row.

    Wolfram indexing: neighborhood 111, 110, ..., 000 maps to rule bits 7..0.
    Periodic boundaries are used.
    """
    if not isinstance(rule, (int, np.integer)) or not 0 <= int(rule) <= 255:
        raise ValueError("rule must be an integer in [0, 255]")
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    state = np.asarray(initial, dtype=np.uint8)
    if state.ndim != 1 or state.size == 0:
        raise ValueError("initial must be a non-empty 1-D array")
    if not np.isin(state, [0, 1]).all():
        raise ValueError("initial must contain only 0 and 1")
    out = np.empty((int(steps) + 1, state.size), dtype=np.uint8)
    out[0] = state
    for t in range(1, int(steps) + 1):
        left = np.roll(state, 1)
        right = np.roll(state, -1)
        idx = (left << 2) | (state << 1) | right
        state = ((int(rule) >> idx) & 1).astype(np.uint8)
        out[t] = state
    return out


def random_initial(width: int, seed: int, p_one: float = 0.5) -> np.ndarray:
    """Generate a reproducible Bernoulli initial state."""
    if width < 1:
        raise ValueError("width must be positive")
    if not 0.0 <= p_one <= 1.0:
        raise ValueError("p_one must be in [0, 1]")
    return np.random.default_rng(seed).binomial(1, p_one, size=width).astype(np.uint8)
