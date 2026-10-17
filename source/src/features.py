"""Local rule-table descriptors for elementary cellular automata."""
import numpy as np


def outputs(rule: int) -> np.ndarray:
    if not isinstance(rule, (int, np.integer)) or not 0 <= int(rule) <= 255:
        raise ValueError("rule must be an integer in [0, 255]")
    return np.array([(int(rule) >> i) & 1 for i in range(8)], dtype=np.uint8)


def rule_features(rule: int) -> dict:
    """Descriptors under uniformly weighted 3-bit neighborhoods.

    Influence I_i is the probability that flipping input i changes the output.
    Bit positions are left, center, right in the neighborhood index 4L+2C+R.
    """
    f = outputs(rule)
    lam = float(f.mean())
    influences = []
    for bit in (2, 1, 0):
        changed = 0
        for x in range(8):
            if not (x & (1 << bit)):
                changed += int(f[x] != f[x | (1 << bit)])
        influences.append(changed / 4.0)
    # Sensitivity is the sum of the three coordinate influences.
    return {
        "rule": int(rule), "lambda": lam,
        "influence_left": influences[0],
        "influence_center": influences[1],
        "influence_right": influences[2],
        "sensitivity": float(sum(influences)),
        # Binary output entropy is mathematically determined by lambda; retained
        # for transparency, but should not be used as an independent predictor.
        "output_entropy": float(0.0 if lam in (0.0, 1.0) else
                                 -lam*np.log2(lam) - (1-lam)*np.log2(1-lam)),
    }
