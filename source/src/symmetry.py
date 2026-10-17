"""Exact ECA symmetries used to prevent equivalent rules crossing CV folds."""
import numpy as np
from .features import outputs


def transform_rule(rule: int, reflect: bool = False, complement: bool = False) -> int:
    f = outputs(rule)
    out = np.empty(8, dtype=np.uint8)
    for idx in range(8):
        left, center, right = (idx >> 2) & 1, (idx >> 1) & 1, idx & 1
        if complement:
            left, center, right = 1-left, 1-center, 1-right
        if reflect:
            left, right = right, left
        j = 4*left + 2*center + right
        val = int(f[j])
        out[idx] = 1-val if complement else val
    return int(sum(int(out[i]) << i for i in range(8)))


def equivalence_class(rule: int) -> int:
    """Canonical representative under reflection and state complementation."""
    orbit = {transform_rule(rule, r, c) for r in (False, True) for c in (False, True)}
    return min(orbit)


def all_equivalence_classes():
    return {r: equivalence_class(r) for r in range(256)}
