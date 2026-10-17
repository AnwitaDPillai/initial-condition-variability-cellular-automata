"""Global trajectory metrics. Definitions are deliberately explicit."""
import numpy as np
try:
    from numba import njit
    @njit(cache=True)
    def _lz78_count_numba(a):
        n = len(a)
        child0 = np.full(n + 1, -1, dtype=np.int64)
        child1 = np.full(n + 1, -1, dtype=np.int64)
        node = 0
        count = 0
        nodes = 1
        for i in range(n):
            if a[i] == 0:
                nxt = child0[node]
                if nxt == -1:
                    child0[node] = nodes
                    nodes += 1
                    count += 1
                    node = 0
                else:
                    node = nxt
            else:
                nxt = child1[node]
                if nxt == -1:
                    child1[node] = nodes
                    nodes += 1
                    count += 1
                    node = 0
                else:
                    node = nxt
        if node != 0:
            count += 1
        return count
except Exception:  # optional acceleration; pure Python fallback below
    _lz78_count_numba = None


def lz78_phrase_count(bits) -> int:
    """Count phrases using incremental LZ78 parsing of a binary sequence."""
    a = np.asarray(bits, dtype=np.uint8).ravel()
    if _lz78_count_numba is not None:
        return int(_lz78_count_numba(a))
    dictionary = set()
    phrase = ''
    count = 0
    for bit in a:
        phrase += '1' if bit else '0'
        if phrase not in dictionary:
            dictionary.add(phrase)
            count += 1
            phrase = ''
    if phrase:
        count += 1
    return count


def normalized_lz78(bits) -> float:
    """Finite-length normalized LZ78 phrase count C(n)*log2(n)/n.

    The score is not forced into [0,1]; compare only equal sequence lengths.
    """
    a = np.asarray(bits, dtype=np.uint8).ravel()
    n = a.size
    if n < 2:
        return 0.0
    return float(lz78_phrase_count(a) * np.log2(n) / n)


def trajectory_metrics(spacetime: np.ndarray) -> dict:
    """Compute per-run metrics; excludes the initial row from LZ averaging."""
    grid = np.asarray(spacetime, dtype=np.uint8)
    if grid.ndim != 2 or grid.shape[0] < 2:
        raise ValueError("spacetime must be a 2-D array with at least two rows")
    evolved = grid[1:]
    row_lz = [normalized_lz78(row) for row in evolved]
    densities = evolved.mean(axis=1)
    return {
        "lz_mean_rows": float(np.mean(row_lz)),
        "lz_sd_rows": float(np.std(row_lz, ddof=0)),
        "density_mean": float(np.mean(densities)),
        "density_temporal_sd": float(np.std(densities, ddof=0)),
    }


def damage_curve(rule: int, initial: np.ndarray, steps: int) -> np.ndarray:
    """Hamming damage fraction after flipping one middle cell in the initial row."""
    from .automaton import evolve
    a = np.asarray(initial, dtype=np.uint8).copy()
    b = a.copy()
    b[len(b)//2] ^= 1
    x, y = evolve(rule, a, steps), evolve(rule, b, steps)
    return np.mean(x != y, axis=1)
