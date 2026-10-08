"""Agreement between two labellers: Cohen's kappa over the points both confirmed (current text only)."""
from __future__ import annotations

from collections import Counter


def cohens_kappa(a: dict[str, str], b: dict[str, str]) -> dict:
    """Kappa over the IDs both label maps share; None when it is undefined (no overlap, or no variation)."""
    shared = sorted(set(a) & set(b))
    n = len(shared)
    if not n:
        return {"points": 0, "observed": None, "kappa": None}
    observed = sum(1 for pid in shared if a[pid] == b[pid]) / n
    counts_a, counts_b = Counter(a[pid] for pid in shared), Counter(b[pid] for pid in shared)
    expected = sum(counts_a[label] * counts_b[label] for label in counts_a) / (n * n)
    kappa = None if expected == 1 else round((observed - expected) / (1 - expected), 4)
    return {"points": n, "observed": round(observed, 4), "kappa": kappa}
