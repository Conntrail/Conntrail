"""
Entropy utilities — Shannon entropy over routing outcomes.

routing_entropy(["a","a","a","a"]) → 0.0  (all same, robust)
routing_entropy(["a","b","c","d"]) → 1.0  (all different, fragile)
"""
from __future__ import annotations

import math
from collections import Counter


def routing_entropy(routes: list[str]) -> float:
    """
    Compute normalised Shannon entropy over a list of route labels.

    Returns a value in [0.0, 1.0]:
      0.0 → all 4 variants took the same route (confident decision)
      1.0 → every variant routed differently (maximally fragile)

    Args:
        routes: List of route labels, typically length 4
                (original + similar + neutral + opposite).

    Returns:
        Normalised entropy in [0.0, 1.0].
    """
    n = len(routes)
    if n == 0:
        return 0.0
    counts = Counter(routes)
    raw_entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
    max_entropy = math.log2(n)
    return raw_entropy / max_entropy if max_entropy > 0 else 0.0
