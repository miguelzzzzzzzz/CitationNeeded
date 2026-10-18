"""Wilson score intervals for binomial proportions.

For ``successes`` out of ``n`` trials the two-sided interval at confidence
level ``c`` is::

    p = successes / n
    z = Phi^-1(1 - (1 - c) / 2)
    center = (p + z^2 / (2n)) / (1 + z^2 / n)
    half   = z * sqrt(p (1 - p) / n + z^2 / (4 n^2)) / (1 + z^2 / n)

with ``Phi^-1`` the standard-normal quantile function.  Wilson is preferred
over the normal approximation because answer-eval denominators are small and
frequently saturate at 0/n or n/n, where the normal interval collapses to zero
width or escapes [0, 1]; Wilson stays inside the unit interval and keeps a
sensible width in both regimes.
"""

from __future__ import annotations

import math
import statistics

__all__ = ["wilson_interval"]


def wilson_interval(
    successes: int,
    n: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return the Wilson score interval ``(low, high)`` for ``successes / n``.

    Args:
        successes: Number of observed successes.
        n: Number of trials.
        confidence: Two-sided coverage level in the open interval ``(0, 1)``.

    Returns:
        Bounds clamped to ``[0, 1]``.

    Raises:
        TypeError: If ``successes`` or ``n`` is a ``bool``, or if either is not
            an ``int``, or if ``confidence`` is not a real number.
        ValueError: If ``n < 1``, ``successes`` is outside ``[0, n]``, or
            ``confidence`` is not strictly between 0 and 1.
    """
    if isinstance(successes, bool) or isinstance(n, bool):
        raise TypeError("successes and n must be ints, not bools")
    if not isinstance(successes, int) or not isinstance(n, int):
        raise TypeError("successes and n must be ints")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if successes < 0 or successes > n:
        raise ValueError(f"successes must be between 0 and n={n}, got {successes}")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise TypeError("confidence must be a real number")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")

    z = statistics.NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p = successes / n
    z_squared = z * z
    denominator = 1.0 + z_squared / n
    center = (p + z_squared / (2.0 * n)) / denominator
    half_width = z * math.sqrt(p * (1.0 - p) / n + z_squared / (4.0 * n * n)) / denominator

    low = max(0.0, center - half_width)
    high = min(1.0, center + half_width)
    # At p == 0 / p == 1 the exact Wilson bound touches the edge, so snap it
    # against floating-point noise in the general clamp above.
    if successes == 0:
        low = 0.0
    if successes == n:
        high = 1.0
    return low, high
