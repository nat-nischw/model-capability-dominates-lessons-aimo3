"""
Lottery ticket analysis for competition submissions.

Each submission is an independent draw. The probability of hitting a
target score at least once over K submissions:

    P(max >= target in K runs) = 1 - (1 - p_single)^K

where p_single = P(score >= target) from a Normal(mu, sigma^2) model
of run-to-run score variance.

Paper finding: with mu=39.3, sigma=1.7 (21-run baseline), each submission
has P(score >= 42) ~= 5.6%. All 42 submissions exhausted; best run = 42.
The winning strategy is simply to submit many times with the strongest
baseline. The Diverse Mixer reduces expected score (mu drops to ~39.0)
without improving tail probability meaningfully, while burning submissions.
"""

import math
from typing import Optional


def _norm_cdf(x: float) -> float:
    """Standard normal CDF without scipy dependency."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def p_hit_target(target: float, mu: float, sigma: float) -> float:
    """
    Probability of a single run scoring >= target.

    Assumes score ~ Normal(mu, sigma^2).

    Args:
        target: Target score to hit
        mu:     Mean score across runs
        sigma:  Standard deviation across runs

    Returns:
        P(score >= target)

    Examples:
        >>> p_hit_target(42, 39.3, 1.7)   # baseline hitting best
        0.056...
        >>> p_hit_target(46, 39.3, 1.7)   # hitting top LB
        0.0001...
    """
    if sigma <= 0:
        return 1.0 if mu >= target else 0.0
    z = (target - mu) / sigma
    return 1 - _norm_cdf(z)


def cumulative_p_over_k(p_single: float, K: int) -> float:
    """
    P(at least one hit in K independent submissions).

    P = 1 - (1 - p_single)^K

    Args:
        p_single: Per-submission probability of hitting target
        K:        Number of submissions

    Returns:
        Cumulative probability.

    Examples:
        >>> p = p_hit_target(42, 39.3, 1.7)
        >>> cumulative_p_over_k(p, 42)   # all 42 submissions exhausted
        0.91...
    """
    return 1 - (1 - p_single) ** K


def submissions_needed(p_single: float, confidence: float = 0.90) -> int:
    """
    How many submissions needed to hit target with given confidence?

    K = ceil(log(1 - confidence) / log(1 - p_single))

    Args:
        p_single:   Per-submission probability
        confidence: Desired cumulative probability (default 0.90)

    Returns:
        Number of submissions needed.

    Example:
        >>> p = p_hit_target(42, 39.3, 1.7)
        >>> submissions_needed(p, 0.90)
        41
    """
    if p_single <= 0:
        return float('inf')
    if p_single >= 1:
        return 1
    return math.ceil(math.log(1 - confidence) / math.log(1 - p_single))


if __name__ == '__main__':
    print("=" * 65)
    print("Lottery Ticket Analysis — AIMO 3 Submission Strategy")
    print("=" * 65)

    # Paper baseline: 21 runs, mu=39.3, sigma=1.7, range 34-42
    # All 42 submissions exhausted (one daily submission allowed)
    mu, sigma = 39.3, 1.7
    used = 42

    for target in [42, 44, 46]:
        p = p_hit_target(target, mu, sigma)
        cum = cumulative_p_over_k(p, used)
        needed_90 = submissions_needed(p, 0.90)
        print(f"\nTarget >= {target}:")
        print(f"  P(single run)      = {p*100:.2f}%")
        print(f"  P(hit in {used} subs) = {cum*100:.1f}%")
        print(f"  Subs for 90% conf  = {needed_90}")

    # Best run reached 42; top AIMO-3 LB is 46+
    print(f"\nBaseline vs Mixer at target=42 (over 42 submissions):")
    p_base = p_hit_target(42, 39.3, 1.7)
    p_mix = p_hit_target(42, 39.0, 2.0)
    print(f"  Baseline: {p_base*100:.1f}%/run -> {cumulative_p_over_k(p_base, 42)*100:.1f}% in 42 subs")
    print(f"  Mixer:    {p_mix*100:.1f}%/run -> {cumulative_p_over_k(p_mix, 42)*100:.1f}% in 42 subs")
    print("  -> Mixer drops expected score; modest tail-prob gain doesn't justify weaker mean.")
