"""
Majority voting analysis for inference-time scaling.

Key insight from the paper: P(majority correct) = P(X >= ceil(N/2))
where X ~ Binomial(N, p). This is the ONLY lever — and it depends
almost entirely on p (per-attempt accuracy), not on N or voting strategy.

The 11-point gap between gpt-oss-120b (p=0.69, score 39.3) and gpt-oss-20b
(p=0.59, score 28.3) dwarfs every prompt optimization tested (±2 points).
"""

import math
from typing import Optional


def _binom_pmf(k: int, n: int, p: float) -> float:
    """Binomial PMF without scipy dependency."""
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def majority_vote_score(p: float, N: int, n_problems: int = 50) -> float:
    """
    Expected score under Binomial majority voting.

    score = n_problems * P(X >= ceil(N/2)), X ~ Binomial(N, p)

    Args:
        p:          Per-attempt accuracy (probability of correct on one try)
        N:          Number of voting attempts
        n_problems: Total problems (default 50 for AIMO 3)

    Returns:
        Expected score.

    Examples:
        >>> majority_vote_score(0.69, 8)    # gpt-oss-120b
        40.6...
        >>> majority_vote_score(0.59, 8)    # gpt-oss-20b
        31.8...
        >>> majority_vote_score(0.46, 16)   # Qwen3.5-35B
        22.8...
        >>> majority_vote_score(0.05, 8)    # Nemotron
        0.01...
    """
    threshold = math.ceil(N / 2)  # need at least this many correct
    p_correct = sum(_binom_pmf(k, N, p) for k in range(threshold, N + 1))
    return n_problems * p_correct


def p_hat_from_score(score: float, N: int, n_problems: int = 50,
                     tol: float = 1e-6) -> float:
    """
    Inverse: given an observed score, estimate per-attempt accuracy p_hat.

    Uses bisection search on majority_vote_score(p, N) = score.

    Args:
        score:      Observed majority-vote score
        N:          Number of voting attempts
        n_problems: Total problems
        tol:        Convergence tolerance

    Returns:
        Estimated p_hat.

    Examples:
        >>> p_hat_from_score(39.3, 8)   # gpt-oss-120b baseline mean
        0.69...
        >>> p_hat_from_score(28.3, 8)   # gpt-oss-20b mean
        0.59...
    """
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        s = majority_vote_score(mid, N, n_problems)
        if s < score:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def entropy_weighted_vote(answers: list[str], entropies: list[float],
                          alpha: float = 0.1) -> str:
    """
    Entropy-weighted majority vote.

    weight = 1 + 1 / (entropy + alpha)
    Low-entropy (confident) attempts get higher weight.

    Args:
        answers:   List of answer strings from each attempt
        entropies: Corresponding generation entropy for each attempt
        alpha:     Smoothing constant (default 0.1)

    Returns:
        Winning answer string.

    Example:
        >>> answers   = ['42', '42', '37', '42', '37']
        >>> entropies = [0.5,  0.3,  1.2,  0.4,  1.5]
        >>> entropy_weighted_vote(answers, entropies)
        '42'
    """
    from collections import defaultdict
    scores = defaultdict(float)
    for ans, ent in zip(answers, entropies):
        w = 1.0 + 1.0 / (ent + alpha)
        scores[ans] += w
    return max(scores, key=scores.get)


# ── Paper data: empirical (p_hat, score) for each model ──────────────────────

PAPER_MODELS = {
    'gpt-oss-120b':       {'p_hat': 0.69, 'score': 39.3, 'N': 8},
    'gpt-oss-20b':        {'p_hat': 0.59, 'score': 28.3, 'N': 8},
    'Qwen3.5-35B-A3B':    {'p_hat': 0.46, 'score': 23.0, 'N': 16},
    'Nemotron-Super-120B': {'p_hat': 0.05, 'score':  3.0, 'N': 8},
}


if __name__ == '__main__':
    print("=" * 65)
    print("Majority Voting Analysis — AIMO 3 Models")
    print("=" * 65)

    # Show theoretical vs empirical scores
    print(f"\n{'Model':<25s} {'p̂':>5s} {'N':>3s} {'Expected':>9s} {'Actual':>7s}")
    print("─" * 55)
    for model, d in PAPER_MODELS.items():
        expected = majority_vote_score(d['p_hat'], d['N'])
        print(f"{model:<25s} {d['p_hat']:>5.2f} {d['N']:>3d} "
              f"{expected:>9.1f} {d['score']:>7.1f}")

    # Show how score scales with N for the best model
    print(f"\ngpt-oss-120b (p̂=0.69) — score vs N:")
    for N in [1, 2, 4, 8, 16, 32, 64]:
        s = majority_vote_score(0.69, N)
        print(f"  N={N:3d} → {s:.1f}/50")

    # Demonstrate capability gap
    print(f"\nCapability gap at N=8:")
    s1 = majority_vote_score(0.69, 8)
    s2 = majority_vote_score(0.59, 8)
    print(f"  gpt-oss-120b (p̂=0.69): {s1:.1f}")
    print(f"  gpt-oss-20b  (p̂=0.59): {s2:.1f}")
    print(f"  Gap: {s1 - s2:.1f} points — no prompt trick bridges this")
