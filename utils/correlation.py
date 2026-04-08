"""
Pairwise error correlation and effective sample size.

From the paper (Section 4):
  rho_hat = [v_c(v_c-1)/(N(N-1)) - p_hat^2] / [p_hat(1 - p_hat)]

where v_c = number of correct votes out of N attempts, p_hat = v_c / N.

When rho > 0, errors are positively correlated (attempts make the same mistakes).
When rho < 0, errors are negatively correlated (attempts are more diverse than
independent coin flips). For N=3, rho_hat = -0.500 is the mathematical minimum
regardless of v_c in {1, 2} — it confirms sign but not magnitude.

Effective sample size (Equation 1):
  N_eff = N / (1 + (N-1) * rho)

At rho=0.3, N=8 gives only 2.6 effective votes.
"""

import math
from typing import Optional


def compute_rho(v_c: int, N: int) -> Optional[float]:
    """
    Method-of-moments estimator for pairwise error correlation.

    Args:
        v_c: Number of correct votes (must be in [1, N-1] for defined rho)
        N:   Total number of attempts (voters)

    Returns:
        rho_hat, or None if undefined (v_c=0, v_c=N, or N<2)

    Examples:
        >>> compute_rho(4, 7)   # gpt-oss-120b on problem dd7f5e
        -0.1666...
        >>> compute_rho(1, 3)   # N=3 forced minimum
        -0.5
        >>> compute_rho(0, 8)   # all wrong -> undefined
        None
        >>> compute_rho(8, 8)   # all correct -> undefined
        None
    """
    if N < 2 or v_c == 0 or v_c == N:
        return None
    p = v_c / N
    numerator = v_c * (v_c - 1) / (N * (N - 1)) - p ** 2
    denominator = p * (1 - p)
    return numerator / denominator


def compute_neff(N: int, rho: float) -> float:
    """
    Effective sample size under correlated voting.

    N_eff = N / (1 + (N-1) * rho)

    Args:
        N:   Number of attempts
        rho: Pairwise error correlation

    Returns:
        Effective number of independent votes.

    Examples:
        >>> compute_neff(8, 0.0)    # independent -> full 8
        8.0
        >>> compute_neff(8, 0.3)    # correlated -> ~2.6
        2.58...
        >>> compute_neff(8, -0.14)  # anti-correlated -> >8 (diversity helps)
        inf or large value when denominator -> 0
    """
    denom = 1 + (N - 1) * rho
    if denom <= 0:
        return float('inf')
    return N / denom


def compute_rho_table(data: list[tuple[str, int, int, bool]]) -> list[dict]:
    """
    Compute rho for a list of (problem_id, N, v_c, correct_answer) tuples.

    Args:
        data: List of (problem_id, N, v_c, is_correct_answer)

    Returns:
        List of dicts with keys: pid, N, v_c, p_hat, rho_hat, correct

    Example:
        >>> data = [
        ...     ('42d360', 7, 6, True),   # Qwen
        ...     ('dd7f5e', 7, 4, True),   # gpt-oss-120b
        ...     ('641659', 3, 1, True),   # Nemotron (forced)
        ... ]
        >>> table = compute_rho_table(data)
        >>> for row in table:
        ...     print(f"{row['pid']}: rho={row['rho_hat']:.3f}")
        42d360: rho=-0.167
        dd7f5e: rho=-0.167
        641659: rho=-0.500
    """
    results = []
    for pid, N, v_c, correct in data:
        rho = compute_rho(v_c, N)
        if rho is not None:
            results.append({
                'pid': pid,
                'N': N,
                'v_c': v_c,
                'p_hat': v_c / N,
                'rho_hat': rho,
                'correct': correct,
            })
    return results


# ── Paper data: all 16 computable points across 4 models ─────────────────────

PAPER_DATA = {
    'Qwen3.5-35B-A3B': [
        ('42d360',  7,  6, True),
        ('a295e9', 11,  6, True),
        ('424e18', 16,  3, False),
        ('dd7f5e', 16,  1, False),
    ],
    'gpt-oss-120b': [
        ('dd7f5e',  7,  4, True),
    ],
    'Nemotron-Super-120B': [
        ('641659', 3, 1, True),
        ('dd7f5e', 3, 1, True),
        ('a295e9', 3, 1, True),
    ],
    'gpt-oss-20b': [
        ('641659', 3, 1, True),
        ('42d360', 3, 2, True),
        ('9c1c5f', 3, 2, True),
        ('dd7f5e', 3, 2, True),
        ('424e18', 3, 2, True),
        ('26de63', 3, 2, True),
        ('0e644e', 3, 2, True),
        ('92ba6a', 3, 2, True),
    ],
}


if __name__ == '__main__':
    import numpy as np

    print("=" * 70)
    print("Pairwise Correlation Analysis — All 4 Models from AIMO 3 Paper")
    print("=" * 70)

    all_rhos = []
    large_n_rhos = []

    for model, data in PAPER_DATA.items():
        table = compute_rho_table(data)
        print(f"\n{model}:")
        for row in table:
            tag = " (N=3 forced)" if row['N'] == 3 else ""
            print(f"  {row['pid']}: N={row['N']:2d}, v_c={row['v_c']:2d}, "
                  f"p̂={row['p_hat']:.3f}, ρ̂={row['rho_hat']:+.3f}{tag}")
            all_rhos.append(row['rho_hat'])
            if row['N'] >= 7:
                large_n_rhos.append(row['rho_hat'])

    print(f"\n{'─' * 70}")
    print(f"All {len(all_rhos)} points:  mean ρ̂ = {np.mean(all_rhos):.4f}")
    print(f"N≥7 ({len(large_n_rhos)} points): mean ρ̂ = {np.mean(large_n_rhos):.4f}")

    print(f"\nNeff at mean ρ̂ (N≥7):")
    rho_mean = np.mean(large_n_rhos)
    for N in [8, 16, 32]:
        neff = compute_neff(N, rho_mean)
        print(f"  N={N:2d} → Neff = {neff:.1f}")
