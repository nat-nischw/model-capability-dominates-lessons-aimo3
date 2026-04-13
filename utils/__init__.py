"""
Model Capability Dominates — Utility functions for AIMO 3 analysis.

Core concepts from the paper:
  - Pairwise correlation (rho): measures how correlated errors are across attempts
  - Effective sample size (Neff): how many independent votes you actually get
  - Majority voting: P(correct) under Binomial/correlated voting
  - Lottery ticket: P(hitting target score) over K submissions
  - Verified voting: verifier-aware selection to close the majority-vote gap
"""

from .correlation import compute_rho, compute_rho_table, compute_neff
from .voting import majority_vote_score, p_hat_from_score, entropy_weighted_vote
from .lottery import p_hit_target, cumulative_p_over_k
from .verified_voting import (
    Attempt,
    verified_vote,
    constraint_check,
    code_execute_verify,
    lm_reverify,
    extract_code_blocks,
)
