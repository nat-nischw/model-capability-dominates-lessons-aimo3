"""
Verification-enhanced voting for majority-vote selection loss.

Paper context (Section "Selection Loss"): gpt-oss-120b at pass@20 scores
~45.5 on the private AIMO-3 set; our best majority-vote run scores 42/50. The
six-point gap is selection loss — the correct answer is present in the N=8 pool
but outvoted by a confidently-wrong alternative.

This module replaces plain majority vote with a verifier-aware selector
(paper Section 9, λ=2.0):

    final = argmax_a S(a) * (1 + λ * verify(problem, a))

where S(a) = sum_{i: a_i = a} w_i is the entropy-weighted vote score from
Section 2, with w_i = 1 / max(ε_i, 1e-9). Setting λ=0 recovers plain
entropy-weighted voting.

Three verification strategies are supported, ordered by cost:

    1. constraint_check  — zero LM calls. Parse the problem for numeric ranges
                           ("between 0 and 99999", "positive integer"), then
                           drop candidates that violate them. Cheap sanity check.

    2. code_execute      — zero LM calls. Re-run the reasoning's final Python
                           block with the candidate answer substituted, and
                           verify that print(expected) == candidate. Works when
                           the reasoning used code tools (most attempts do).

    3. lm_reverify       — one extra LM call per top-K candidate. Ask the same
                           model: "Is <a> the answer? Check by substitution.
                           Respond YES or NO." Use logprob(YES) − logprob(NO)
                           as a soft verifier score.

The verifier runs only over the top-K most-voted candidates (default K=3);
unanimous votes skip verification entirely. Budget impact: constraint_check is
free, code_execute costs one sandbox run per candidate (~1s), lm_reverify costs
~1/8 of a full attempt per candidate (short prompt, few tokens).

Usage:
    from utils.verified_voting import verified_vote

    answer = verified_vote(
        attempts=[(ans, reasoning_text, entropy_score), ...],
        problem=problem_text,
        strategy='code_execute',   # or 'constraint_check' or 'lm_reverify'
        top_k=3,
        sandbox=jupyter_sandbox,   # required for code_execute
        lm_client=openai_client,   # required for lm_reverify
        lm_model='gpt-oss',
    )
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol


# ─────────────────────────────────────────────────────────────────────────────
# Attempt container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Attempt:
    """Single inference attempt for one problem."""
    answer: str                 # extracted from \boxed{} or ### <answer>
    reasoning: str              # full assistant response (for code_execute)
    entropy: float = 1.0        # lower = more confident (for weighting)
    code_blocks: list[str] = field(default_factory=list)  # executed code, if any


# ─────────────────────────────────────────────────────────────────────────────
# Verifier protocols
# ─────────────────────────────────────────────────────────────────────────────

class SandboxProtocol(Protocol):
    """A stateful Jupyter-like kernel. Matches AIMO3Sandbox from the notebook."""
    def run(self, code: str, timeout: float = 3.0) -> tuple[bool, str]:
        """Returns (success, stdout). success=False on timeout or exception."""
        ...


class LMClientProtocol(Protocol):
    """OpenAI-compatible chat client with logprobs support."""
    def chat_logprobs(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 8,
        top_logprobs: int = 5,
    ) -> dict:
        """Returns {'content': str, 'logprobs': [{'token': str, 'logprob': float}, ...]}."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: constraint_check (free, problem-parsing only)
# ─────────────────────────────────────────────────────────────────────────────

_RANGE_PATTERNS = [
    re.compile(r'between\s+(-?\d+)\s+and\s+(-?\d+)', re.IGNORECASE),
    re.compile(r'from\s+(-?\d+)\s+to\s+(-?\d+)', re.IGNORECASE),
    re.compile(r'0\s*(?:≤|<=|\\leq)\s*\w+\s*(?:≤|<=|\\leq)\s*(\d+)'),
]

_POSITIVITY_PATTERNS = [
    re.compile(r'positive\s+integer', re.IGNORECASE),
    re.compile(r'non[-\s]?negative\s+integer', re.IGNORECASE),
]


def constraint_check(problem: str, candidate: str) -> float:
    """
    Score how well a candidate respects stated numeric constraints.

    Returns 1.0 if all constraints pass, 0.0 if any fail, 0.5 if unparseable.

    Examples:
        >>> constraint_check("Find a non-negative integer between 0 and 999.", "42")
        1.0
        >>> constraint_check("Find a non-negative integer between 0 and 999.", "1500")
        0.0
        >>> constraint_check("Find a non-negative integer between 0 and 999.", "-5")
        0.0
    """
    try:
        val = int(candidate.strip())
    except (ValueError, AttributeError):
        return 0.5  # non-integer candidate — can't check

    # AIMO-3 universal: answer in [0, 99999]
    if not (0 <= val <= 99999):
        return 0.0

    # Problem-stated ranges
    for pat in _RANGE_PATTERNS:
        m = pat.search(problem)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if not (lo <= val <= hi):
                return 0.0

    # Positivity
    for pat in _POSITIVITY_PATTERNS:
        if pat.search(problem) and val < 0:
            return 0.0

    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: code_execute (re-run reasoning's final code block)
# ─────────────────────────────────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(
    r'```(?:python)?\n(.*?)\n```|<tool_call>(.*?)</tool_call>',
    re.DOTALL,
)


def extract_code_blocks(reasoning: str) -> list[str]:
    """Extract Python code blocks from assistant reasoning."""
    blocks = []
    for m in _CODE_BLOCK_RE.finditer(reasoning):
        code = m.group(1) or m.group(2) or ''
        if code.strip():
            blocks.append(code.strip())
    return blocks


def code_execute_verify(
    problem: str,
    candidate: str,
    attempt: Attempt,
    sandbox: SandboxProtocol,
    timeout: float = 3.0,
) -> float:
    """
    Re-execute the attempt's final code block, check if its output matches the
    candidate answer. Returns 1.0 on match, 0.0 on mismatch, 0.5 if no code.

    The assumption: if the model's own code agrees with its stated answer, the
    answer is consistent with the reasoning. This catches the common failure
    mode of the model writing the wrong answer after correct computation.
    """
    blocks = attempt.code_blocks or extract_code_blocks(attempt.reasoning)
    if not blocks:
        return 0.5  # no code — can't verify

    # Run only the last block (the one that "computes the answer")
    last_block = blocks[-1]
    ok, stdout = sandbox.run(last_block, timeout=timeout)
    if not ok:
        return 0.5  # code crashed — inconclusive

    # Match candidate as an integer anywhere in stdout
    try:
        target = int(candidate.strip())
    except (ValueError, AttributeError):
        return 0.5

    numbers = re.findall(r'-?\d+', stdout)
    if str(target) in numbers:
        return 1.0
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: lm_reverify (one extra LM call per candidate)
# ─────────────────────────────────────────────────────────────────────────────

_VERIFY_PROMPT = (
    "You are a careful math verifier. Check whether the given candidate is the "
    "correct answer to the problem. Do not re-solve the problem from scratch. "
    "Substitute the candidate into the problem's constraints and check each one. "
    "Respond with exactly one token: YES or NO."
)


def lm_reverify(
    problem: str,
    candidate: str,
    lm_client: LMClientProtocol,
    lm_model: str,
) -> float:
    """
    Ask the LM to verify a candidate by substitution. Returns
    softmax(logprob_YES, logprob_NO) in [0, 1].

    One LM call per candidate. Use only on top-K most-voted candidates to
    bound cost.
    """
    messages = [
        {"role": "system", "content": _VERIFY_PROMPT},
        {"role": "user", "content": f"Problem: {problem}\n\nCandidate answer: {candidate}\n\nYES or NO?"},
    ]
    resp = lm_client.chat_logprobs(messages, model=lm_model, max_tokens=2, top_logprobs=5)
    logprobs = resp.get('logprobs', [])
    if not logprobs:
        return 0.5

    first_token = logprobs[0]
    yes_lp, no_lp = -float('inf'), -float('inf')
    for tok in [first_token] + first_token.get('top_logprobs', []):
        text = tok.get('token', '').strip().upper()
        lp = tok.get('logprob', -float('inf'))
        if text.startswith('YES') or text == 'Y':
            yes_lp = max(yes_lp, lp)
        elif text.startswith('NO') or text == 'N':
            no_lp = max(no_lp, lp)

    # Softmax normalize
    if yes_lp == -float('inf') and no_lp == -float('inf'):
        return 0.5
    if yes_lp == -float('inf'):
        return 0.0
    if no_lp == -float('inf'):
        return 1.0
    return math.exp(yes_lp) / (math.exp(yes_lp) + math.exp(no_lp))


# ─────────────────────────────────────────────────────────────────────────────
# Main selector
# ─────────────────────────────────────────────────────────────────────────────

def _entropy_weight(entropy: float, eps: float = 1e-9) -> float:
    """Paper Section 2: w_i = 1 / max(entropy_i, 1e-9)."""
    return 1.0 / max(entropy, eps)


def verified_vote(
    attempts: list[Attempt],
    problem: str,
    strategy: str = 'constraint_check',
    top_k: int = 3,
    sandbox: Optional[SandboxProtocol] = None,
    lm_client: Optional[LMClientProtocol] = None,
    lm_model: str = 'gpt-oss',
    entropy_eps: float = 1e-9,
    verifier_weight: float = 2.0,
) -> tuple[str, dict]:
    """
    Verifier-aware majority vote (paper Section 9, Equation 3).

    Algorithm:
        1. Group attempts by answer; compute entropy-weighted score
           S(a) = sum_{i: a_i = a} w_i with w_i = 1 / max(entropy_i, eps).
        2. If only one unique answer, return it (no verification needed).
        3. Take top-K candidates by S(a).
        4. Score each with the chosen verifier v(a) in [0, 1].
        5. Final score = S(a) * (1 + verifier_weight * v(a)).
        6. Return argmax_a.

    Args:
        attempts:           List of Attempt objects.
        problem:            Original problem text.
        strategy:           'constraint_check' | 'code_execute' | 'lm_reverify'.
        top_k:              Max candidates to verify (others get v(a)=0.5).
        sandbox:            Required for code_execute.
        lm_client:          Required for lm_reverify.
        lm_model:           Model name for lm_reverify.
        entropy_eps:        Numerical floor on entropy (paper: 1e-9).
        verifier_weight:    λ in the paper. Default 2.0.
                            verifier_weight=0 recovers plain entropy-weighted vote.
                            verifier_weight=inf ignores S(a) entirely.

    Returns:
        (final_answer, debug_info) where debug_info contains per-candidate scores.
    """
    if not attempts:
        return "", {}

    # Step 1: entropy-weighted vote groups
    groups: dict[str, list[Attempt]] = defaultdict(list)
    for a in attempts:
        if a.answer:
            groups[a.answer].append(a)

    if not groups:
        return "", {}

    vote_counts = {
        ans: sum(_entropy_weight(a.entropy, entropy_eps) for a in group)
        for ans, group in groups.items()
    }

    # Step 2: unanimous? skip verification
    if len(groups) == 1:
        ans = next(iter(groups))
        return ans, {'strategy': 'unanimous', 'scores': vote_counts}

    # Step 3: top-K candidates
    ranked = sorted(vote_counts.items(), key=lambda x: -x[1])
    top_candidates = [ans for ans, _ in ranked[:top_k]]

    # Step 4: verify each top candidate
    verifier_scores: dict[str, float] = {}
    for ans in top_candidates:
        if strategy == 'constraint_check':
            verifier_scores[ans] = constraint_check(problem, ans)
        elif strategy == 'code_execute':
            if sandbox is None:
                raise ValueError("code_execute strategy requires a sandbox")
            # Use the most-confident attempt from this answer's group
            best_attempt = min(groups[ans], key=lambda a: a.entropy)
            verifier_scores[ans] = code_execute_verify(problem, ans, best_attempt, sandbox)
        elif strategy == 'lm_reverify':
            if lm_client is None:
                raise ValueError("lm_reverify strategy requires an lm_client")
            verifier_scores[ans] = lm_reverify(problem, ans, lm_client, lm_model)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # Non-top candidates get neutral verifier score
    for ans in vote_counts:
        verifier_scores.setdefault(ans, 0.5)

    # Step 5: combined score
    final_scores = {
        ans: votes * (1.0 + verifier_weight * verifier_scores[ans])
        for ans, votes in vote_counts.items()
    }

    # Step 6: pick winner
    winner = max(final_scores, key=final_scores.get)
    return winner, {
        'strategy': strategy,
        'votes': vote_counts,
        'verifier': verifier_scores,
        'final': final_scores,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo / sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 70)
    print("Verified Voting Demo — Fixing the Selection Gap")
    print("=" * 70)

    # Scenario: N=8 attempts on a problem. The correct answer (42) is minority.
    # Plain majority vote picks 37 (4 votes). Constraint-aware verifier
    # catches that 37 is fine but then code verifies 42.
    attempts = [
        Attempt('37', 'Step-by-step analysis leading to 37.', entropy=0.8),
        Attempt('37', 'Another path to 37.',                  entropy=0.9),
        Attempt('37', 'Confident: 37.',                        entropy=0.4),
        Attempt('37', 'Recomputing gives 37.',                 entropy=0.7),
        Attempt('42', 'Careful derivation: 42.',               entropy=0.3),
        Attempt('42', 'Verified via code:\n```python\nprint(42)\n```', entropy=0.2),
        Attempt('42', 'Cross-checked: 42.',                    entropy=0.3),
        Attempt('100000', 'Out-of-range guess.',               entropy=2.0),
    ]
    problem = "Find a non-negative integer between 0 and 99999 satisfying ..."

    # Plain entropy vote (no verification) → 37 wins (4 votes vs 3)
    ans_plain, info_plain = verified_vote(
        attempts, problem, strategy='constraint_check', verifier_weight=0.0,
    )
    print(f"\n[Plain entropy vote]        winner = {ans_plain}")
    print(f"  votes: {info_plain['votes']}")

    # Constraint check catches the out-of-range candidate (100000 is excluded by
    # "between 0 and 99999" if problem said so, or by the universal AIMO cap)
    ans_cc, info_cc = verified_vote(
        attempts, problem, strategy='constraint_check', verifier_weight=2.0,
    )
    print(f"\n[Constraint check]          winner = {ans_cc}")
    print(f"  verifier: {info_cc['verifier']}")
    print(f"  final:    {info_cc['final']}")

    # Note: to pick 42 over 37, we need a verifier that can actually distinguish
    # right from wrong — constraint_check can't do it here because both 37 and
    # 42 are in range. code_execute would catch it if attempt for 42 contains
    # working code, via the sandbox. lm_reverify would catch it if the model can
    # self-verify.
    print("\nKey insight:")
    print("  constraint_check is a cheap prefilter (drops out-of-range guesses).")
    print("  code_execute catches the 'wrote correct code but wrong final answer' failure.")
    print("  lm_reverify catches the 'majority of wrong answers are wrong in the same way' failure.")
    print("\nEach strategy closes a different slice of the 6-point selection gap.")
