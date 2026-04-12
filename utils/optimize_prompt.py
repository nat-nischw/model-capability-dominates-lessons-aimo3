"""
GEPA Prompt Optimization for AIMO-3 experiments.
Output format is LOCKED. GEPA only optimizes reasoning instructions.

The seed prompt is loaded from `utils/system_prompts.py` and selected by
experiment id (any key in `SYSTEM_PROMPTS`, e.g. 'baseline', 'E1', ..., 'BF1').

Run from the repo root:
    python -m utils.optimize_prompt                # defaults to BF1
    python -m utils.optimize_prompt --experiment EF1
"""
import argparse
import os

from dotenv import load_dotenv
load_dotenv()  # reads .env
os.environ["ANTHROPIC_API_KEY"] = os.environ.get("CLAUDE_API_KEY", "")

from utils.system_prompts import SYSTEM_PROMPTS 

import litellm 
litellm.request_timeout = 600  # 10 min timeout
litellm.num_retries = 2

import gepa 

# ── Fixed output format (LOCKED — GEPA cannot modify this)
LOCKED_OUTPUT_FORMAT = (
    '\n\n# Output Format:\n'
    'The final answer must be a non-negative integer between 0 and 99999.\n'
    'Put your final answer in the format \'### <answer>\'\n'
    'Also place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'
    'Think step-by-step and show your complete reasoning process. Quality of reasoning '
    'is as important as the final answer.'
)

# ── Monkey-patch: append locked output format before sending to API
from gepa.adapters.default_adapter.default_adapter import DefaultAdapter 
from gepa.core.adapter import EvaluationBatch 


def _safe_evaluate(self, batch, candidate, capture_traces=False):
    """Evaluate with locked output format appended to whatever GEPA generates."""
    system_content = next(iter(candidate.values())) + LOCKED_OUTPUT_FORMAT

    litellm_requests = []
    for data in batch:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{data['input']}"},
        ]
        litellm_requests.append(messages)

    raw_responses = self.litellm.batch_completion(
        model=self.model,
        messages=litellm_requests,
        max_workers=self.max_litellm_workers,
        **self.litellm_batch_completion_kwargs,
    )

    # Handle timeout: replace with empty string (scored as wrong)
    responses = []
    for resp in raw_responses:
        try:
            responses.append(resp.choices[0].message.content.strip())
        except (AttributeError, IndexError, TypeError):
            responses.append("")

    outputs = []
    scores = []
    objective_scores = []
    trajectories = [] if capture_traces else None

    for data, assistant_response in zip(batch, responses, strict=True):
        eval_result = self.evaluator(data, assistant_response)
        outputs.append({"full_assistant_response": assistant_response})
        scores.append(eval_result.score)
        objective_scores.append(eval_result.objective_scores)
        if trajectories is not None:
            trajectories.append({
                "data": data,
                "full_assistant_response": assistant_response,
                "feedback": eval_result.feedback,
            })

    obj_arg = None
    if objective_scores and all(x is not None for x in objective_scores):
        obj_arg = objective_scores

    return EvaluationBatch(
        outputs=outputs,
        scores=scores,
        trajectories=trajectories,
        objective_scores=obj_arg,
    )


DefaultAdapter.evaluate = _safe_evaluate
print("Patched: output format LOCKED, GEPA optimizes reasoning only")


def _strip_locked_output_format(prompt: str) -> str:
    """Remove the '# Output Format:' trailer from a notebook prompt.

    The notebook prompts embed the output format inside the system prompt, but
    GEPA must not optimize it (it is appended at evaluation time). Strip it so
    the seed candidate matches what GEPA will see.
    """
    marker = "# Output Format:"
    idx = prompt.find(marker)
    return prompt[:idx].rstrip() if idx != -1 else prompt.rstrip()


def main():
    parser = argparse.ArgumentParser(description="GEPA prompt optimization for AIMO-3.")
    parser.add_argument(
        "--experiment",
        "-e",
        default="BF1",
        choices=sorted(SYSTEM_PROMPTS.keys()),
        help="Experiment id whose system prompt to use as the GEPA seed.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="GEPA run output directory (default: ./gepa_run_<experiment>).",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    # Dataset
    trainset, valset, testset = gepa.examples.aime.init_dataset()

    # Add AIME3 competition problems from data/reference.csv
    # Source: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/data?select=reference.csv
    import csv
    reference_csv = "./data/reference.csv"
    aime3_by_id = {}
    with open(reference_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            aime3_by_id[row["id"]] = {
                "input": row["problem"],
                "answer": f"### {row['answer']}",
                "additional_context": {"solution": ""},
            }
    print(f"Loaded {len(aime3_by_id)} AIME3 problems from {reference_csv}")

    # Hard problems (lowest p̂, frequently wrong) -> test
    hard_ids = ["641659", "dd7f5e", "424e18", "0e644e", "26de63"]
    # Easier problems -> replace 5 val entries
    easy_ids = ["42d360", "92ba6a", "9c1c5f", "a295e9", "86e8e5"]

    aime3_hard = [aime3_by_id[i] for i in hard_ids]
    aime3_easy = [aime3_by_id[i] for i in easy_ids]

    valset = valset[:-5] + aime3_easy
    testset = testset[:-5] + aime3_hard
    print(f"Train: {len(trainset)}, Val: {len(valset)} (5 swapped), Test: {len(testset)} (5 swapped)")

    # Seed prompt selected by experiment id (strip locked output format)
    seed_prompt = {
        "system_prompt": _strip_locked_output_format(SYSTEM_PROMPTS[args.experiment]),
    }
    print(f"Seed prompt: experiment={args.experiment} ({len(seed_prompt['system_prompt'])} chars)")

    run_dir = args.run_dir or f"./gepa_run_{args.experiment}"

    # Optimize
    result = gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        task_lm="anthropic/claude-sonnet-4-6",
        reflection_lm="anthropic/claude-opus-4-6",
        max_metric_calls=args.max_metric_calls,
        seed=42,
        run_dir=run_dir,
        display_progress_bar=True,
        raise_on_exception=False,
    )

    # Results
    optimized_reasoning = result.best_candidate["system_prompt"]
    full_prompt = optimized_reasoning + LOCKED_OUTPUT_FORMAT

    print("\n" + "=" * 80)
    print(f"OPTIMIZED {args.experiment} PROMPT (output format locked)")
    print("=" * 80)
    print(full_prompt)
    print("=" * 80)

    out_path = f"./optimized_{args.experiment}_prompt.txt"
    with open(out_path, "w") as f:
        f.write(full_prompt)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
