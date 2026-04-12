# [Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3]((https://arxiv.org/abs/2603.27844))

A single model capability metric, per-attempt accuracy *p*, determines majority-vote performance. Diverse reasoning strategies, temperature tuning, and prompt engineering do not help. Across 23+ experiments and 4 model families, none reliably improved on the baseline.

## Summary

gpt-oss-120b (5.1B active parameters, MoE) with N=8 attempts at T=1.0 and entropy-weighted voting scores **39.3/50** on average across **21 runs** (best: 42/50, σ=1.7). At equal N=8, gpt-oss-20b scores **31.0/50** (3-run mean: 35, 28, 30). This **8-point gap** is **4× larger** than any prompt optimization (±2 points). Scaling N=8 to N=32 on gpt-oss-20b backfires. Per-attempt time shrinks, p̂ drops from 0.61 to 0.52, and the score drops to 26. Cross-model validation on Qwen3.5-35B-A3B (23/50, N=16) and Nemotron-Super-120B-NVFP4 (23/50, N=3) confirms that score tracks **capability**, not inference-time strategy.

![Model capability dominates](figs/fig3_p_vs_score.png)

## Implementation

The `utils/` package provides method-of-moments estimators, voting analysis, and prompt-optimization tooling used in the paper. Each module is self-contained. Run every command from the repo root.

```bash
# Pairwise error correlation across 19 points / 4 models
python -m utils.correlation

# Binomial majority voting and the 8-point capability gap
python -m utils.voting

# Submission-as-lottery-ticket analysis (42 submissions)
python -m utils.lottery

# GEPA-based prompt optimization (Claude API). Seed prompts live in
# utils/system_prompts.py and are selected by experiment id.
python -m utils.optimize_prompt                              # default: BF1
python -m utils.optimize_prompt --experiment EF1             # any id in SYSTEM_PROMPTS
python -m utils.optimize_prompt -e E4 --max-metric-calls 200 # longer search
python -m utils.optimize_prompt -e BF1 --run-dir ./gepa_out  # custom output dir
```

| Quantity | Definition | Source |
|----------|-----------|--------|
| Pairwise correlation $\hat{\rho}$ | $\hat{\rho} = \dfrac{v_c(v_c-1)\,/\,[N(N-1)] - \hat{p}^{2}}{\hat{p}(1-\hat{p})}$ | `correlation.py` |
| Effective sample size $N_{\text{eff}}$ | $N_{\text{eff}} = \dfrac{N}{1 + (N-1)\,\rho}$ | `correlation.py` |
| Expected score | $50 \cdot \Pr\!\left(X \geq \lceil N/2 \rceil\right),\ X \sim \mathrm{Bin}(N, \hat{p})$ | `voting.py` |
| Cumulative hit rate | $1 - (1-p)^{K}$ over $K$ submissions | `lottery.py` |
| Prompt optimization | GEPA reflective evolutionary search | `optimize_prompt.py` |

---

Fork [the notebook on Kaggle](https://www.kaggle.com/code/natnitarach/aimo-3-model-capability-dominates-inference-time), set `EXPERIMENT` to any configuration from the ablation table, and run on a single H100.

**Citation**:
```bibtex
@misc{nitarach2026modelcapabilitydominatesinferencetime,
      title={Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3},
      author={Natapong Nitarach},
      year={2026},
      eprint={2603.27844},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.27844},
}
```
