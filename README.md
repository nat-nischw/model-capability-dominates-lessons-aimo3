# Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3

A single model capability metric — per-attempt accuracy *p* — determines majority-vote performance. Diverse reasoning strategies, temperature tuning, and prompt engineering do not help. Twenty experiments confirm this. None improved on the baseline.

## Summary

gpt-oss-120b (5.1B active parameters, MoE) with N=8 attempts at T=1.0 and entropy-weighted voting scores 39.3/50 on average across 21 runs (best: 42/50, σ=1.7). Cross-model validation on three additional architectures — Qwen3.5-35B-A3B (23/50), gpt-oss-20b (28.3/50), Nemotron-Super-120B (3/50) — confirms that score tracks capability, not inference-time strategy.

![Model capability dominates](figs/fig3_p_vs_score.png)

## Reference Implementation

The `utils/` package provides method-of-moments estimators and voting analysis used in the paper. Each module is self-contained.

```bash
python -m utils.correlation   # Pairwise error correlation across 4 models, 16 problems
python -m utils.voting        # Binomial majority voting and the capability gap
python -m utils.lottery       # Submission-as-lottery-ticket analysis
```

| Quantity | Definition | Source |
|----------|-----------|--------|
| Pairwise correlation ρ̂ | `[v_c(v_c−1) / N(N−1) − p̂²] / [p̂(1−p̂)]` | `correlation.py` |
| Effective sample size | `N_eff = N / [1 + (N−1)ρ]` | `correlation.py` |
| Expected score | `50 · P(X ≥ ⌈N/2⌉)`, X ~ Bin(N, p̂) | `voting.py` |
| Cumulative hit rate | `1 − (1−p)^K` over K submissions | `lottery.py` |

## Reproduction

Fork [the notebook](https://www.kaggle.com/code/natnitarach/aimo-3-model-capability-dominates-inference-time). Set `EXPERIMENT` to any configuration from the ablation table. Run on a single H100.


**Paper**: [arXiv:2603.27844](https://arxiv.org/abs/2603.27844)

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
