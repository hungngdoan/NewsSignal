# results.md — Benchmark Table

Every number here must trace to a committed eval artifact or a W&B run URL.

| Metric | DistilBERT baseline | Qwen 2.5 7B QLoRA | Notes |
|--------|--------------------|--------------------|-------|
| Sentiment accuracy | — | — | test split |
| Sentiment macro-F1 | — | — | test split |
| JSON validity rate | n/a | — | target ≥90% |
| Field-level accuracy | n/a | — | 100 spot-checked |
| Latency (local, p50) | — | — | ms per sample |
| Cold start (Modal) | n/a | — | target <60s |
| Warm latency (Modal) | n/a | — | target <10s |
| Cost per 1k inferences | — | — | USD |

*Populated on Day 3 (DistilBERT) and Day 7 (Qwen).*
