# NewsSignal — Claude Code Operating Rules

## Python execution
Always invoke Python via `uv run`. Never use bare `python` or `python3`.

## PyTorch environment
PyTorch is nightly cu130 for Blackwell sm_120. Never "fix" the environment by downgrading torch.

## Data integrity
No mocked metrics, no placeholder data, no fabricated benchmark numbers.
Every metric must trace to a committed eval artifact or a W&B run URL.

## Session discipline
One scoped task per session. Commit per completed task, imperative mood messages.

## Plan adherence
Any deviation from PLAN_10DAY.md requires a dated entry in decisions.md before the code change.

## Blocked on environment
If blocked on environment issues for 30+ minutes, stop and hand back to the human.
Environment debugging is where vibe-coding burns days.

## Project structure
```
src/newsignal/
  data/       — loaders, preprocessing, schema
  models/     — DistilBERT wrapper, Qwen loader
  training/   — train_distilbert.py, train_qwen_lora.py
  eval/       — eval_sentiment.py, eval_qwen.py
  serving/    — Gradio app, Modal endpoint
tests/
scripts/
notebooks/
data/
  raw/          (gitignored)
  processed/    (gitignored)
  weak_labeling/
    prompts/
```

## Key files
- `PLAN_10DAY.md` — execution plan and gates
- `decisions.md` — all deviations and gate outcomes, dated
- `results.md` — benchmark table (no number without a source artifact)
