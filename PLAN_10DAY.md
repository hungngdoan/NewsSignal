# NewsSignal: 10-Day Execution Plan (Claude Code)

Goal: a running public product in 10 days. Two tracks (DistilBERT baseline on HF Spaces CPU, Qwen 2.5 7B QLoRA on Modal A10G), live GDELT input, honest benchmark, recruiter-ready README.

---

## 0. Definition of done

- Public HF Space loads on a phone in incognito, no login
- DistilBERT inference under 5 seconds on free CPU
- Qwen via Modal: cold start under 60s, warm under 10s
- Side-by-side output: sentiment + confidence (baseline) vs structured JSON (Qwen)
- Live mode: pull recent GDELT headlines and analyze them
- results.md benchmark table: accuracy, macro-F1, JSON validity rate, latency, cost per 1k inferences
- README passes the 3-minute recruiter test: what it does, working demo link, honest numbers, architecture diagram
- CI green (ruff + pytest), no secrets committed, monthly cost at or under $10

## 1. Hard assumptions (confirm or the plan is fiction)

- Time budget: 3 to 4 hours per weekday, 6 to 8 hours per weekend day. Roughly 40 to 45 hours total. With 1 to 2 hours per day this is a 20-day plan, not a 10-day plan.
- Hardware: RTX 5070 Ti 16GB (Blackwell sm_120), Windows 11 + WSL2 Ubuntu 24.04, PyTorch nightly cu130.
- Calendar: if starting Wednesday, Days 4 and 5 land on the weekend. Those are the heaviest days (deploy + weak labeling). Keep that alignment.
- Accounts ready by Day 1: GitHub, HuggingFace (write token), Weights and Biases, Modal. No GCP account. Live data comes from the GDELT DOC 2.0 REST API (free, no auth).
- The only real spend: roughly $5 of frontier API calls for weak labeling (Day 5, one time) plus Modal usage, which stays inside Modal's free monthly credits at demo traffic. Confirm the current credit amount at signup.
- Execution mode: Claude Code writes the code. You run every verification gate yourself. You never accept "done" without running the command.

## 2. What was cut from the 5-week plan (deliberate, logged)

- mypy strict: downgraded to lenient or skipped. Ruff stays.
- Weak-label dataset: 2,500 samples instead of 5,000 to 10,000.
- Week 5 production hardening: reduced to smoke tests, graceful degradation, and a 24h soak.
- Multiple eval suites: reduced to sentiment metrics + JSON validity + field accuracy on a spot-checked set.
- vLLM is preferred but not mandatory: transformers generate on Modal is the sanctioned fallback.

## 3. Critical gates (the plan dies here if ignored)

- Gate A (Day 1): bitsandbytes 4-bit load of Qwen 2.5 7B on Blackwell. Fail = pivot to Qwen 2.5 3B bf16 LoRA the same day. Decision goes in decisions.md.
- Gate B (Day 4): baseline publicly deployed. From Day 4 onward you have a running product no matter what happens to the LLM track. This is the insurance policy.
- Gate C (Day 7): benchmark table exists with real numbers traced to committed eval outputs or W&B run URLs. No number enters results.md without a source artifact.

---

## Day 1 (Wed): Scaffold + the Blackwell gate

Objective: repo live, environment verified, quantization decision made.

Claude Code tasks:
- [ ] Initialize repo structure: src/newsignal/{data,models,training,eval,serving}, tests/, scripts/, notebooks/, data/{raw,processed,weak_labeling} (raw and processed gitignored)
- [ ] pyproject.toml via uv; install: transformers, datasets, peft, trl, accelerate, bitsandbytes, gradio, pandas, polars, wandb, huggingface_hub; dev: ruff, pytest, pre-commit
- [ ] .gitignore (Python, ML artifacts, secrets, data), MIT LICENSE, README stub, decisions.md, this PLAN_10DAY.md committed
- [ ] CLAUDE.md written (see Section "Claude Code operating rules")
- [ ] Pre-commit: ruff format + ruff check
- [ ] Smoke test the GDELT DOC 2.0 API with one curl (no auth, no GCP); confirm a JSON article list returns. 5 minutes now de-risks Day 9.

Human verification gate (you run these):
```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python - <<'EOF'
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb, device_map="auto")
print("4-bit OK. VRAM MB:", torch.cuda.memory_allocated()/1e6)
EOF
```

Outcomes:
- Loads at roughly 5GB: proceed with 7B QLoRA.
- Import/compile error or CUDA failure: pivot to Qwen 2.5 3B bf16 LoRA today. Log in decisions.md. Do not burn more than 2 hours fighting bitsandbytes.

Exit criteria: repo pushed, CUDA verified, Gate A decision logged, wandb and HF login complete.

## Day 2 (Thu): Data pipeline

Objective: unified, tested, reproducible dataset.

Claude Code tasks:
- [ ] Loaders in src/newsignal/data/loaders.py for: Financial PhraseBank (use the 75%+ annotator-agreement subset), FiQA-2018 (map continuous scores to 3-class; thresholds logged in decisions.md), Twitter Financial News sentiment
- [ ] Unified schema: {text: str, sentiment_label: int, source: str, split: str}; dedupe on normalized text
- [ ] Stratified 70/15/15 split on label; persist Parquet to data/processed/
- [ ] pytest: schema validation, no train/test text overlap (hash check), label distribution within sane bounds, loader round-trip

Human gate:
```bash
uv run pytest tests/ -q
uv run python -m newsignal.data.loaders --summary
```

Exit criteria: tests green, row counts and class balance recorded in decisions.md.

## Day 3 (Fri): DistilBERT train + eval

Objective: baseline locked.

Claude Code tasks:
- [ ] src/newsignal/models/distilbert.py wrapper with explicit config
- [ ] training/train_distilbert.py: distilbert-base-uncased, max_len 256, batch 32, lr 5e-5, 3 epochs, W&B logging
- [ ] eval/eval_sentiment.py: accuracy, macro-F1, per-class F1, confusion matrix; writes eval JSON artifact and updates results.md
- [ ] Push weights to HF Hub: hungngdoan/newsignal-distilbert-sentiment

Human gate:
```bash
uv run python -m newsignal.training.train_distilbert
uv run python -m newsignal.eval.eval_sentiment --model distilbert
cat results.md
```

Acceptance: macro-F1 at or above 0.80 on the held-out test split. If under: one iteration on lr/epochs, then move on regardless and log the number honestly.

Exit criteria: metrics in results.md, traced to a committed eval artifact and a W&B run URL.

## Day 4 (Sat): Deploy baseline publicly (Gate B)

Objective: a stranger can use the product today.

Claude Code tasks:
- [ ] serving/app.py: Gradio app, text input, sentiment + confidence output, 3 example buttons, clean layout
- [ ] Create HF Space (Gradio, free CPU), deploy with the Hub-hosted model
- [ ] Handle edge cases: empty input, 10k-char input, non-English (return graceful message)

Human gate:
- Open the Space URL on your phone in incognito. Time the first inference. Must be under 5 seconds after Space wake.

Exit criteria: public URL works on mobile, link added to README stub.

## Day 5 (Sun): Schema lock + weak labeling

Objective: instruction dataset for Qwen, quality-checked.

Schema (locked, copy into decisions.md):
- sentiment: one of positive, negative, neutral, mixed
- event_type: one of earnings_beat, earnings_miss, guidance_change, m_and_a, regulatory_investigation, product_launch, leadership_change, layoffs, partnership, legal_action, macro_policy, other
- topics: list of 1 to 5 lowercase snake_case strings
- entities: list of {name, type} with type in ORG, PERSON, PRODUCT, GPE

Claude Code tasks:
- [ ] Sample ~2,500 texts (blend of the three datasets; optionally add a few hundred fresh headlines via the GDELT DOC API)
- [ ] Labeling script calling a frontier LLM API with strict JSON output, retry on parse failure; prompts saved under data/weak_labeling/prompts/
- [ ] Output JSONL: {input, instruction, output}; train/val split 90/10

Human gate:
- Manually review 75 random labels. Record field-level agreement. Under 85 percent: fix the prompt once, relabel, move on. Do not iterate a third time.

Exit criteria: labeled JSONL committed or Hub-stored, agreement rate logged in decisions.md.

## Day 6 (Mon): Qwen QLoRA training run

Objective: adapter trained overnight.

Config:
- LoRA r=16, alpha=32, dropout 0.05, target modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
- 4-bit NF4, bf16 compute, paged_adamw_8bit, gradient checkpointing
- batch 1 to 2, gradient accumulation to effective 16, seq_len 768, lr 2e-4, 3 epochs (~2,250 train samples)
- Checkpoint every 100 steps. Expect 2 to 4 hours on the 5070 Ti.

Claude Code tasks:
- [ ] models/qwen.py loader with quantization config exposed
- [ ] training/train_qwen_lora.py with the config above, W&B logging
- [ ] Smoke generation script: 5 prompts through the latest checkpoint, print raw JSON

Human gate:
- Launch training in the evening. Before bed, confirm loss is decreasing in W&B and one checkpoint exists. Morning: run the smoke script, eyeball 5 outputs.

Exit criteria: adapter pushed to HF Hub, training curves in W&B.

## Day 7 (Tue): Benchmark (Gate C)

Objective: the honest table.

Claude Code tasks:
- [ ] eval/eval_qwen.py: run test split through the adapter; map mixed to neutral for the 3-class head-to-head (mapping rule logged)
- [ ] Metrics: sentiment accuracy + macro-F1 (both models, same split), JSON validity rate (parse success percent, target 90+), field-level accuracy vs 100 held-out spot-checked labels, per-sample latency local
- [ ] Update results.md benchmark table; collect 5 to 10 contrast examples where the baseline collapses mixed signals and Qwen preserves structure

Human gate:
```bash
uv run python -m newsignal.eval.eval_qwen
git diff results.md
```

Exit criteria: table committed with source artifacts. If Qwen loses on raw sentiment F1, keep it. The story is structure extraction, not leaderboard.

## Day 8 (Wed): Modal endpoint

Objective: GPU serving over HTTPS.

Claude Code tasks:
- [ ] Modal app: load base + adapter (merge optional). Try vLLM first; if the Blackwell-or-image fight exceeds 2 hours, fall back to transformers generate and log it
- [ ] FastAPI endpoint on Modal, bearer-token auth, max concurrency 2, container scaledown after idle
- [ ] Set a hard spend limit in the Modal dashboard ($15)

Human gate:
```bash
curl -s -X POST "$MODAL_URL" -H "Authorization: Bearer $TOKEN" -d '{"text":"Boeing delivered more jets than expected, but the FAA opened a probe into 737 MAX production."}'
```
- Measure cold and warm latency. Cold under 60s, warm under 10s. Log both.

Exit criteria: valid JSON over HTTPS from your laptop, latencies in decisions.md.

## Day 9 (Thu): Wire it together + live data

Objective: the full product, end to end, public.

Claude Code tasks:
- [ ] Space UI v2: side-by-side panels (baseline left, Qwen right), example picker, "fetch live headlines" button
- [ ] Live mode: GDELT DOC 2.0 REST API (free, no auth): last 24h English market/business news, dedupe on domain + normalized title, top 10. Cache the last successful pull for 15 minutes to respect rate limits
- [ ] Space calls Modal over HTTPS with 75s timeout; on failure or cold start, degrade gracefully: show baseline result + "GPU warming" message
- [ ] Secrets via HF Space secrets; nothing in git
- [ ] Edge cases re-tested through the full path

Human gate:
- Phone, incognito: paste text, run both models. Press live mode. Kill nothing, fake nothing.

Exit criteria: full product public and working.

## Day 10 (Fri): Polish + soak

Objective: pass the recruiter test.

Claude Code tasks:
- [ ] README: one-paragraph pitch, demo link, mermaid architecture diagram, demo GIF, benchmark table, 3 contrast examples, honest limitations, "what I would do next"
- [ ] GitHub Actions CI: ruff + pytest (GPU tests marked and skipped in CI)
- [ ] decisions.md cleanup: every gate outcome dated
- [ ] Final pass against Section 0

Human gate:
- Send the repo link to one friend who is not in ML. If they cannot tell you what it does and show you a working output in 3 minutes, the README failed. Fix it.
- Leave the Space up overnight. Check it in the morning (24h soak).

Exit criteria: Definition of done, every box.

---

## Claude Code operating rules (put in CLAUDE.md)

- Always invoke Python via uv run. Never bare python.
- PyTorch is nightly cu130 for Blackwell sm_120. Never "fix" the environment by downgrading torch.
- No mocked metrics, no placeholder data, no fabricated benchmark numbers. Every metric must trace to a committed eval artifact or a W&B run URL.
- One scoped task per session. Commit per completed task, imperative mood messages.
- Any deviation from PLAN_10DAY.md requires a dated entry in decisions.md before the code change.
- If blocked on environment issues for 30+ minutes, stop and hand back to the human. Environment debugging is where vibe-coding burns days.

## Risk register

| # | Risk | Trigger | Pivot |
| - | ---- | ------- | ----- |
| 1 | bitsandbytes fails on Blackwell | Day 1 gate test | Qwen 2.5 3B bf16 LoRA same day |
| 2 | vLLM image fight on Modal | 2 hours of Day 8 burned | transformers generate, accept slower warm latency |
| 3 | Weak label agreement under 85% | Day 5 spot check | One prompt iteration, then ship and document |
| 4 | GDELT DOC API rate limit or outage | Day 9 live mode flaky | Serve cached last pull; ship 10 bundled sample articles as offline mode |
| 5 | Schedule slip | Any day overruns | Drop in this order: live GDELT mode (use cached samples), CI. Never drop the benchmark or README |
| 6 | Modal cost runaway | Spend alert | Concurrency 1, longer scaledown, cap already at $15 |