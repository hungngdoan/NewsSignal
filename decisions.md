# decisions.md

All deviations from PLAN_10DAY.md and gate outcomes are logged here, dated.

---

## 2026-06-10 — Day 1: Scaffold

**Gate A (Blackwell 4-bit load):** PASS — 2026-06-10, native Windows (no WSL2, see deviation entry below).

- Command: `uv run python scripts/gate_a_4bit_load.py` (committed; equivalent to the inline test below plus a generation check)
- Result: `4-bit OK. VRAM MB: 5853`, greedy generation returned expected output, `GATE A: PASS`
- Stack: torch 2.12.0+cu130 (stable), bitsandbytes 0.49.2, transformers 5.11.0, Python 3.12.13, driver 596.36 / CUDA 13.2
- **Decision: proceed with Qwen 2.5 7B QLoRA.** No pivot.
- Note: transformers 5.x `apply_chat_template` returns a dict; gate script uses `return_dict=True` + `generate(**inputs)`. Keep this in mind for Day 6/7 scripts.

Original verification block (for reference):
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

Expected: ~5GB VRAM. If load fails → pivot to Qwen 2.5 3B bf16 LoRA, log outcome here.

---

## 2026-06-10 — Day 1: Environment deviation — native Windows instead of WSL2

**Deviation:** PLAN_10DAY.md assumes Windows 11 + WSL2 Ubuntu 24.04. This machine has no WSL installed; installing it requires admin elevation and a reboot. Executing Day 1 natively on Windows 11 instead.

**Setup chosen:**
- uv 0.11.20 (user-level install, no admin)
- Python 3.12 pinned via uv (system Python is 3.14; 3.12 has the widest ML wheel coverage)
- PyTorch from the official cu130 Windows wheel index (stable preferred over nightly now that cu130/sm_120 wheels ship in stable; nightly only if stable fails to resolve). Driver: 596.36, CUDA 13.2, RTX 5070 Ti 16GB confirmed via nvidia-smi.
- bitsandbytes Windows wheels (shipped since 0.43.x)

**Gate A interpretation under this deviation:** if the 4-bit load fails natively, the failure mode must be diagnosed before invoking the 3B pivot. A bitsandbytes-on-native-Windows failure is an environment artifact, not a Blackwell failure; the first remediation is installing WSL2 (human action: `wsl --install`, reboot), not the 3B pivot. The 3B pivot triggers only if 4-bit fails *on a working CUDA torch* for Blackwell-specific reasons.

**GDELT smoke test (Day 1 task):** PASS — DOC 2.0 API returned a JSON article list (5 articles, business/markets, English) with no auth at 2026-06-10 22:09 local.

---
