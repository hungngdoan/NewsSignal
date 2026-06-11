# decisions.md

All deviations from PLAN_10DAY.md and gate outcomes are logged here, dated.

---

## 2026-06-10 — Day 1: Scaffold

**Gate A (Blackwell 4-bit load):** PENDING — run verification on Windows/WSL2 with RTX 5070 Ti.

Run this on the Windows machine:
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
