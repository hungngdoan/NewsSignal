"""Gate A (Day 1): 4-bit bitsandbytes load of Qwen2.5-7B-Instruct on Blackwell sm_120.

Pass: model loads at roughly 5GB VRAM and generates tokens.
Fail: pivot per PLAN_10DAY.md Gate A, outcome logged in decisions.md.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    print(
        f"torch {torch.__version__}, cuda={torch.cuda.is_available()}, "
        f"device={torch.cuda.get_device_name(0)}"
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto"
    )
    print(f"4-bit OK. VRAM MB: {torch.cuda.memory_allocated() / 1e6:.0f}")

    # Generate a few tokens so the gate proves sm_120 kernels run, not just that weights fit.
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    msgs = [{"role": "user", "content": "Reply with the single word: ok"}]
    inputs = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    print("generation:", tok.decode(out[0][prompt_len:], skip_special_tokens=True))
    print("GATE A: PASS")


if __name__ == "__main__":
    main()
