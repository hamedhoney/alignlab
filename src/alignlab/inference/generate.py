
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B"):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> dict:
        messages = [
            {
                "role": "user",
                "content": (
                    "Complete the following Python function. "
                    "Return only valid Python code.\n\n"
                    f"{prompt}"
                ),
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.model.device)

        start = time.perf_counter()

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        elapsed = time.perf_counter() - start

        generated_tokens = output[0][inputs["input_ids"].shape[-1]:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return {
            "response": response,
            "latency_seconds": elapsed,
            "generated_tokens": len(generated_tokens),
        }
