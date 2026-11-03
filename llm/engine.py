import os
import json
import re

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda *args, **kwargs: None

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


class ErgasiaLLMEngine:
    """
    Manages a single 4-bit Qwen2.5-7B-Instruct model in VRAM and hot-swaps between:
      - 'base'       : Prompted RAG baseline (adapters disabled for ablation study)
      - 'generation' : Stage 3 Question + CO-PO + Rubric generation adapter
      - 'grading'    : Stage 5 Rubric-based automated grading adapter
    """
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        gen_adapter_dir: str = "lora_generation_adapter",
        grade_adapter_dir: str = "lora_grading_adapter",
        max_seq_length: int = 2048,
    ):
        # 1. Load the base model with the generation adapter attached first
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=gen_adapter_dir if os.path.exists(gen_adapter_dir) else base_model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="chatml",
            mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
        )

        # Rename default adapter to 'generation' and load 'grading' as a second hot-swappable adapter
        if os.path.exists(gen_adapter_dir) and os.path.exists(grade_adapter_dir):
            # PEFT loads the first adapter as 'default'; also register 'grading'
            self.model.load_adapter(grade_adapter_dir, adapter_name="grading")
            self.has_adapters = True
        else:
            self.has_adapters = False

        FastLanguageModel.for_inference(self.model)
        self.active_mode = "generation"

    def set_mode(self, mode: str):
        """
        mode: 'base' | 'generation' | 'grading'
        """
        if not self.has_adapters:
            return

        if mode == "base":
            self.model.disable_adapter_layers()
        elif mode == "generation":
            self.model.enable_adapter_layers()
            self.model.set_adapter("default")
        elif mode == "grading":
            self.model.enable_adapter_layers()
            self.model.set_adapter("grading")
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'base', 'generation', or 'grading'.")
        self.active_mode = mode

    def generate_json(self, system_prompt: str, user_prompt: str, mode: str = "generation", max_new_tokens: int = 768) -> dict:
        """Runs inference in the requested adapter mode and extracts structured JSON."""
        self.set_mode(mode)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                max_length=None,  # Silences transformers warning
                temperature=0.1,
                do_sample=True,
            )

        raw_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
        return self._parse_json_safely(raw_text)

    @staticmethod
    def _parse_json_safely(raw_text: str) -> dict:
        """Extracts JSON object even if wrapped in markdown code fences."""
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
        try:
            return {"valid_json": True, "data": json.loads(cleaned), "raw": raw_text}
        except json.JSONDecodeError:
            # Fallback: find first '{' and last '}'
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return {"valid_json": True, "data": json.loads(cleaned[start:end+1]), "raw": raw_text}
                except json.JSONDecodeError:
                    pass
            return {"valid_json": False, "data": None, "raw": raw_text}