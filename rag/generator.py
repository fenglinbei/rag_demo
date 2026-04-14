from __future__ import annotations

import importlib.util
import logging
from typing import Final

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import AppConfig
from .prompts import SYSTEM_PROMPT


LOGGER = logging.getLogger(__name__)


class LocalChatGenerator:
    """A lightweight local generator based on Hugging Face Transformers.

    The default model is Qwen/Qwen2.5-3B-Instruct. When CUDA is available and
    bitsandbytes is installed, the model is loaded in 4-bit mode by default.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.device = config.device
        self.input_device = torch.device("cuda:0" if self.device == "cuda" and torch.cuda.is_available() else "cpu")
        LOGGER.info("加载生成模型 tokenizer: %s", config.generator_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.generator_model_name,
            trust_remote_code=True,
            cache_dir=str(config.cache_dir),
        )
        LOGGER.info("加载生成模型权重: %s", config.generator_model_name)
        self.model = self._load_model()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
            )
        else:
            fallback_text = SYSTEM_PROMPT + "\n\n" + user_prompt
            inputs = self.tokenizer(fallback_text, return_tensors="pt").input_ids

        inputs = inputs.to(self.input_device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=max(self.config.temperature, 1e-5),
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _load_model(self):
        common_kwargs = {
            "trust_remote_code": True,
            "cache_dir": str(self.config.cache_dir),
            "low_cpu_mem_usage": True,
        }

        if self.device == "cuda" and torch.cuda.is_available():
            common_kwargs["device_map"] = "auto"
            if self.config.use_4bit and importlib.util.find_spec("bitsandbytes") is not None:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                common_kwargs["quantization_config"] = quantization_config
                common_kwargs["torch_dtype"] = torch.float16
            else:
                common_kwargs["torch_dtype"] = torch.float16
        else:
            common_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            self.config.generator_model_name,
            **common_kwargs,
        )
        if self.device != "cuda" or not torch.cuda.is_available():
            model.to(self.input_device)
        model.eval()
        return model
