"""Reusable local inference engine for the trained adapter (no hosted inference API)."""

from pathlib import Path

from korean_humanizer.protection import check_protection


class LocalModel:
    def __init__(self, spec, adapter=None, device="auto", memory_mode="full"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA GPU unavailable; select a GPU runtime or use --device cpu")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Apple MPS unavailable")
        quantized = device == "cuda" and (
            memory_mode == "4bit"
            or (
                memory_mode == "auto"
                and torch.cuda.get_device_properties(0).total_memory < 12 * 1024**3
            )
        )
        if memory_mode == "4bit" and device != "cuda":
            raise ValueError("메모리 절약 모드는 NVIDIA CUDA GPU가 필요합니다.")
        dtype = torch.float32 if device == "cpu" else torch.float16
        if device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        adapter = Path(adapter) if adapter else None
        if adapter and (adapter / "tokenizer_config.json").is_file():
            tokenizer = AutoTokenizer.from_pretrained(
                str(adapter), local_files_only=True, trust_remote_code=False
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                spec["model_id"], revision=spec["revision"], trust_remote_code=False
            )
        options = {}
        if quantized:
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            spec["model_id"],
            revision=spec["revision"],
            torch_dtype=dtype,
            device_map={"": device},
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation="sdpa",
            **options,
        ).eval()
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, str(adapter), is_trainable=False, local_files_only=True
            ).eval()
        self.model, self.tokenizer, self.torch = model, tokenizer, torch
        self.spec, self.adapter, self.device = spec, adapter, device
        self.dtype, self.quantized = dtype, quantized

    def rewrite(self, text, max_new_tokens=512):
        if not text.strip():
            raise ValueError("다듬을 글을 입력해 주세요.")
        if not 1 <= max_new_tokens <= 1024:
            raise ValueError("출력 길이 범위가 올바르지 않습니다.")
        tokenizer, model, torch = self.tokenizer, self.model, self.torch
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.spec["instruction"]},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        count = inputs.input_ids.shape[-1]
        if count > 2048:
            raise ValueError("입력이 너무 깁니다. 짧은 문단으로 나눠 주세요. (최대 2,048토큰)")
        with torch.inference_mode():
            generated = model.generate(
                **inputs.to(self.device),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        tokens = generated[0, count:]
        output = tokenizer.decode(tokens, skip_special_tokens=True)
        eos = model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        return {
            "candidate": "local-adapter" if self.adapter else "qwen3-4b-base",
            "base_revision": self.spec["revision"],
            "device": self.device,
            "dtype": str(self.dtype),
            "quantization": "nf4" if self.quantized else "none",
            "output": output,
            "finish_reason": "stop" if len(tokens) and int(tokens[-1]) in eos_ids else "length",
            "checks": check_protection(text, output),
            "human_review_completed": False,
        }
