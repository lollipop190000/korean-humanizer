"""Finite LoRA pilot inside an authorized GPU environment. Default: offline dry-run.

This script does not create/terminate Pods. Provisioning, budget and artifact retrieval
must be handled by the job controller. An --execute flag is not spending authorization.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from korean_humanizer.io import sha256_file, write_json
from korean_humanizer.sft import load_sft


def pad_batch(rows, pad_id):
    length = max(len(r["input_ids"]) for r in rows)
    return {
        key: [r[key] + [fill] * (length - len(r[key])) for r in rows]
        for key, fill in [("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)]
    }


def train(bundle, output, manifest_sha, resume=None, stop_after=None):
    manifest, config, splits = load_sft(bundle)
    if sha256_file(bundle / "manifest.json") != manifest_sha:
        raise ValueError("exact reviewed manifest hash is required")
    if resume:
        if (
            not resume.resolve().is_relative_to(output.resolve())
            or not (resume / "trainer_state.json").is_file()
        ):
            raise ValueError("resume needs a checkpoint inside this output directory")
        previous = json.loads((output / "run-info.json").read_text())
        if previous["manifest_sha256"] != manifest_sha:
            raise ValueError("resume bundle changed")
    elif output.exists():
        raise ValueError("refusing to overwrite an existing run")
    if stop_after is not None and not 1 <= stop_after < config.max_steps:
        raise ValueError("smoke stop must be before the planned final step")
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA GPU required")
    if not torch.__version__.startswith("2.6.0") or transformers.__version__ != "4.51.3":
        raise RuntimeError("GPU runtime differs from the candidate pinned environment")
    output.mkdir(parents=True, exist_ok=bool(resume), mode=0o700)
    if not resume:
        write_json(
            output / "run-info.json",
            {
                "manifest_sha256": manifest_sha,
                "model_id": config.model_id,
                "revision": config.revision,
            },
        )
    (output / "runtime.freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        bundle, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("verified tokenizer needs a padding token")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        use_safetensors=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def parameter_digest():
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().float().numpy().tobytes())
        return digest.hexdigest()

    before_parameters = parameter_digest() if not resume else None

    class Examples(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return {k: self.rows[index][k] for k in ["input_ids", "attention_mask", "labels"]}

    class SmokeStop(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = output / ("checkpoint-" + str(state.global_step))
            write_json(
                checkpoint / "checkpoint-complete.json",
                {
                    "step": state.global_step,
                    "files": {
                        str(p.relative_to(checkpoint)): sha256_file(p)
                        for p in sorted(checkpoint.rglob("*"))
                        if p.is_file() and p.name != "checkpoint-complete.json"
                    },
                },
            )
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if stop_after is not None and state.global_step >= stop_after:
                control.should_training_stop = True
                control.should_save = True
            return control

    def collate(rows):
        return {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in pad_batch(rows, tokenizer.pad_token_id).items()
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output),
            max_steps=config.max_steps,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            lr_scheduler_type="cosine",
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=config.eval_steps,
            save_strategy="steps",
            save_steps=config.save_steps,
            save_total_limit=2,
            logging_steps=1,
            report_to=[],
            seed=config.seed,
            data_seed=config.seed,
            remove_unused_columns=False,
            save_safetensors=True,
            dataloader_num_workers=0,
        ),
        train_dataset=Examples(splits["train"]),
        eval_dataset=Examples(splits["dev"]),
        data_collator=collate,
        callbacks=[SmokeStop()],
    )
    before = trainer.evaluate(metric_key_prefix="before") if not resume else None
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    after = trainer.evaluate(metric_key_prefix="after")
    adapter = output / ("adapter-step-" + str(trainer.state.global_step))
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    report = {
        "training_executed": True,
        "trainable_parameters": trainable_parameters,
        "initial_trainable_sha256": before_parameters,
        "final_trainable_sha256": parameter_digest(),
        "initial_parameters_compared": not bool(resume),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "data_policy": config.data_policy,
        "human_semantic_review_completed": manifest.get("human_semantic_review_completed", True),
        "global_step": trainer.state.global_step,
        "planned_steps": config.max_steps,
        "completed": trainer.state.global_step == config.max_steps,
        "resumed_from": str(resume) if resume else None,
        "before": before,
        "after": after,
        "training_metrics": result.metrics,
        "quality_verdict": "not_assessed",
        "note": "Validation loss is not human naturalness or semantic fidelity.",
    }
    write_json(output / ("result-step-" + str(trainer.state.global_step) + ".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest, config, _ = load_sft(args.bundle)
    if args.execute:
        if not args.output_dir or not args.manifest_sha256:
            parser.error("execution requires --output-dir and reviewed --manifest-sha256")
        result = train(
            args.bundle,
            args.output_dir,
            args.manifest_sha256,
            args.resume_checkpoint,
            args.stop_after_step,
        )
    else:
        result = {
            "training_executed": False,
            "network_calls": 0,
            "config": config.model_dump(),
            "data_policy": config.data_policy,
            "human_semantic_review_completed": manifest.get(
                "human_semantic_review_completed", True
            ),
            "examples": manifest["counts"],
            "manifest_sha256": sha256_file(args.bundle / "manifest.json"),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
