"""
LoRA Fine-Tuning Recipe for AbArk LLM Fine-Tuning Recipes.
Efficient parameter-efficient fine-tuning using Low-Rank Adapters.

Supports: instruction following, domain adaptation, classification.
Works on: local GPU, RunPod, AWS SageMaker.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoRARecipeConfig:
    """Configuration for LoRA fine-tuning."""
    # Model
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    output_dir: str = "./outputs/lora_model"
    task: str = "instruction_following"  # instruction_following | classification | domain_adaptation

    # Data
    dataset_path: str = "data/train.jsonl"
    val_ratio: float = 0.05
    max_seq_length: int = 2048
    seed: int = 42

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.01

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Efficiency
    packing: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True

    # Logging
    use_wandb: bool = False
    wandb_project: str = "abark-finetune"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200


class LoRARecipe:
    """
    LoRA fine-tuning recipe.
    Memory-efficient, fast, good for single-GPU training.

    Usage:
        config = LoRARecipeConfig(base_model="Qwen/Qwen2.5-7B-Instruct")
        recipe = LoRARecipe(config)
        recipe.run()
    """

    def __init__(self, config: LoRARecipeConfig):
        self.config = config

    def run(self):
        logger.info("=" * 65)
        logger.info("  AbArk LLM Fine-Tuning Recipes — LoRA")
        logger.info("=" * 65)
        logger.info(f"  Model:   {self.config.base_model}")
        logger.info(f"  Task:    {self.config.task}")
        logger.info(f"  LoRA r:  {self.config.lora_r}")
        logger.info(f"  Data:    {self.config.dataset_path}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from trl import SFTTrainer, SFTConfig
            from peft import LoraConfig, TaskType, get_peft_model
            from datasets import Dataset, load_dataset
        except ImportError as e:
            raise ImportError(f"Missing: {e}\nInstall: pip install trl peft transformers datasets accelerate")

        # ── Step 1: Load Dataset ───────────────────────────────────────────
        logger.info("[1/5] Loading dataset...")
        if self.config.dataset_path.endswith(".jsonl"):
            import json
            rows = []
            with open(self.config.dataset_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            dataset = Dataset.from_list(rows)
        else:
            dataset = load_dataset(self.config.dataset_path, split="train")

        split = dataset.train_test_split(test_size=self.config.val_ratio, seed=self.config.seed)
        logger.info(f"  Train: {len(split['train'])} | Val: {len(split['test'])}")

        # Log 3 sample rows
        for i, row in enumerate(split["train"].select(range(min(3, len(split["train"]))))):
            keys = list(row.keys())
            logger.debug(f"  Sample {i+1}: keys={keys}, preview={str(row)[:120]}")

        # ── Step 2: Load Tokenizer ─────────────────────────────────────────
        logger.info("[2/5] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            use_fast=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        logger.info(f"  Vocab size: {tokenizer.vocab_size} | pad_token: {tokenizer.pad_token}")

        # ── Step 3: Load Model + LoRA ──────────────────────────────────────
        logger.info("[3/5] Loading model and applying LoRA...")
        attn_impl = "flash_attention_2" if self.config.use_flash_attention else "eager"
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        model.config.use_cache = False
        if self.config.gradient_checkpointing:
            model.enable_input_require_grads()

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # ── Step 4: Training Args ──────────────────────────────────────────
        logger.info("[4/5] Configuring training arguments...")
        sft_config = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler,
            weight_decay=self.config.weight_decay,
            max_seq_length=self.config.max_seq_length,
            packing=self.config.packing,
            gradient_checkpointing=self.config.gradient_checkpointing,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            report_to="wandb" if self.config.use_wandb else "none",
            seed=self.config.seed,
            bf16=torch.cuda.is_available(),
            load_best_model_at_end=True,
        )

        # ── Step 5: Train ──────────────────────────────────────────────────
        logger.info("[5/5] Starting LoRA training...")
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            tokenizer=tokenizer,
        )
        train_result = trainer.train()
        logger.info(f"  Training complete. Loss: {train_result.training_loss:.4f}")

        # Save
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        logger.info(f"  Model saved to: {self.config.output_dir}")

        # Eval metrics
        if trainer.eval_dataset:
            metrics = trainer.evaluate()
            logger.info(f"  Eval metrics: {metrics}")

        return train_result
