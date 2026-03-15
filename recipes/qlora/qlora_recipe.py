"""
QLoRA Fine-Tuning Recipe for AbArk LLM Fine-Tuning Recipes.
4-bit quantized LoRA — fine-tune 7B+ models on a single consumer GPU.

Requires: bitsandbytes, peft, trl, transformers
"""

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class QLoRARecipeConfig:
    """Configuration for QLoRA fine-tuning."""
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "./outputs/qlora_model"
    dataset_path: str = "data/train.jsonl"
    val_ratio: float = 0.05
    max_seq_length: int = 1024
    seed: int = 42

    # Training
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"

    # QLoRA specific
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # bfloat16 | float16
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Logging
    use_wandb: bool = False
    wandb_project: str = "abark-finetune"
    logging_steps: int = 10


class QLoRARecipe:
    """
    QLoRA fine-tuning recipe for memory-constrained environments.
    Enables fine-tuning 7B+ models on a single 16GB GPU.

    Usage:
        config = QLoRARecipeConfig(base_model="mistralai/Mistral-7B-Instruct-v0.2")
        recipe = QLoRARecipe(config)
        recipe.run()
    """

    def __init__(self, config: QLoRARecipeConfig):
        self.config = config

    def run(self):
        logger.info("=" * 65)
        logger.info("  AbArk LLM Fine-Tuning Recipes — QLoRA (4-bit)")
        logger.info("=" * 65)
        logger.info(f"  Model:             {self.config.base_model}")
        logger.info(f"  4-bit quant type:  {self.config.bnb_4bit_quant_type}")
        logger.info(f"  LoRA r:            {self.config.lora_r}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from trl import SFTTrainer, SFTConfig
            from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
            from datasets import Dataset
        except ImportError as e:
            raise ImportError(f"Missing: {e}\nInstall: pip install trl peft transformers bitsandbytes datasets accelerate")

        import json

        # ── Load Dataset ───────────────────────────────────────────────────
        logger.info("[1/5] Loading dataset...")
        rows = []
        with open(self.config.dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        dataset = Dataset.from_list(rows)
        split = dataset.train_test_split(test_size=self.config.val_ratio, seed=self.config.seed)
        logger.info(f"  Train: {len(split['train'])} | Val: {len(split['test'])}")

        # ── Load Tokenizer ─────────────────────────────────────────────────
        logger.info("[2/5] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # ── Load Model in 4-bit ────────────────────────────────────────────
        logger.info("[3/5] Loading model in 4-bit quantization...")
        compute_dtype = torch.bfloat16 if self.config.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config.use_nested_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

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

        # ── Training ───────────────────────────────────────────────────────
        logger.info("[4/5] Configuring training arguments...")
        sft_config = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler,
            max_seq_length=self.config.max_seq_length,
            fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            report_to="wandb" if self.config.use_wandb else "none",
            seed=self.config.seed,
            logging_steps=self.config.logging_steps,
        )

        logger.info("[5/5] Starting QLoRA training...")
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            tokenizer=tokenizer,
        )
        result = trainer.train()
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        logger.info(f"QLoRA model saved: {self.config.output_dir}")
        return result
