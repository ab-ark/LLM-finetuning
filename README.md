# abark-llm-finetuning-recipes

> **LLM Fine-Tuning Recipes by AbArk**
> Production-ready, config-driven fine-tuning scripts for LoRA, QLoRA, and Full Fine-Tuning.
> Supports local GPU, RunPod, and AWS SageMaker deployment.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Recipes

| Recipe | VRAM | Use Case |
|---|---|---|
| **LoRA** | 8–16GB | Single GPU fine-tuning, good quality |
| **QLoRA** | 6–12GB | Consumer GPU fine-tuning, 4-bit quantization |
| **Full FT** | 4× A100+ | Maximum quality, domain specialization |

---

## Quick Start

```bash
pip install -r requirements.txt

# LoRA fine-tuning
python scripts/run_lora.py --config configs/lora_config.yaml

# QLoRA fine-tuning (memory efficient)
python scripts/run_qlora.py --config configs/qlora_config.yaml
```

---

## Dataset Format

All recipes expect JSONL with chat-template messages:

```jsonl
{"messages": [{"role": "user", "content": "What is LangChain?"}, {"role": "assistant", "content": "LangChain is a framework for building LLM apps..."}]}
{"messages": [{"role": "user", "content": "Explain RAG"}, {"role": "assistant", "content": "RAG stands for Retrieval Augmented Generation..."}]}
```

---

## LoRA Recipe

```python
from recipes.lora.lora_recipe import LoRARecipe, LoRARecipeConfig

config = LoRARecipeConfig(
    base_model="meta-llama/Llama-3.2-1B-Instruct",
    dataset_path="data/train.jsonl",
    lora_r=64,
    num_train_epochs=3,
)
recipe = LoRARecipe(config)
recipe.run()
```

---

## QLoRA Recipe (4-bit, consumer GPU)

```python
from recipes.qlora.qlora_recipe import QLoRARecipe, QLoRARecipeConfig

config = QLoRARecipeConfig(
    base_model="mistralai/Mistral-7B-Instruct-v0.2",
    dataset_path="data/train.jsonl",
    lora_r=16,
)
recipe = QLoRARecipe(config)
recipe.run()
```

---

## SageMaker Deployment

```python
from deploy.sagemaker_deploy import SageMakerDeployer

deployer = SageMakerDeployer(role_arn="arn:aws:iam::123:role/SageMakerRole")
s3_uri = deployer.upload_model("./outputs/lora_model", "my-bucket")
endpoint = deployer.deploy(s3_uri, "meta-llama/Llama-3.2-1B-Instruct")
print(deployer.predict(endpoint, "What is machine learning?"))
```

---

## Architecture

```
recipes/
├── lora/
│   └── lora_recipe.py       # LoRA trainer with detailed step logging
├── qlora/
│   └── qlora_recipe.py      # 4-bit QLoRA trainer
└── full_ft/
    └── full_ft_recipe.py    # Full fine-tuning (multi-GPU)
deploy/
└── sagemaker_deploy.py      # Upload + deploy to SageMaker
configs/
├── lora_config.yaml
├── qlora_config.yaml
└── full_ft_config.yaml
scripts/
├── run_lora.py
└── run_qlora.py
```

---

## References & Inspiration

- [HuggingFace TRL](https://github.com/huggingface/trl) — SFTTrainer, DPOTrainer
- [PEFT](https://github.com/huggingface/peft) — LoRA, QLoRA, IA³
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — distributed RLHF

---

## License

MIT © [AbArk](https://github.com/AbArk)
