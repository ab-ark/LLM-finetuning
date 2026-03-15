#!/usr/bin/env python
"""Run LoRA fine-tuning. Usage: python scripts/run_lora.py --config configs/lora_config.yaml"""
import argparse, logging, yaml
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
from recipes.lora.lora_recipe import LoRARecipe, LoRARecipeConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_config.yaml")
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--dataset_path", default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.base_model: cfg["base_model"] = args.base_model
    if args.dataset_path: cfg["dataset_path"] = args.dataset_path
    recipe = LoRARecipe(LoRARecipeConfig(**{k: v for k, v in cfg.items() if hasattr(LoRARecipeConfig, k)}))
    recipe.run()

if __name__ == "__main__":
    main()
