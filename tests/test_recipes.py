"""
Tests for AbArk LLM Fine-Tuning Recipes (no GPU needed).
Run with: pytest tests/ -v
"""

import json
import os
import pytest

from recipes.lora.lora_recipe import LoRARecipeConfig, LoRARecipe
from recipes.qlora.qlora_recipe import QLoRARecipeConfig, QLoRARecipe


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_jsonl(tmp_path):
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Question {i}?"},
                {"role": "assistant", "content": f"Answer {i}."},
            ]
        }
        for i in range(30)
    ]
    path = tmp_path / "train.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


# ── Config Tests ───────────────────────────────────────────────────────────────

class TestLoRARecipeConfig:
    def test_defaults(self):
        cfg = LoRARecipeConfig()
        assert cfg.lora_r == 64
        assert cfg.lora_alpha == 128
        assert cfg.packing is True
        assert "q_proj" in cfg.lora_target_modules

    def test_custom_config(self):
        cfg = LoRARecipeConfig(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            lora_r=32,
            learning_rate=1e-4,
            num_train_epochs=5,
        )
        assert cfg.lora_r == 32
        assert cfg.learning_rate == 1e-4
        assert cfg.num_train_epochs == 5

    def test_lora_alpha_default(self):
        cfg = LoRARecipeConfig()
        # lora_alpha should be 2x lora_r by default
        assert cfg.lora_alpha == cfg.lora_r * 2


class TestQLoRARecipeConfig:
    def test_defaults(self):
        cfg = QLoRARecipeConfig()
        assert cfg.load_in_4bit is True
        assert cfg.bnb_4bit_quant_type == "nf4"
        assert cfg.use_nested_quant is True

    def test_custom_quant(self):
        cfg = QLoRARecipeConfig(bnb_4bit_compute_dtype="float16")
        assert cfg.bnb_4bit_compute_dtype == "float16"

    def test_lora_params(self):
        cfg = QLoRARecipeConfig(lora_r=8, lora_alpha=16)
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 16


# ── Recipe Instantiation Tests (no model download) ─────────────────────────────

class TestLoRARecipe:
    def test_recipe_instantiation(self):
        config = LoRARecipeConfig()
        recipe = LoRARecipe(config)
        assert recipe.config.lora_r == 64

    def test_recipe_run_raises_without_deps(self, monkeypatch, sample_jsonl):
        """Without trl/peft installed, run() should raise ImportError with helpful message."""
        import sys
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("torch", "trl", "peft"):
                raise ImportError(f"Mocked missing: {name}")
            return original_import(name, *args, **kwargs)

        config = LoRARecipeConfig(dataset_path=sample_jsonl)
        recipe = LoRARecipe(config)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError) as exc_info:
            recipe.run()
        assert "pip install" in str(exc_info.value).lower() or "Missing" in str(exc_info.value)


class TestQLoRARecipe:
    def test_recipe_instantiation(self):
        config = QLoRARecipeConfig()
        recipe = QLoRARecipe(config)
        assert recipe.config.load_in_4bit is True


# ── YAML Config Tests ──────────────────────────────────────────────────────────

class TestYAMLConfigs:
    def test_lora_yaml_loadable(self):
        import yaml
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "lora_config.yaml")
        if os.path.exists(path):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            assert "base_model" in cfg
            assert "lora_r" in cfg

    def test_qlora_yaml_loadable(self):
        import yaml
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "qlora_config.yaml")
        if os.path.exists(path):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            assert "load_in_4bit" in cfg
