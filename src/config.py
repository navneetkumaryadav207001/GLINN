"""Configuration dataclasses for chess_nla."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NLAConfig:
    """Model and architecture settings for Chess NLA."""
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    d_model: int = 896
    target_d_model: int = 768
    target_model: str = "SE-ResNet-20 Dual-Head Chess Expert"
    target_extraction_layer: str = "penultimate_dense"

    # Adapter projection settings
    use_adapter_projection: bool = True
    adapter_init_mode: str = "truncated_identity"

    # LoRA settings for Qwen
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Prompt templates
    critic_prompt_template: str = "Analyze the internal chess dynamics: {explanation}"
    verbalizer_prompt_template: str = (
        "Decode the chess position activation vector into a strategic and tactical explanation."
    )
    max_explanation_tokens: int = 96
    vector_prefix_length: int = 1

    @property
    def mse_scale(self) -> float:
        """Target scale sqrt(d_target) for direction-only MSE."""
        return math.sqrt(self.target_d_model)


@dataclass
class DataConfig:
    """Dataset extraction and partitioning settings for Chess NLA."""
    source_file: str = "/home/leo/Codes/GLINN/Data/external/raw_chess_annotated.jsonl"
    weights_path: str = (
        "/home/leo/.cache/huggingface/hub/models--engdarwish--chess-position-evaluator/snapshots/1cd8369d616808f623b6656dbe92bd97356c7d75/model_weights.pt"
    )
    num_positions: int = 350
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class TrainConfig:
    """Optimization hyperparameters for all Chess NLA training stages."""
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0

    # AutoReader (AR / Critic) settings
    ar_epochs: int = 3
    ar_batch_size: int = 8
    ar_lr: float = 3e-4
    ar_weight_decay: float = 0.01
    ar_warmup_steps: int = 20
    ar_max_steps: int = 400

    # AutoVerbalizer (AV SFT) settings
    av_epochs: int = 3
    av_batch_size: int = 4
    av_lr: float = 2e-4
    av_weight_decay: float = 0.01
    av_warmup_steps: int = 30
    av_max_steps: int = 500

    # AutoVerbalizer (AV RL - GRPO) settings
    rl_epochs: int = 1
    rl_batch_size: int = 1
    rl_group_size: int = 4
    rl_lr: float = 1e-5
    rl_kl_coef: float = 0.04
    rl_clip_eps: float = 0.2
    rl_temperature: float = 0.7
    rl_max_steps: int = 300
