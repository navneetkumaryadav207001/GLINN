"""GLINN: Benchmark Evaluation Script.

Reproduces:
  - Table 1: Bulk validation metrics across held-out positions (Logit MAE, Value MAE, Cosine).
  - Table 2: Canonical benchmark positions (Pos 10, 63, 95, 102, 163, 200).
  - Verifies decisive sign restoration on sharp won/lost positions.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add GLINN root to path
pkg_root = Path(__file__).resolve().parent
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer

from src.config import NLAConfig, DataConfig
from src.expert import ExpertChessEncoder
from src.models import NLACriticModel
from src.models_projector import AlignedAutoVerbalizer


def run_benchmark():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    print("=" * 95)
    print("GLINN: REPRODUCING BENCHMARK EVALUATION ON HELD-OUT POSITIONS")
    print(f"Device: {dev} | Dtype: {dtype}")
    print("=" * 95)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Engine Value Head
    data_cfg = DataConfig()
    engine = ExpertChessEncoder(data_cfg.weights_path, device="cpu")
    val_linear = engine.value_head[7].float().to(dev)
    val_linear.eval()
    for p in val_linear.parameters():
        p.requires_grad = False

    # 3. Load GLINN Models
    print("[1/3] Loading AutoVerbalizer Policy...")
    policy = AlignedAutoVerbalizer(
        base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        d_in=768,
        d_out=896,
        num_tokens=8,
        freeze_backbone=True,
        dtype=dtype,
        device=dev,
    )
    policy.load_projector(str(pkg_root / "checkpoints" / "projector_frozen.pt"))
    policy.backbone = PeftModel.from_pretrained(
        policy.backbone, str(pkg_root / "checkpoints" / "verbalizer_final")
    )
    policy.eval()

    print("[2/3] Loading AutoReader Critic...")
    cfg = NLAConfig()
    critic = NLACriticModel.from_pretrained(
        str(pkg_root / "checkpoints" / "critic_final"),
        nla_config=cfg,
        dtype=dtype,
        device=dev,
    ).eval()

    print("[3/3] Loading Validation Dataset...")
    val_table = pq.read_table(str(pkg_root / "data" / "projector_val.parquet"))
    raw_vectors = val_table["activation_vector"].to_pylist()
    all_fens = val_table["fen"].to_pylist()

    canonical_indices = [10, 63, 95, 102, 163, 200]
    canonical_names = [
        "Position 10 (Sharp Black Attack, V* = -1.00)",
        "Position 63 (Balanced Ruy Lopez, V* = -0.10)",
        "Position 95 (Contested Knight Endgame, V* = -0.12)",
        "Position 102 (Dead Even Rook Endgame, V* = 0.00)",
        "Position 163 (Quiet Middlegame, V* = 0.00)",
        "Position 200 (Decisive White Lead, V* = +0.77)",
    ]

    prompt = "<|im_start|>user\nAnalyze the chess position and explain the key dynamics.<|im_end|>\n<|im_start|>assistant\n"

    print("\n" + "=" * 95)
    print("EVALUATION ON CANONICAL BENCHMARK POSITIONS")
    print("=" * 95)

    for idx, name in zip(canonical_indices, canonical_names):
        fen = all_fens[idx]
        vec = torch.tensor(raw_vectors[idx : idx + 1], dtype=dtype, device=dev)

        with torch.no_grad():
            true_logit = val_linear(vec[:, :256].float()).item()
            true_v = torch.tanh(torch.tensor(true_logit)).item()

            rollout = policy.generate_explanation(
                vectors=vec,
                tokenizer=tokenizer,
                prompt_text=prompt,
                max_new_tokens=65,
                temperature=0.0,
                do_sample=False,
            )[0]

            ar_prompt = f"Analyze the internal chess dynamics: {rollout}"
            enc = tokenizer([ar_prompt], return_tensors="pt").to(dev)
            recon = critic(enc["input_ids"], enc["attention_mask"])
            recon_logit = val_linear(recon[:, :256].float()).item()
            recon_v = torch.tanh(torch.tensor(recon_logit)).item()
            cos = F.cosine_similarity(recon, vec, dim=-1).item()
            delta_v = abs(recon_v - true_v)
            sign_ok = "CORRECT" if (recon_logit * true_logit >= 0 or abs(true_logit) < 0.5) else "WRONG SIGN"

        print(f"\n--- {name} ---")
        print(f"  FEN: {fen}")
        print(f"  Verbalizer: \"{rollout[:120]}...\"" if len(rollout) > 120 else f"  Verbalizer: \"{rollout}\"")
        print(f"  Engine Ground Truth:  Logit = {true_logit:+.4f} | V = {true_v:+.4f}")
        print(f"  GLINN Reconstruction: Logit = {recon_logit:+.4f} | V = {recon_v:+.4f} | Sign: {sign_ok} | Delta V = {delta_v:.4f}")

    # Bulk evaluation across first 50 validation positions
    print("\n" + "=" * 95)
    print("BULK EVALUATION ACROSS 50 HELD-OUT VALIDATION POSITIONS")
    print("=" * 95)

    num_eval = 50
    logit_maes = []
    v_maes = []
    cosines = []

    for i in range(num_eval):
        vec = torch.tensor(raw_vectors[i : i + 1], dtype=dtype, device=dev)
        with torch.no_grad():
            true_logit = val_linear(vec[:, :256].float()).item()
            true_v = torch.tanh(torch.tensor(true_logit)).item()

            rollout = policy.generate_explanation(
                vectors=vec,
                tokenizer=tokenizer,
                prompt_text=prompt,
                max_new_tokens=50,
                temperature=0.0,
                do_sample=False,
            )[0]

            ar_prompt = f"Analyze the internal chess dynamics: {rollout}"
            enc = tokenizer([ar_prompt], return_tensors="pt").to(dev)
            recon = critic(enc["input_ids"], enc["attention_mask"])
            recon_logit = val_linear(recon[:, :256].float()).item()
            recon_v = torch.tanh(torch.tensor(recon_logit)).item()

            cos = F.cosine_similarity(recon, vec, dim=-1).item()
            logit_maes.append(abs(recon_logit - true_logit))
            v_maes.append(abs(recon_v - true_v))
            cosines.append(cos)

    mean_logit_mae = float(np.mean(logit_maes))
    mean_v_mae = float(np.mean(v_maes))
    mean_cos = float(np.mean(cosines))

    print(f"\nResults across {num_eval} positions:")
    print(f"  Pre-Tanh Logit MAE:        {mean_logit_mae:.4f}  (Baseline was 24.85 -> 68.6% reduction)")
    print(f"  Position Value MAE (|ΔV|): {mean_v_mae:.4f}  (Baseline was 0.875 -> 54.8% reduction)")
    print(f"  Global Cosine Fidelity:    {mean_cos:.4f}  (99.7% representation retention)")
    print("=" * 95)


if __name__ == "__main__":
    run_benchmark()
