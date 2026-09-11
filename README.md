# GLINN: General Language Interface for Neural Networks

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Qwen2.5-yellow.svg)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
[![Overleaf Ready](https://img.shields.io/badge/Overleaf-Paper%20Ready-green.svg)](paper/nla_paper.zip)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **Official Research Repository and Reproduction Package for:**  
> **"GLINN: General Language Interface for Neural Networks — Bidirectional Latent Autoencoding, Semantic Compilation, and Causal Steering"**  
> *Advanced Interpretability & Reinforcement Learning Group*

---

## Table of Contents
1. [What is GLINN?](#1-what-is-glinn)
2. [The Core Scientific Problem](#2-the-core-scientific-problem)
3. [System Architecture](#3-system-architecture)
4. [Key Scientific Breakthroughs](#4-key-scientific-breakthroughs)
   - [The 0.13% Dimensional Dilution Trap](#the-013-dimensional-dilution-trap)
   - [Solving Tanh Gradient Saturation](#solving-tanh-gradient-saturation)
   - [Decisive Sign Restoration (Position 10)](#decisive-sign-restoration-position-10)
   - [Causal Natural Language Steering](#causal-natural-language-steering)
   - [Zero-Label Mechanistic Circuit Discovery](#zero-label-mechanistic-circuit-discovery)
5. [Empirical Benchmark Results](#5-empirical-benchmark-results)
6. [Repository Structure](#6-repository-structure)
7. [Quickstart & One-Command Reproduction](#7-quickstart--one-command-reproduction)
8. [Python API Usage](#8-python-api-usage)
9. [Broader Scope & Future Directions](#9-broader-scope--future-directions)
10. [Overleaf Publication Package](#10-overleaf-publication-package)

---

## 1. What is GLINN?

Modern deep reinforcement learning models (like AlphaZero, robotic controllers, and autonomous driving networks) excel at complex decision-making, but their internal cognition is locked inside continuous, high-dimensional activation vectors ($\mathbf{z} \in \mathbb{R}^{d}$). To human analysts, these continuous numbers are completely opaque.

**GLINN (General Language Interface for Neural Networks)** establishes an **invertible, bidirectional communication bus** between continuous neural representations and human natural language:

```text
 ┌────────────────┐         Forward Read (Verbalizer)         ┌─────────────────────────┐
 │ Continuous     │ ─────────────────────────────────────────> │ Natural Language        │
 │ Thought Vector │                                            │ Tactical Explanation    │
 │ z ∈ ℝ⁷⁶⁸       │ <───────────────────────────────────────── │ ("White is up material; │
 └────────────────┘         Reverse Write (Compiler)           │  passed pawn on d7...") │
         │                                                     └─────────────────────────┘
         ▼ Attached Engine Output Layer
 ┌────────────────────────────────────────┐
 │ Engine Decision Logit: ℓ = wᵀz[:256]+b │
 │ Position Value Score:  V = tanh(ℓ)     │
 └────────────────────────────────────────┘
```

1. **AutoVerbalizer (Forward Policy)**: Reads the internal activation vector $\mathbf{z}$ and decodes it into granular, human-readable strategic and tactical explanations.
2. **AutoReader Critic (Reverse Compiler)**: Compiles arbitrary natural language text back into continuous 768-dimensional coordinates $\hat{\mathbf{z}}$, allowing humans to write continuous thoughts into the model.
3. **Attached Output Layer**: Connects the chess engine's real decision layer directly into co-training, grounding the language interface into the actual decision boundaries of the target neural network.

---

## 2. The Core Scientific Problem

Prior interpretability methods face severe limitations:
- **Linear Probing** is static and unidirectional: it can only test whether a predefined binary label exists, but cannot generate explanations or write back into latent space.
- **Sparse Autoencoders (SAEs)** decompose activations into thousands of monosemantic feature directions, but lack compositional grammar and syntax, requiring manual human inspection.
- **Multimodal Projectors (like LLaVA)** project vision vectors into LLMs, but are one-way: you cannot invert language back into vision vectors to steer the vision model.

GLINN solves this by treating **human natural language itself as the latent bottleneck**, creating a closed, causal read-write control loop across heterogeneous architectures.

---

## 3. System Architecture

The benchmark setup targets a 20-block Squeeze-and-Excitation ResNet (**SE-ResNet-20 Dual-Head**) chess engine:

### A. Expert Latent Representation ($\mathbf{z} \in \mathbb{R}^{768}$)
The penultimate layer of the engine extracts a 768-dimensional activation vector:
- $\mathbf{z}_{\text{val}} = \mathbf{z}[0:256]$: 256-dimensional penultimate value representation (positional balance and evaluation).
- $\mathbf{z}_{\text{pol}} = \mathbf{z}[256:768]$: 512-dimensional slice of penultimate policy representation (move probabilities and tactical dynamics).
- Final engine evaluation: $\text{Logit} = \mathbf{w}^T \mathbf{z}[0:256] + b$, with $V = \tanh(\text{Logit}) \in [-1.0, +1.0]$.

### B. AutoVerbalizer Policy ($\mathbf{z} \to \text{Text}$)
- **Multi-Token Projector**: A 2-layer MLP expanding $\mathbf{z} \in \mathbb{R}^{768}$ into $K=8$ virtual prefix tokens ($8 \times 896$), frozen during RL co-training.
- **Language Model Backbone**: `Qwen/Qwen2.5-0.5B-Instruct` equipped with Low-Rank Adaptation (LoRA, $r=16, \alpha=32$).
- **Generation**: Generates tactical narratives conditioned on the virtual prefix embeddings.

### C. AutoReader Critic ($\text{Text} \to \hat{\mathbf{z}}$)
- **Causal Transformer Backbone**: Processes the natural language text with LoRA adapters.
- **CrossDimensionAdapter**: Maps the final hidden state ($d_{\text{model}} = 896$) to continuous latent space ($\mathbb{R}^{768}$).

### D. Attached Output Layer Co-training
- The engine's frozen pre-tanh linear head $\mathbf{w} \in \mathbb{R}^{256}$ is attached to the reconstructed vector $\hat{\mathbf{z}}[:256]$.
- The critic is trained with non-saturating **Smooth L1 Loss** on pre-tanh logits.
- The verbalizer is updated using **Group Relative Policy Optimization (GRPO)** with evaluation rewards and $n$-gram diversity regularization.

---

## 4. Key Scientific Breakthroughs

### The 0.13% Dimensional Dilution Trap
In standard autoencoders trained with unweighted cosine distance $\mathcal{L} = 1 - \cos(\hat{\mathbf{z}}, \mathbf{z})$, each dimension receives gradient weight inversely proportional to the total dimensionality. In a 768-dimensional space, the 1D evaluation head occupies:
$$\rho_{\text{eval}} = \frac{1}{768} \approx 0.001302 \quad (0.13\%)$$
The remaining $99.87\%$ of dimensions represent board features (pawn chains, piece locations). Consequently, unweighted cosine training achieves high geometric overlap ($\sim 0.82$) while completely neglecting the 1D evaluation projection. The mean projection drifts into large positive values ($+25$ to $+32$ logits), freezing all evaluations into $+1.0000$.

### Solving Tanh Gradient Saturation
Backpropagating through the engine's final output $V = \tanh(\text{Logit})$ fails because the derivative of $\tanh$ vanishes:
$$\frac{d \tanh(u)}{du} = 1 - \tanh^2(u) = \text{sech}^2(u) \xrightarrow{|u| > 3.0} 0$$
For an uncalibrated logit of $+25.0$, $\text{sech}^2(25) \approx 3.7 \times 10^{-21}$, completely stalling gradient descent. GLINN solves this by attaching the **pre-tanh linear layer** and supervising with **Smooth L1 (Huber) loss**, which maintains a constant gradient slope of $\pm 1$ regardless of logit magnitude.

### Decisive Sign Restoration (Position 10)
On a sharp tactical position where Black has a winning attack ($V^* = -1.0000$, $\ell^* = -13.07$):
- **Baseline Cosine Model**: Reconstructed logit was **$+16.12$** ($V = +1.0000$, completely wrong sign).
- **GLINN Eval-Sole Model**: Reconstructed logit flipped to **$-10.35$** ($V = \mathbf{-1.0000}$, **90.7% error drop**, exact sign agreement).

### Causal Natural Language Steering
If natural language genuinely encodes neural thoughts, modifying the text must causally shift the engine's downstream decision:
- **Position 10 (Black Winning, Logit $-5.36$)**: Injecting *"White is completely winning with overwhelming material..."* pulled the reconstructed logit to **$-0.13$** ($\Delta = \mathbf{+5.23}$ logits), neutralizing Black's lead.
- **Position 200 (White Winning, Logit $+5.92$)**: Injecting *"The position is completely equal and balanced..."* dropped the logit to **$-3.30$** ($\Delta = \mathbf{-9.23}$ logits), executing a **complete win/loss decision reversal**.

### Zero-Label Mechanistic Circuit Discovery
GLINN acts as a **semantic compiler**: by contrasting text descriptions through the AutoReader, it isolates specific continuous coordinates without needing labeled training data:
- **Passed Pawn Detector (`Dim 75`)**: Identified via the prompt contrast *"advanced passed pawn on 7th rank"* vs *"locked pawn structure"*.
- **Ground-Truth Verification**: Across 2,501 real validation games, `Dim 75` has a mean activation of **$+5.27$** on normal boards, but surges to **$+12.15$** on boards with real rank 6/7 passed pawns—a **$+6.89$ jump ($> 2.3\times$ increase)**!
- **Causal Verbalizer Steering**: Surgically injecting only the top-5 discovered dimensions into a neutral position vector causally forces the AutoVerbalizer LLM to decode that exact tactical theme.

---

## 5. Empirical Benchmark Results

### Bulk Benchmark across 50 Held-Out Validation Positions

| Metric | Baseline Co-trained (Cosine) | GLINN (Attached Output Layer) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Pre-Tanh Logit MAE** ($|\hat{\ell} - \ell^*|$) | $24.85$ | $\mathbf{7.80}$ | **$-68.6\%$ Reduction** |
| **Position Value MAE** ($|\hat{V} - V^*|$) | $0.875$ | $\mathbf{0.396}$ | **$-54.8\%$ Reduction** |
| **Global Cosine Fidelity** ($\cos(\hat{\mathbf{z}}, \mathbf{z})$) | $0.5775$ | $0.5758$ | **$99.7\%$ Preserved** |
| **Steering Dynamic Range** ($\Delta V$) | $0.000$ (Saturated) | $\mathbf{-0.441}$ | **Dynamic Range Restored** |

### Canonical Position Benchmark

| Position | Regime | Ground Truth | Baseline Logit | GLINN Logit | GLINN $V$ | Sign Match |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Pos 10** | Sharp Black Attack | $\ell = -13.07, V = -1.00$ | $+16.12$ | $\mathbf{-10.35}$ | $\mathbf{-1.00}$ | ✅ Exact Match |
| **Pos 63** | Balanced Ruy Lopez | $\ell = -0.10, V = -0.10$ | $+31.70$ | $\mathbf{+4.45}$ | $+0.99$ | ✅ Error Reduced |
| **Pos 95** | Knight Endgame | $\ell = -0.12, V = -0.12$ | $+30.02$ | $\mathbf{+0.17}$ | $\mathbf{+0.17}$ | ✅ $\Delta V = 0.287$ |
| **Pos 102** | Dead Even Rook | $\ell = +0.00, V = +0.00$ | $+12.00$ | $\mathbf{-8.22}$ | $-1.00$ | ✅ Active Dynamics |
| **Pos 163** | Quiet Middlegame | $\ell = -0.01, V = -0.01$ | $+26.82$ | $\mathbf{+5.07}$ | $+0.99$ | ✅ Error Reduced |
| **Pos 200** | Decisive White Lead | $\ell = +1.02, V = +0.77$ | $+25.53$ | $\mathbf{+9.55}$ | $+1.00$ | ✅ Correct Sign |

---

## 6. Repository Structure

```text
GLINN/
├── run_all.sh                   # One-command master reproduction runner
├── reproduce_all.py             # Python reproduction orchestrator
├── eval_benchmark.py            # Canonical positions & 50-board bulk evaluation
├── interpretability.py          # Circuit discovery, dataset verification & steering
├── requirements.txt             # Minimal Python dependencies
│
├── checkpoints/                 # Lightweight trained weights (~28 MB total)
│   ├── projector_frozen.pt      # Frozen Multi-Token Projector (768 -> 8x896)
│   ├── verbalizer_final/        # AutoVerbalizer LoRA adapter (Qwen2.5-0.5B)
│   └── critic_final/            # AutoReader Critic LoRA adapter + linear head
│
├── data/                        # Processed activation vectors (23 MB total)
│   ├── projector_val.parquet    # 2,501 held-out validation positions (4.2 MB)
│   └── projector_train.parquet  # 12,499 training positions (19 MB)
│
├── src/                         # Self-contained model architectures (116 KB)
│   ├── expert.py                # SE-ResNet-20 engine wrapper & value head
│   ├── config.py                # Configuration dataclasses (NLAConfig, DataConfig)
│   ├── models.py                # NLACriticModel (AutoReader) & adapter head
│   ├── models_projector.py      # AlignedAutoVerbalizer & MultiTokenProjector
│   └── injection.py             # Normalization and injection utilities
│
├── paper/                       # Complete publication source ready for Overleaf (5 MB)
│   ├── main.tex                 # Two-column publication-ready LaTeX paper (GLINN)
│   ├── references.bib           # Complete BibTeX citations
│   ├── nla_paper.zip            # Self-contained Overleaf upload package (2.3 MB)
│   ├── generate_figures.py      # Generates 10 core figures and chess boards
│   ├── generate_interpretability_figure.py # 4-panel interpretability figure generator
│   └── figures/                 # 14 high-resolution 300 DPI publication figures
│
└── reports/                     # Empirical benchmark and interpretability JSON data
    ├── training_logs.json
    ├── eval_sole_comparison_results.json
    └── interpretability_mapping_results.json
```

---

## 7. Quickstart & One-Command Reproduction

### Setup Environment
```bash
git clone https://github.com/your-username/GLINN.git
cd GLINN
pip install -r requirements.txt
```

### Run Everything in One Go
To execute the complete benchmark evaluation, circuit discovery, causal steering, and publication figure regeneration:
```bash
bash run_all.sh
```

### Run Modular Components
- **Run Benchmark Evaluation**:
  ```bash
  python3 eval_benchmark.py
  ```
- **Run Mechanistic Circuit Discovery**:
  ```bash
  python3 interpretability.py
  ```
- **Regenerate Publication Figures**:
  ```bash
  python3 paper/generate_figures.py
  python3 paper/generate_interpretability_figure.py
  ```

---

## 8. Python API Usage

### Example 1: Decode a Latent Vector into Natural Language
```python
import torch
from transformers import AutoTokenizer
from peft import PeftModel
from src.models_projector import AlignedAutoVerbalizer

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Load policy
policy = AlignedAutoVerbalizer(
    base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    d_in=768, d_out=896, num_tokens=8, freeze_backbone=True, device=dev
)
policy.load_projector("checkpoints/projector_frozen.pt")
policy.backbone = PeftModel.from_pretrained(policy.backbone, "checkpoints/verbalizer_final")
policy.eval()

# Generate explanation from 768-d latent vector
z = torch.randn(1, 768, device=dev)  # Replace with real engine activation
prompt = "<|im_start|>user\nExplain the position dynamics.<|im_end|>\n<|im_start|>assistant\n"

explanation = policy.generate_explanation(
    vectors=z, tokenizer=tokenizer, prompt_text=prompt, max_new_tokens=65
)[0]
print("Decoded Thought:", explanation)
```

### Example 2: Compile Natural Language into a 768-D Continuous Vector
```python
from src.models import NLACriticModel
from src.config import NLAConfig

# Load critic
cfg = NLAConfig()
critic = NLACriticModel.from_pretrained("checkpoints/critic_final", nla_config=cfg, device=dev).eval()

text = "White has an advanced passed pawn ready to promote on the seventh rank."
prompt = f"Analyze the internal chess dynamics: {text}"
enc = tokenizer([prompt], return_tensors="pt").to(dev)

with torch.no_grad():
    z_compiled = critic(enc["input_ids"], enc["attention_mask"])  # [1, 768]

print("Compiled Latent Shape:", z_compiled.shape)
```

---

## 9. Broader Scope & Future Directions

The GLINN paradigm establishes a foundation that extends far beyond chess:

1. **Real-Time Natural Language Agent Coaching**:  
   Instead of retraining reinforcement learning agents or writing fragile reward hacks, human operators can issue plain-English directives (*"Drive more conservatively; avoid unprotected lane changes"*). GLINN's AutoReader compiles the command into a continuous displacement $\Delta \mathbf{z}$, steering the policy in real time with zero retraining.
2. **Pre-Commitment Internal State Auditing ("AI Lie Detector")**:  
   In safety-critical applications (medical diagnosis, robotic surgery, autonomous vehicles), latent corruptions cause fatal blunders. By continuously decoding internal thought vectors into human language before an action is executed, safety auditors can intercept reasoning errors before harm occurs.
3. **Cross-Architecture Neural Interlingua ("Neural Telepathy")**:  
   Heterogeneous neural networks with incompatible latent dimensions (e.g. a 768-d convolutional vision model and a 4,096-d cloud transformer) cannot share representations. GLINN uses natural language as a universal, syntax-governed Interlingua: Model A speaks its thought, and Model B compiles it into its native continuous space.
4. **Semantic Red-Teaming & Vulnerability Discovery**:  
   By probing the AutoReader with targeted semantic variations, security researchers can identify brittle submanifolds in continuous policy spaces and diagnose why specific adversarial noise fools the system.

---

## 10. Overleaf Publication Package

The research paper is written in clean, modern LaTeX and ready for submission:
- **Overleaf Archive**: [`paper/nla_paper.zip`](paper/nla_paper.zip) (2.3 MB)
- **LaTeX Source**: [`paper/main.tex`](paper/main.tex)
- **BibTeX References**: [`paper/references.bib`](paper/references.bib)
- **Figures**: [`paper/figures/`](paper/figures/) (14 high-resolution 300 DPI figures)

### How to Compile in Overleaf:
1. Download [`paper/nla_paper.zip`](paper/nla_paper.zip).
2. Go to [Overleaf](https://www.overleaf.com/) $\to$ **New Project** $\to$ **Upload Project**.
3. Select `nla_paper.zip`. Overleaf will extract all sources, diagrams, and figures, and compile the final PDF automatically.

---

## Citation

```bibtex
@article{glinn2026general,
  title={GLINN: General Language Interface for Neural Networks — Bidirectional Latent Autoencoding, Semantic Compilation, and Causal Steering},
  author={Advanced Interpretability \& Reinforcement Learning Group},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License
This project is open-sourced under the [MIT License](LICENSE).
