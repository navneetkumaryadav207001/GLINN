# GLINN: General Language Interface for Neural Networks

> **Official Reproduction Repository for the Research Paper:**  
> *"GLINN: General Language Interface for Neural Networks — Bidirectional Latent Autoencoding, Semantic Compilation, and Causal Steering"*

---

## 1. Overview

**GLINN** establishes a bidirectional bridge between continuous neural activations ($\mathbf{z} \in \mathbb{R}^{768}$) and human natural language. Applied to an expert dual-head chess neural network (SE-ResNet-20), GLINN provides:

1. **AutoVerbalizer (Policy)**: Translates continuous internal thought vectors into tactical explanations.
2. **AutoReader (Critic)**: Compiles natural language explanations back into continuous activation space.
3. **Attached Output Layer Calibration**: Solves the *0.13% dimensional dilution trap*, reducing logit error by **68.6%** and position value MAE by **54.8%**.
4. **Bidirectional Circuit Discovery**: Automatically isolates continuous coordinates for human chess concepts (e.g. Dim 75 as an internal passed-pawn detector, surging $+6.89$ on real boards).
5. **Causal Natural Language Steering**: Editing English explanations shifts the engine's internal evaluation by up to **9.23 logits**, executing full win/loss decision reversals.

---

## 2. Quickstart: One-Command Reproduction

To run the complete pipeline (benchmark evaluation, circuit discovery, causal steering, and figure regeneration) in one go:

```bash
cd GLINN
bash run_all.sh
```
*(Or directly via Python: `python3 reproduce_all.py`)*

---

## 3. Repository Architecture

```text
GLINN/
├── run_all.sh                   # One-command executable bash runner
├── reproduce_all.py             # Master Python reproduction orchestrator
├── eval_benchmark.py            # Benchmark evaluation on 50 positions & canonical boards
├── interpretability.py          # Circuit discovery, dataset verification & causal steering
├── requirements.txt             # Minimal Python dependencies
│
├── checkpoints/                 # Lightweight trained checkpoints (~28 MB total)
│   ├── projector_frozen.pt      # Frozen Multi-Token Projector (768 -> 8x896)
│   ├── verbalizer_final/        # Trained AutoVerbalizer LoRA adapter (Qwen2.5-0.5B)
│   └── critic_final/            # Trained AutoReader Critic LoRA adapter + linear head
│
├── data/                        # Processed activation datasets
│   ├── projector_val.parquet    # 2,501 held-out validation positions + vectors (4.2 MB)
│   └── projector_train.parquet  # 12,499 training positions + vectors (19 MB)
│
├── src/                         # Self-contained core model architectures
│   ├── expert.py                # SE-ResNet-20 dual-head engine wrapper & value head
│   ├── config.py                # Configuration dataclasses (NLAConfig, DataConfig)
│   ├── models.py                # NLACriticModel (AutoReader) & CrossDimensionAdapter
│   ├── models_projector.py      # AlignedAutoVerbalizer & MultiTokenProjector
│   └── injection.py             # Vector normalization and injection utilities
│
├── paper/                       # Complete publication source ready for Overleaf
│   ├── main.tex                 # Two-column publication-ready LaTeX paper
│   ├── references.bib           # Complete BibTeX citations
│   ├── nla_paper.zip            # Self-contained Overleaf upload package (2.3 MB)
│   ├── generate_figures.py      # Generates 10 core figures and chess boards
│   ├── generate_interpretability_figure.py # Generates 4-panel interpretability figure
│   └── figures/                 # 14 high-resolution 300 DPI publication figures
│
└── reports/                     # Empirical benchmark and interpretability JSON data
    ├── training_logs.json
    ├── eval_sole_comparison_results.json
    └── interpretability_mapping_results.json
```

---

## 4. Individual Script Usage

### A. Run Benchmark Evaluation
Evaluates canonical positions (Pos 10, 63, 95, 102, 163, 200) and 50 held-out validation positions:
```bash
python3 eval_benchmark.py
```

### B. Run Mechanistic Interpretability & Circuit Discovery
Extracts semantic coordinates from text, verifies activations against 2,501 real boards, and executes causal verbalizer steering:
```bash
python3 interpretability.py
```

### C. Regenerate Publication Figures
Re-renders all board SVGs, training curves, comparison bar charts, and circuit figures:
```bash
python3 paper/generate_figures.py
python3 paper/generate_interpretability_figure.py
```

---

## 5. Overleaf Deployment

The folder `paper/` is formatted to compile out-of-the-box on Overleaf:
1. Download `paper/nla_paper.zip`.
2. Go to [Overleaf](https://www.overleaf.com/) $\to$ **New Project** $\to$ **Upload Project**.
3. Select `nla_paper.zip`. Overleaf will unpack and compile the PDF with all figures and tables automatically.
