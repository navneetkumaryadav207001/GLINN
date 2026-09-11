"""Generate all figures for the NLA Chess Research Paper.

Outputs:
  - High-res chess board diagrams (pos10, pos63, pos95, pos102, pos163, pos200, pos25, pos1)
  - Training dynamics curves (training_curves.png)
  - Benchmark comparison chart (baseline_vs_evalsole.png)
  - Causal steering shift chart (causal_steering.png)
  - Position 10 calibration comparison (position10_calibration.png)
  - Architecture pipeline diagram (system_architecture.png)
"""

from __future__ import annotations
import json
from pathlib import Path

import chess
import chess.svg
import cairosvg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig_dir = Path(__file__).resolve().parent / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Generate Chess Board PNGs
# -------------------------------------------------------------
boards = {
    "pos10": ("3r4/p4pkp/1p2P1p1/2p1Qp2/5nq1/2P5/PP1N1P2/4RK2 b - - 1 28", "Position 10: Sharp Black Attack (V = -1.00)"),
    "pos63": ("r1bqk2r/pppp1ppp/2n2n2/1Bb1p3/4P3/2P2N2/PP1P1PPP/RNBQ1RK1 b kq - 0 5", "Position 63: Balanced Ruy Lopez (V = -0.10)"),
    "pos95": ("8/p4kpp/1pp5/1P1p1P2/P1nPp1P1/2N1P3/5K1P/8 b - - 0 33", "Position 95: Contested Knight Endgame (V = -0.12)"),
    "pos102": ("6N1/3k4/1R6/1bp1p1p1/4P1P1/1r1PKP2/8/8 b - - 0 54", "Position 102: Dead Even Rook Endgame (V = 0.00)"),
    "pos163": ("6k1/1p1b1pbp/4p1p1/2BpP3/n2P3P/8/1PrN1PP1/1RR3K1 b - - 0 29", "Position 163: Quiet Middlegame (V = 0.00)"),
    "pos200": ("2rr2k1/pp3ppp/5n2/4p3/8/P1NPPB1q/1PPQ4/2RR3K w - - 0 20", "Position 200: Decisive White Lead (V = +0.77)"),
    "pos25": ("8/p5pk/7p/1p6/6N1/PP3PKP/2r5/8 b - - 0 43", "Position 25: Contested Rook Endgame"),
    "pos1": ("r1b2rk1/ppp2ppp/5q2/4b3/2B5/4P3/PP3PPP/1RBQ1RK1 b - - 1 12", "Position 1: Tactical Middlegame"),
}

for name, (fen, title) in boards.items():
    b = chess.Board(fen)
    svg_str = chess.svg.board(board=b, size=450, coordinates=True)
    out_png = fig_dir / f"{name}.png"
    cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), write_to=str(out_png))
    print(f"Generated {out_png}")

# -------------------------------------------------------------
# 2. Training Dynamics Curves
# -------------------------------------------------------------
log_path = Path(__file__).resolve().parents[1] / "reports" / "training_logs.json"
with open(log_path) as f:
    logs = json.load(f)

# Extract steps
steps = []
ar_losses = []
train_v_maes = []
train_logit_maes = []
cosines = []

val_steps = [0]
val_v_maes = [logs[0]["val_v_mae"]]
val_logit_maes = [logs[0]["val_logit_mae"]]

for entry in logs[1:]:
    s = entry["step"]
    steps.append(s)
    ar_losses.append(entry["ar_loss"])
    train_v_maes.append(entry["train_v_mae"])
    train_logit_maes.append(entry["train_logit_mae"])
    cosines.append(entry["passive_cosine"])
    if "val_v_mae" in entry:
        val_steps.append(s)
        val_v_maes.append(entry["val_v_mae"])
        val_logit_maes.append(entry["val_logit_mae"])

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

# (A) Logit MAE
ax = axes[0, 0]
ax.plot(steps, train_logit_maes, color="#4C72B0", alpha=0.35, label="Train Batch Logit MAE")
# moving average
ma_logit = np.convolve(train_logit_maes, np.ones(5)/5, mode="valid")
ax.plot(steps[2:-2], ma_logit, color="#4C72B0", lw=2, label="Train MA (w=5)")
ax.plot(val_steps, val_logit_maes, "o--", color="#C44E52", lw=2.5, markersize=8, label="Validation Logit MAE")
ax.set_title("(A) Pre-Tanh Logit Calibration Error", fontsize=12, fontweight="bold")
ax.set_xlabel("Co-training Steps")
ax.set_ylabel("Logit MAE")
ax.legend(frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)

# (B) Value Evaluation MAE
ax = axes[0, 1]
ax.plot(steps, train_v_maes, color="#55A868", alpha=0.35, label="Train Batch V MAE")
ma_v = np.convolve(train_v_maes, np.ones(5)/5, mode="valid")
ax.plot(steps[2:-2], ma_v, color="#55A868", lw=2, label="Train MA (w=5)")
ax.plot(val_steps, val_v_maes, "s--", color="#8172B2", lw=2.5, markersize=8, label="Validation V MAE")
ax.set_title("(B) Bounded Position Evaluation MAE (|ΔV|)", fontsize=12, fontweight="bold")
ax.set_xlabel("Co-training Steps")
ax.set_ylabel("Evaluation Error |V - V*| ∈ [0, 2]")
ax.legend(frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)

# (C) AutoReader Optimization Loss (Smooth L1)
ax = axes[1, 0]
ax.plot(steps, ar_losses, color="#DD8452", alpha=0.4, label="Raw Smooth L1 Loss")
ma_loss = np.convolve(ar_losses, np.ones(5)/5, mode="valid")
ax.plot(steps[2:-2], ma_loss, color="#DD8452", lw=2.2, label="Smooth L1 (MA)")
ax.set_title("(C) AutoReader Critic Loss (Pre-Tanh Logit Difference)", fontsize=12, fontweight="bold")
ax.set_xlabel("Co-training Steps")
ax.set_ylabel("Smooth L1 Loss")
ax.legend(frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)

# (D) Passive 768-d Vector Cosine Fidelity
ax = axes[1, 1]
ax.plot(steps, cosines, color="#64B5CD", alpha=0.4, label="Batch Cosine Sim")
ma_cos = np.convolve(cosines, np.ones(5)/5, mode="valid")
ax.plot(steps[2:-2], ma_cos, color="#1F77B4", lw=2.2, label="Cosine Sim (MA)")
ax.axhline(0.5775, color="#D62728", linestyle=":", lw=2, label="Baseline Cosine (0.5775)")
ax.set_title("(D) Representation Retention (Passive Cosine Similarity)", fontsize=12, fontweight="bold")
ax.set_xlabel("Co-training Steps")
ax.set_ylabel("Cosine Similarity")
ax.set_ylim(0.2, 0.8)
ax.legend(frameon=True)
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(fig_dir / "training_curves.png", dpi=300)
plt.close()
print("Generated training_curves.png")

# -------------------------------------------------------------
# 3. Baseline vs Eval-Sole Quantitative Comparison Chart
# -------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

# (A) Logit MAE
categories = ["Baseline (Cosine)", "Eval-Sole (Attached Head)"]
vals_logit = [24.85, 7.80]
colors_a = ["#A0CBE8", "#2E669A"]
bars1 = ax[0].bar(categories, vals_logit, color=colors_a, width=0.55, edgecolor="black", linewidth=1.2)
ax[0].set_title("Pre-Tanh Logit Error (50 Pos)", fontsize=11, fontweight="bold")
ax[0].set_ylabel("Mean Absolute Error")
ax[0].set_ylim(0, 30)
for b in bars1:
    h = b.get_height()
    ax[0].text(b.get_x() + b.get_width()/2., h + 0.8, f"{h:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax[0].text(0.5, 20, "−68.6%\nReduction", color="#D62728", fontweight="bold", fontsize=13, ha="center")

# (B) Value MAE
vals_v = [0.875, 0.396]
colors_b = ["#F1A340", "#998EC3"]
bars2 = ax[1].bar(categories, vals_v, color=colors_b, width=0.55, edgecolor="black", linewidth=1.2)
ax[1].set_title("Final Value Error (|ΔV| ∈ [0, 2])", fontsize=11, fontweight="bold")
ax[1].set_ylabel("Evaluation MAE")
ax[1].set_ylim(0, 1.1)
for b in bars2:
    h = b.get_height()
    ax[1].text(b.get_x() + b.get_width()/2., h + 0.03, f"{h:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax[1].text(0.5, 0.70, "−54.8%\nReduction", color="#D62728", fontweight="bold", fontsize=13, ha="center")

# (C) Global Cosine Fidelity
vals_cos = [0.5775, 0.5758]
colors_c = ["#B2DF8A", "#33A02C"]
bars3 = ax[2].bar(categories, vals_cos, color=colors_c, width=0.55, edgecolor="black", linewidth=1.2)
ax[2].set_title("Global 768-d Representation Retention", fontsize=11, fontweight="bold")
ax[2].set_ylabel("Cosine Similarity")
ax[2].set_ylim(0, 0.75)
for b in bars3:
    h = b.get_height()
    ax[2].text(b.get_x() + b.get_width()/2., h + 0.02, f"{h:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax[2].text(0.5, 0.35, "99.7%\nPreserved", color="#1F77B4", fontweight="bold", fontsize=13, ha="center")

plt.tight_layout()
plt.savefig(fig_dir / "baseline_vs_evalsole.png", dpi=300)
plt.close()
print("Generated baseline_vs_evalsole.png")

# -------------------------------------------------------------
# 4. Position 10 Calibration Gauge Chart
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
models = ["Engine Ground Truth", "Baseline Model (Reconstructed)", "Eval-Sole Model (Reconstructed)"]
logits = [-13.07, +16.12, -10.35]
colors = ["#2CA02C", "#D62728", "#1F77B4"]

bars = ax.barh(models, logits, color=colors, height=0.5, edgecolor="black", linewidth=1.2)
ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
ax.set_title("Position 10: Calibration & Sign Agreement (Sharp Black Attack, V* = −1.00)", fontsize=12, fontweight="bold")
ax.set_xlabel("Engine Pre-Tanh Logit Output (Negative = Black Winning, Positive = White Winning)")
ax.set_xlim(-16, 20)

for b, val in zip(bars, logits):
    w = b.get_width()
    if w < 0:
        ax.text(w - 0.5, b.get_y() + b.get_height()/2, f"{val:+.2f} (V = {np.tanh(val):.2f})", ha="right", va="center", fontweight="bold", fontsize=10)
    else:
        ax.text(w + 0.5, b.get_y() + b.get_height()/2, f"{val:+.2f} (V = {np.tanh(val):.2f})", ha="left", va="center", fontweight="bold", fontsize=10)

ax.text(-12, 1, "Wrong Sign!\n(Predicted White Win)", color="#D62728", fontweight="bold", ha="center")
ax.text(-10, 2.3, "Correct Sign!\n(90.7% Error Reduction)", color="#1F77B4", fontweight="bold", ha="center")

plt.tight_layout()
plt.savefig(fig_dir / "position10_calibration.png", dpi=300)
plt.close()
print("Generated position10_calibration.png")

# -------------------------------------------------------------
# 5. Causal Natural Language Steering Shifts
# -------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# (A) Position 10 (Black Winning) Interventions
interv_p10 = ["Original\n(Black Deficit)", "Word Swap\n('down'→'up')", "Inject\n'White Winning'", "Inject\n'Balanced/Draw'"]
logits_p10 = [-5.36, -4.96, -0.13, -3.30]
colors_p10 = ["#7293CB", "#E1974C", "#D35E60", "#84BA5B"]

bars_a = ax[0].bar(interv_p10, logits_p10, color=colors_p10, width=0.55, edgecolor="black", linewidth=1.2)
ax[0].axhline(0, color="black", linestyle="--", alpha=0.7)
ax[0].set_title("Position 10 (True Logit = −13.07)\nSteering Towards Neutral/Advantage", fontsize=11, fontweight="bold")
ax[0].set_ylabel("Reconstructed Logit")
ax[0].set_ylim(-7, 1)
for b in bars_a:
    h = b.get_height()
    ax[0].text(b.get_x() + b.get_width()/2., h - 0.4, f"{h:+.2f}", ha="center", va="top", fontweight="bold")
ax[0].annotate("Δ = +5.23 Logits\n(Neutralized)", xy=(2, -0.13), xytext=(2, -2.5),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
            ha="center", fontweight="bold", color="#D35E60")

# (B) Position 200 (White Winning) Interventions
interv_p200 = ["Original\n(White Lead)", "Word Swap\n('win'→'lose')", "Inject\n'Black Winning'", "Inject\n'Balanced/Draw'"]
logits_p200 = [+5.92, +5.92, +0.05, -3.30]
colors_p200 = ["#7293CB", "#E1974C", "#D35E60", "#84BA5B"]

bars_b = ax[1].bar(interv_p200, logits_p200, color=colors_p200, width=0.55, edgecolor="black", linewidth=1.2)
ax[1].axhline(0, color="black", linestyle="--", alpha=0.7)
ax[1].set_title("Position 200 (True Logit = +1.02)\nSteering Towards Neutral/Deficit", fontsize=11, fontweight="bold")
ax[1].set_ylabel("Reconstructed Logit")
ax[1].set_ylim(-5, 8)
for b in bars_b:
    h = b.get_height()
    ax[1].text(b.get_x() + b.get_width()/2., h + (0.3 if h >= 0 else -0.5), f"{h:+.2f}", ha="center", va="bottom" if h>=0 else "top", fontweight="bold")
ax[1].annotate("Δ = −9.23 Logits\n(Full Sign Inversion!)", xy=(3, -3.30), xytext=(2.5, 3.5),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
            ha="center", fontweight="bold", color="#84BA5B")

plt.tight_layout()
plt.savefig(fig_dir / "causal_steering.png", dpi=300)
plt.close()
print("Generated causal_steering.png")

# -------------------------------------------------------------
# 6. High-level Architecture Diagram
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
ax.axis("off")

def add_box(ax, xy, w, h, text, color, title=""):
    rect = patches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.03", edgecolor="black", facecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    cx = xy[0] + w / 2
    cy = xy[1] + h / 2
    if title:
        ax.text(cx, xy[1] + h - 0.05, title, ha="center", va="top", fontsize=10, fontweight="bold")
        ax.text(cx, cy - 0.03, text, ha="center", va="center", fontsize=8.5)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=9.5, fontweight="bold")

# Forward Loop (Verbalization)
add_box(ax, (0.02, 0.65), 0.18, 0.25, "Board Tensor\n(18 × 8 × 8)", "#E8F4F8", "Chess Position")
add_box(ax, (0.24, 0.65), 0.22, 0.25, "SE-ResNet-20 Dual-Head\nz ∈ R^{768}\n[Val: 256 | Pol: 512]", "#D0E1F9", "Chess Engine")
add_box(ax, (0.50, 0.65), 0.20, 0.25, "Multi-Token Projector\nz → 8 Virtual Tokens", "#FFEAA7", "Projector (Frozen)")
add_box(ax, (0.74, 0.65), 0.24, 0.25, "Qwen2.5-0.5B + LoRA\nGenerates Natural Text", "#D5F5E3", "AutoVerbalizer (Policy)")

# Arrows Forward
ax.annotate("", xy=(0.24, 0.77), xytext=(0.20, 0.77), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(0.50, 0.77), xytext=(0.46, 0.77), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(0.74, 0.77), xytext=(0.70, 0.77), arrowprops=dict(arrowstyle="->", lw=2))

# Intermediate Text
add_box(ax, (0.35, 0.42), 0.40, 0.14, "Natural Language Explanation:\n\"White is up material... tactical opportunity on d4\"", "#FFF3CD", "Natural Language")
ax.annotate("", xy=(0.86, 0.49), xytext=(0.86, 0.65), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(0.75, 0.49), xytext=(0.86, 0.49), arrowprops=dict(arrowstyle="->", lw=2))

# Inversion Loop (Reading & Evaluation Calibration)
add_box(ax, (0.74, 0.08), 0.24, 0.25, "Qwen2.5-0.5B + LoRA\nInverts Text to Vector", "#FADBD8", "AutoReader Critic")
add_box(ax, (0.46, 0.08), 0.24, 0.25, "Reconstructed Vector\nz_hat ∈ R^{768}\n[z_hat[:256] = Value]", "#FCF3CF", "Reconstruction")
add_box(ax, (0.18, 0.08), 0.24, 0.25, "Value Head Linear(256, 1)\nV_hat = tanh(w^T z_hat + b)", "#E8DAEF", "Attached Output Layer")

# Arrows Backward
ax.annotate("", xy=(0.86, 0.33), xytext=(0.55, 0.42), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(0.70, 0.20), xytext=(0.74, 0.20), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(0.42, 0.20), xytext=(0.46, 0.20), arrowprops=dict(arrowstyle="->", lw=2))

# Loss computation
add_box(ax, (0.02, 0.08), 0.12, 0.25, "Loss: Smooth L1\n(Logit Diff)\nRL: GRPO", "#EDBB99", "Optimization")
ax.annotate("", xy=(0.14, 0.20), xytext=(0.18, 0.20), arrowprops=dict(arrowstyle="->", lw=2))

plt.tight_layout()
plt.savefig(fig_dir / "system_architecture.png", dpi=300)
plt.close()
print("Generated system_architecture.png")
print("\nAll figures successfully generated!")
