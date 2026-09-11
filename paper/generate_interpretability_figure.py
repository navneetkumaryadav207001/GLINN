"""Generate the Interpretability Concept Circuits Figure for the NLA Paper."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

fig_dir = Path(__file__).resolve().parent / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)

# -------------------------------------------------------------
# Panel A: Value vs Policy Energy Allocation
# -------------------------------------------------------------
ax = axes[0, 0]
concepts = ["King Safety\n(Check Attacks)", "Rook Activity\n(Open Files)", "Material Balance\n(Adv. vs Deficit)", "Passed Pawn\n(Promotion)"]
val_pcts = [64.9, 55.5, 40.2, 37.7]
pol_pcts = [35.1, 44.5, 59.8, 62.3]

y = np.arange(len(concepts))
h = 0.55

b1 = ax.barh(y, val_pcts, height=h, label="Value Head Slice (z[0:256])", color="#4C72B0", edgecolor="black", linewidth=1.2)
b2 = ax.barh(y, pol_pcts, left=val_pcts, height=h, label="Policy Head Slice (z[256:768])", color="#DD8452", edgecolor="black", linewidth=1.2)

ax.set_yticks(y)
ax.set_yticklabels(concepts, fontweight="bold", fontsize=10)
ax.set_xlabel("Proportion of Conceptual Energy in Latent Representation (%)", fontweight="bold", fontsize=10)
ax.set_title("(A) Disentangled Architectural Energy Distribution", fontweight="bold", fontsize=11)
ax.set_xlim(0, 100)
ax.legend(loc="upper right", frameon=True, fontsize=9.5)

for i, (v, p) in enumerate(zip(val_pcts, pol_pcts)):
    ax.text(v/2, i, f"{v:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax.text(v + p/2, i, f"{p:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=10)

# -------------------------------------------------------------
# Panel B: Top Extracted Dimensions
# -------------------------------------------------------------
ax = axes[0, 1]
dims_labels = ["Dim 65\n(King Check)", "Dim 6\n(King Threat)", "Dim 17\n(King Pressure)", "Dim 75\n(Passed Pawn)", "Dim 674\n(Pol: Promotion)"]
deltas = [+7.50, +7.00, +6.75, -9.75, -13.41]
bar_colors = ["#2CA02C", "#2CA02C", "#2CA02C", "#D62728", "#E377C2"]

bars = ax.bar(dims_labels, deltas, color=bar_colors, width=0.55, edgecolor="black", linewidth=1.2)
ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
ax.set_title("(B) Top Semantic Concept Displacements in z (Delta z)", fontweight="bold", fontsize=11)
ax.set_ylabel("Activation Displacement (Delta z)", fontweight="bold", fontsize=10)
ax.set_ylim(-16, 10)

for b, d in zip(bars, deltas):
    h_val = b.get_height()
    if h_val >= 0:
        ax.text(b.get_x() + b.get_width()/2., h_val + 0.5, f"{d:+.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
    else:
        ax.text(b.get_x() + b.get_width()/2., h_val - 0.7, f"{d:+.2f}", ha="center", va="top", fontweight="bold", fontsize=9.5)

# -------------------------------------------------------------
# Panel C: Ground-Truth Dataset Verification (Dim 75 Passed Pawns)
# -------------------------------------------------------------
ax = axes[1, 0]
categories = ["Normal Boards\n(No Advanced Pawns)", "Real Boards with\nRank 6/7 Passed Pawns"]
act_means = [5.27, 12.15]
c_bars = ax.bar(categories, act_means, color=["#A0CBE8", "#E15759"], width=0.5, edgecolor="black", linewidth=1.2)
ax.set_title("(C) Ground-Truth Engine Activation on Dim 75 (Passed Pawn Detector)", fontweight="bold", fontsize=11)
ax.set_ylabel("SE-ResNet-20 Real Activation Value", fontweight="bold", fontsize=10)
ax.set_ylim(0, 15)

for b in c_bars:
    h_val = b.get_height()
    ax.text(b.get_x() + b.get_width()/2., h_val + 0.4, f"{h_val:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

ax.annotate("+6.89 Activation Surge!\n(> 2.3x Increase on Real Pawns)", xy=(1, 12.15), xytext=(0.5, 13.5),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
            ha="center", fontweight="bold", color="#D62728", fontsize=10.5)

# -------------------------------------------------------------
# Panel D: Causal Latent Steering in Verbalizer
# -------------------------------------------------------------
ax = axes[1, 1]
ax.axis("off")
ax.set_title("(D) Causal Latent Steering in AutoVerbalizer (Position 102)", fontweight="bold", fontsize=11)

box_base = dict(boxstyle="round,pad=0.5", facecolor="#F0F0F0", edgecolor="#888888", linewidth=1.2)
box_steer_king = dict(boxstyle="round,pad=0.5", facecolor="#E8F8F5", edgecolor="#27AE60", linewidth=1.5)
box_steer_pawn = dict(boxstyle="round,pad=0.5", facecolor="#FDEDEC", edgecolor="#E74C3C", linewidth=1.5)

ax.text(0.02, 0.90, "Original Position 102 Output (Unsteered Baseline):", fontweight="bold", fontsize=10, color="#333333")
ax.text(0.02, 0.72, "\"Looking at the board, Qd4+ stands out as a viable move... creates passed\npawn threat against the pawn on d5...\"", style="italic", fontsize=9, bbox=box_base)

ax.text(0.02, 0.52, "After Injecting Top-5 King Safety Dimensions (Dims 65, 6, 17, 165, 198):", fontweight="bold", fontsize=10, color="#27AE60")
ax.text(0.02, 0.35, "\"Kd4 stands out as a mistake... creates attacking chances against the\nopponent's king. Strategically, improve active piece coordination.\"", style="italic", fontsize=9, bbox=box_steer_king)

ax.text(0.02, 0.18, "After Injecting Top-5 Passed Pawn Dimensions (Dims 674, 75, 690, 241, 198):", fontweight="bold", fontsize=10, color="#E74C3C")
ax.text(0.02, 0.02, "\"Looking at the passed move g3... we see that Kd4 is a brilliant tactical\nopportunity... Looking at the passed move...\"", style="italic", fontsize=9, bbox=box_steer_pawn)

plt.tight_layout()
out_file = fig_dir / "interpretability_concept_circuits.png"
plt.savefig(out_file, dpi=300)
plt.close()
print(f"Generated {out_file}")
