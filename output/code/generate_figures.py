"""
generate_figures.py — Generate all figures for the T4 constructive transformer paper.

Produces:
  fig_01_architecture.png   — Constructive transformer skeleton (chapter 1)
  fig_02_heatmap.png        — 4-state dot-product matrix + (sign,mag) inset (chapter 2)
  fig_03_phi_gate.png       — SiLU with phi-boundary markers (chapter 3)
  fig_04_cross_copy.png     — Attention vs MLP cross-axis routing (chapter 4)
  fig_05_sva_logits.png     — Subject-verb agreement, all 4 cases (chapter 5)
  fig_06_t5d_trajectory.png — Full T5d auto-loop (chapter 8)         [from F-CT-T5D]
  fig_07_t5d_restart.png    — T5d restart-phase zoom (chapter 8)     [from F-CT-T5D]
  fig_08_t5d_axes.png       — Axis-discovery timeline (chapter 8)

Run: python3 output/code/generate_figures.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Use the shared TruthSpace visual identity (figstyle.py lives next to
# this file).  Falls back to vanilla matplotlib if not available.
sys.path.insert(0, str(HERE))
try:
    from figstyle import (apply_style, INK, INK_SOFT, GOLD, GOLD_DARK,
                          GOLD_SOFT, RED, TEAL, VIOLET, SAND, PAPER,
                          GRID, MUTED, SEQ, CAT)
    apply_style()
    _STYLED = True
except Exception:  # pragma: no cover
    INK = "#1B2D4A"; INK_SOFT = "#4A5A75"; GOLD = "#C9962B"
    GOLD_DARK = "#8C6517"; GOLD_SOFT = "#F0DDA8"; RED = "#C0392B"
    TEAL = "#1F8A82"; VIOLET = "#6E3FA3"; SAND = "#E9DFC9"
    PAPER = "#FBFAF6"; GRID = "#D8DEE9"; MUTED = "#9AA4B2"
    _STYLED = False

PHI = (1 + 5 ** 0.5) / 2
DPI = 200  # match truthspace_paper savefig.dpi

# Paper-wide semantic palette, all sourced from the truthspace
# figstyle so figures in both papers harmonise.
COL_POS    = GOLD          # positive / "+" pole / featured value
COL_NEG    = INK           # negative / opposite pole
COL_ZERO   = SAND          # neutral / zero cell
COL_PASS   = TEAL          # pass / works / margin +2
COL_FAIL   = RED           # fail / breaks / margin 0
COL_AXIS_A = INK           # axis A in routing diagrams
COL_AXIS_B = GOLD          # axis B
COL_AXIS_C = VIOLET        # axis C
COL_HILITE = GOLD          # highlight box around -2 cells

# Custom diverging colormap that matches the paper palette: navy ink
# for negative, paper-cream for zero, gold for positive.  Replaces
# matplotlib's stock RdBu_r in the dot-product heatmap.
TSCMAP = LinearSegmentedColormap.from_list(
    "ts_diverging", [(0.0, INK), (0.5, SAND), (1.0, GOLD)], N=256,
)


# ── Fig 1 (chapter 1.4): constructive-transformer architecture ─────────

def fig_01_architecture():
    """Boxes-and-arrows diagram of the construction's skeleton:
    residual = concat(axis_block_1, ..., axis_block_K), with each block
    showing its sub-dimensions; attention routes content between positions,
    MLP routes between blocks within a position.
    """
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    # ---- Three token positions, each a residual stream of K=3 blocks. ---
    n_pos = 3
    n_blk = 3
    block_h = 0.55
    block_w = 0.95
    pos_x = [1.4, 4.4, 7.4]
    blk_y0 = 1.1
    blk_colors = [COL_AXIS_A, COL_AXIS_B, COL_AXIS_C]
    blk_names = ["NUMBER", "LEX_CLASS", "TENSE"]

    # Token labels.
    tokens = ["boy", "AGREE_BE_PRESENT", "→ is"]
    for i, t in enumerate(tokens):
        ax.text(pos_x[i] + block_w / 2, blk_y0 - 0.45, t,
                ha="center", va="top", fontsize=10, fontweight="bold",
                color=INK)
        ax.text(pos_x[i] + block_w / 2, blk_y0 + n_blk * block_h + 0.18,
                f"position {i}", ha="center", va="bottom",
                fontsize=8, color=INK_SOFT)

    # Each token: K=3 stacked axis blocks.
    for i, x in enumerate(pos_x):
        for k in range(n_blk):
            y = blk_y0 + k * block_h
            box = FancyBboxPatch((x, y), block_w, block_h * 0.9,
                                   boxstyle="round,pad=0.02",
                                   edgecolor="black", facecolor=blk_colors[k],
                                   alpha=0.20, linewidth=1.0)
            ax.add_patch(box)
            ax.text(x + block_w / 2, y + block_h * 0.45,
                    blk_names[k], ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color=blk_colors[k])

    # Residual-stream label on the left.
    ax.text(0.95, blk_y0 + n_blk * block_h * 0.5,
            "residual\nstream\n=\nconcat",
            ha="right", va="center", fontsize=9, color=INK)

    # ---- Attention arrow: from position 0's NUMBER block to position 1's. ---
    arr_atn = FancyArrowPatch(
        (pos_x[0] + block_w, blk_y0 + 0.5 * block_h),
        (pos_x[1], blk_y0 + 0.5 * block_h),
        arrowstyle="->", mutation_scale=20,
        color=COL_AXIS_A, linewidth=2.0,
        connectionstyle="arc3,rad=-0.35",
    )
    ax.add_patch(arr_atn)
    ax.text((pos_x[0] + pos_x[1] + block_w) / 2,
            blk_y0 + 0.5 * block_h - 0.55,
            "attention head:  routes NUMBER across positions",
            ha="center", va="top", fontsize=8.5,
            color=COL_AXIS_A, fontweight="bold")

    # ---- MLP arrow: cross-axis routing within position 1 (LEX <-> TENSE). ----
    mlp_x = pos_x[1] + block_w + 0.45
    mlp_y = blk_y0 + 1.5 * block_h
    mlp_box = FancyBboxPatch((mlp_x, mlp_y - 0.32), 1.55, 0.7,
                                boxstyle="round,pad=0.04",
                                facecolor=GOLD_SOFT, edgecolor=GOLD_DARK,
                                linewidth=1.4)
    ax.add_patch(mlp_box)
    ax.text(mlp_x + 0.78, mlp_y + 0.02, "SwiGLU MLP",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=GOLD_DARK)
    ax.text(mlp_x + 0.78, mlp_y - 0.20, "cross-axis routing",
            ha="center", va="center", fontsize=7.5, color=INK_SOFT)

    # arrow from LEX block into MLP and back into TENSE.
    ax.add_patch(FancyArrowPatch(
        (pos_x[1] + block_w, blk_y0 + 1.5 * block_h),
        (mlp_x, mlp_y),
        arrowstyle="->", mutation_scale=14,
        color=COL_AXIS_B, linewidth=1.2,
        connectionstyle="arc3,rad=0.0",
    ))
    ax.add_patch(FancyArrowPatch(
        (mlp_x, mlp_y),
        (pos_x[1] + block_w, blk_y0 + 2.5 * block_h),
        arrowstyle="->", mutation_scale=14,
        color=COL_AXIS_C, linewidth=1.2,
        connectionstyle="arc3,rad=-0.2",
    ))

    # ---- LM head: position 2 outputs the verb token. ----
    arr_lm = FancyArrowPatch(
        (pos_x[2] + block_w, blk_y0 + 1.5 * block_h),
        (9.6, blk_y0 + 1.5 * block_h),
        arrowstyle="->", mutation_scale=18,
        color=INK_SOFT, linewidth=1.4,
    )
    ax.add_patch(arr_lm)
    ax.text(9.65, blk_y0 + 1.5 * block_h, "LM head\n(integer\ndot product)",
            ha="left", va="center", fontsize=8.5, color=INK)

    # ---- Title and tagline. ----
    ax.text(5.0, 4.95,
            "The constructive transformer skeleton",
            ha="center", va="top", fontsize=13, fontweight="bold",
            color=INK)
    ax.text(5.0, 4.55,
            "Each position holds K axis blocks. "
            "Attention routes content between positions; "
            "MLP routes content between blocks at one position.",
            ha="center", va="top", fontsize=9, color=INK_SOFT,
            style="italic")

    # ---- Legend. ----
    legend_y = 0.38
    handles = [
        mpatches.Patch(facecolor=COL_AXIS_A, alpha=0.35, label="NUMBER block"),
        mpatches.Patch(facecolor=COL_AXIS_B, alpha=0.35, label="LEX_CLASS block"),
        mpatches.Patch(facecolor=COL_AXIS_C, alpha=0.35, label="TENSE block"),
    ]
    ax.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=9)

    fig.subplots_adjust(top=0.88, bottom=0.10)
    fig.savefig(FIGURES / "fig_01_architecture.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_01_architecture.png")


# ── Fig 2 (chapter 2.1): 4-state dot-product heatmap + corner inset ────

def fig_02_heatmap():
    STATES = ["+1", "+0", "-0", "-1"]
    VEC = {
        "+1": np.array([+1, +1]),
        "+0": np.array([+1, -1]),
        "-0": np.array([-1, -1]),
        "-1": np.array([-1, +1]),
    }
    D = np.array([[int(VEC[a] @ VEC[b]) for b in STATES] for a in STATES])

    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0],
                          left=0.06, right=0.98, top=0.86,
                          bottom=0.22, wspace=0.28)
    ax_heat   = fig.add_subplot(gs[0, 0])
    ax_corner = fig.add_subplot(gs[0, 1])

    # ---- Heatmap. -----------------------------------------------------
    im = ax_heat.imshow(D, cmap=TSCMAP, vmin=-2, vmax=2)
    for i in range(4):
        for j in range(4):
            v = D[i, j]
            label = "0" if v == 0 else f"{v:+d}"
            color = "white" if abs(v) == 2 else INK
            ax_heat.text(j, i, label, ha="center", va="center",
                         fontsize=15, fontweight="bold", color=color)
    ax_heat.set_xticks(range(4))
    ax_heat.set_yticks(range(4))
    ax_heat.set_xticklabels(STATES, fontsize=12)
    ax_heat.set_yticklabels(STATES, fontsize=12)
    ax_heat.set_xlabel("state $b$", fontsize=11)
    ax_heat.set_ylabel("state $a$", fontsize=11)
    ax_heat.set_title(r"$D_{ab} = \mathrm{sign}_a\cdot\mathrm{sign}_b "
                      r"+ \mathrm{mag}_a\cdot\mathrm{mag}_b$",
                      fontsize=11, pad=8)

    # Highlight the four genuinely-distinguishing -2 cells.  These
    # are the navy-coloured cells under the new TSCMAP; the gold
    # frame makes them pop without fighting the cell colour.
    minus_two_cells = [(i, j) for i in range(4) for j in range(4)
                        if D[i, j] == -2]
    for i, j in minus_two_cells:
        ax_heat.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         fill=False, edgecolor=GOLD,
                                         linewidth=2.6, linestyle="-"))
    ax_heat.text(0.5, -0.18,
                  "gold frames: diagonally-opposite pairs (differ in BOTH sign and mag) give $-2$\n"
                  "identical = $+2$,  pairs that differ in just one coordinate = $0$",
                  transform=ax_heat.transAxes, ha="center", va="top",
                  fontsize=8.5, color=INK_SOFT, style="italic")

    # ---- Corner inset: 4 states at corners of the (sign,mag) square. ---
    ax_corner.set_xlim(-1.7, 1.7)
    ax_corner.set_ylim(-1.7, 1.7)
    ax_corner.set_aspect("equal")
    ax_corner.axhline(0, color="#888", linewidth=0.6)
    ax_corner.axvline(0, color="#888", linewidth=0.6)
    ax_corner.set_xticks([-1, 0, +1])
    ax_corner.set_yticks([-1, 0, +1])
    ax_corner.set_xticklabels(["$-1$", "$0$", "$+1$"], fontsize=10)
    ax_corner.set_yticklabels(["$-1$", "$0$", "$+1$"], fontsize=10)
    ax_corner.set_xlabel("sign", fontsize=10)
    ax_corner.set_ylabel("mag", fontsize=10)
    ax_corner.set_title("The 4 states as corners of the unit square",
                         fontsize=11, fontweight="bold", pad=8)
    ax_corner.grid(alpha=0.25)

    state_pos = {
        "+1": (+1, +1),
        "+0": (+1, -1),
        "-0": (-1, -1),
        "-1": (-1, +1),
    }
    # Place labels OUTSIDE the markers so they don't overlap.
    label_offset = {
        "+1": (+0.30, +0.22),
        "+0": (+0.30, -0.22),
        "-0": (-0.30, -0.22),
        "-1": (-0.30, +0.22),
    }
    for name, (x, y) in state_pos.items():
        ax_corner.scatter([x], [y], s=110, color="white",
                          edgecolor="black", zorder=3, linewidth=1.4)
        dx, dy = label_offset[name]
        ax_corner.annotate(name, xy=(x, y), xytext=(x + dx, y + dy),
                            fontsize=12, fontweight="bold",
                            ha="center", va="center", zorder=4)

    # Diagonals (the -2 pairs).
    diag_pairs = [("+1", "-0"), ("+0", "-1")]
    for a, b in diag_pairs:
        x0, y0 = state_pos[a]
        x1, y1 = state_pos[b]
        ax_corner.plot([x0, x1], [y0, y1], color=COL_NEG,
                       linewidth=2.0, linestyle="-", zorder=2,
                       label="$-2$ (diagonal)" if a == "+1" else None)
    # Adjacent edges (the 0 pairs).
    edge_pairs = [("+1", "+0"), ("+0", "-0"), ("-0", "-1"), ("-1", "+1")]
    for a, b in edge_pairs:
        x0, y0 = state_pos[a]
        x1, y1 = state_pos[b]
        ax_corner.plot([x0, x1], [y0, y1], color="#888",
                       linewidth=1.4, linestyle=":", zorder=1,
                       label="$0$ (adjacent)" if a == "+1" else None)
    ax_corner.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45),
                      fontsize=9, frameon=False, ncol=2)

    fig.suptitle("The 4-state alphabet: geometry of the 2-bit register",
                 fontsize=13, fontweight="bold", y=0.96)
    fig.savefig(FIGURES / "fig_02_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("  Saved fig_02_heatmap.png")


# ── Fig 3 (chapter 3.2): SiLU with phi-boundary markers ────────────────

def fig_03_phi_gate():
    x = np.linspace(-4, 4, 1000)
    sig = 1.0 / (1.0 + np.exp(-x))
    silu = sig * x

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    # Shade the dark-fringe region (deeply negative bias) — context for §3.2.
    ax.axvspan(-4, -1.5, color=MUTED, alpha=0.18)
    ax.text(-2.75, 3.95,
            "dark-fringe region (deep negative bias) —\n"
            r"$97.1\%$ of output energy in the construction",
            ha="center", va="top", fontsize=8.5, color=INK,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=PAPER,
                       edgecolor=MUTED, linewidth=0.6, alpha=0.95))

    # Main SiLU curve.
    ax.plot(x, silu, color=INK, linewidth=2.2, label=r"SiLU$(x) = \sigma(x)\cdot x$")

    # phi-boundary markers.
    log_phi = math.log(PHI)
    sig_pos = 1 / (1 + math.exp(-log_phi))
    sig_neg = 1 / (1 + math.exp(log_phi))

    for xp, sval, label, color, anchor in [
        (+log_phi, sig_pos, r"$\sigma(+\log\varphi) = 1/\varphi \approx 0.6180$",
         GOLD_DARK, ("right", "bottom")),
        (-log_phi, sig_neg, r"$\sigma(-\log\varphi) = 1/\varphi^2 \approx 0.3820$",
         TEAL, ("left", "bottom")),
    ]:
        ax.axvline(xp, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
        # Plot the sigma value (not silu value) to make the identity literal.
        ax.scatter([xp], [sval], s=90, color=color, zorder=5, edgecolor=PAPER,
                    linewidth=1.2)
        ha, va = anchor
        dx = +0.3 if ha == "left" else -0.3
        ax.annotate(label, xy=(xp, sval),
                    xytext=(xp + dx * 4.0, sval + 1.3),
                    fontsize=10, color=color,
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=0.9, shrinkA=2, shrinkB=4))
        tick_label = r"$-\log\varphi$" if xp < 0 else r"$+\log\varphi$"
        ax.text(xp, -0.55, tick_label,
                ha="center", va="top", fontsize=9, color=color, fontweight="bold")

    # Sigma curve in light tone for context (so the identity is visible).
    ax.plot(x, sig, color=MUTED, linewidth=1.1, linestyle="-",
            alpha=0.85, label=r"$\sigma(x)$")

    ax.axhline(0, color=INK_SOFT, linewidth=0.6)
    ax.axvline(0, color=INK_SOFT, linewidth=0.6)
    ax.set_xlabel(r"gate pre-activation $x$", fontsize=11)
    ax.set_ylabel("output", fontsize=11)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.85, 4.4)
    ax.set_title(r"$\varphi$-boundary gating in SwiGLU",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_03_phi_gate.png", dpi=DPI)
    plt.close(fig)
    print("  Saved fig_03_phi_gate.png")


# ── Fig 4 (chapter 4): Attention vs MLP cross-axis routing ─────────────

def fig_04_cross_copy():
    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                          wspace=0.30, left=0.06, right=0.98,
                          top=0.86, bottom=0.10)
    ax_dia = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    # ---- Left: schematic of CROSS_COPY_a→b on a single token. ---------
    ax_dia.set_xlim(0, 6.0)
    ax_dia.set_ylim(0, 4.0)
    ax_dia.axis("off")
    ax_dia.set_title("What is CROSS_COPY$_{a \\to b}$?",
                     fontsize=11, fontweight="bold", pad=4)

    # Two axis blocks side by side; each is a stack of 3 dimensions
    # (sign, mag, any_state) shown as small rectangles.
    blk_w = 1.1
    blk_h = 0.55
    sub_h = blk_h
    blk_x = [0.6, 4.3]
    blk_y = 1.6
    blk_color = [COL_AXIS_A, COL_AXIS_B]
    blk_label = ["axis $a$", "axis $b$"]
    sub_label = ["sign", "mag", "any"]

    for i, x in enumerate(blk_x):
        for k, lab in enumerate(sub_label):
            ax_dia.add_patch(FancyBboxPatch(
                (x, blk_y + (2 - k) * (sub_h * 0.55) - 0.2),
                blk_w, sub_h * 0.55,
                boxstyle="round,pad=0.02",
                facecolor=blk_color[i], alpha=0.25,
                edgecolor="black", linewidth=1.0))
            ax_dia.text(x + blk_w / 2,
                         blk_y + (2 - k) * (sub_h * 0.55) - 0.2 + sub_h * 0.275,
                         lab, ha="center", va="center", fontsize=8.5,
                         color="#222")
        ax_dia.text(x + blk_w / 2, blk_y - 0.4, blk_label[i],
                     ha="center", va="top", fontsize=10, fontweight="bold",
                     color=blk_color[i])

    # State of axis a (above its block).  State of axis b before/after
    # (also above its block).  Kept short so they don't collide with the
    # explanatory line above.
    ax_dia.text(blk_x[0] + blk_w / 2, blk_y + 1.55,
                 r"state $= +0$", ha="center", fontsize=9.5,
                 color=blk_color[0], fontweight="bold")
    ax_dia.text(blk_x[1] + blk_w / 2, blk_y + 1.55,
                 r"$-1 \to +0$", ha="center", fontsize=9.5,
                 color=blk_color[1], fontweight="bold")

    # The MLP box and the routing arrow.
    mlp_box = FancyBboxPatch((2.15, blk_y + 0.05), 1.6, 0.9,
                              boxstyle="round,pad=0.04",
                              facecolor=GOLD_SOFT, edgecolor=GOLD_DARK,
                              linewidth=1.6)
    ax_dia.add_patch(mlp_box)
    ax_dia.text(2.95, blk_y + 0.65, "SwiGLU MLP",
                 ha="center", va="center", fontsize=10,
                 fontweight="bold", color=GOLD_DARK)
    ax_dia.text(2.95, blk_y + 0.30, "(6 channels)",
                 ha="center", va="center", fontsize=8.5, color=INK_SOFT)

    arr1 = FancyArrowPatch((blk_x[0] + blk_w, blk_y + 0.5),
                            (2.15, blk_y + 0.5),
                            arrowstyle="->", mutation_scale=18,
                            color=blk_color[0], linewidth=1.6)
    arr2 = FancyArrowPatch((2.15 + 1.6, blk_y + 0.5),
                            (blk_x[1], blk_y + 0.5),
                            arrowstyle="->", mutation_scale=18,
                            color=blk_color[1], linewidth=1.6)
    ax_dia.add_patch(arr1)
    ax_dia.add_patch(arr2)

    # SWITCH_FLAG indicator
    ax_dia.text(2.95, blk_y - 0.30,
                 r"SWITCH_FLAG = $+2$  $\Rightarrow$  gate fires",
                 ha="center", fontsize=8.5, color=INK_SOFT, style="italic")
    ax_dia.text(2.95, 3.85,
                  "axis $a$'s state is copied into axis $b$\n"
                  "(content-conditional cross-block routing)",
                  ha="center", va="top", fontsize=9.5, color=INK)

    # ---- Right: pass/fail bar chart. ----------------------------------
    ax_bar.set_title("Pass rate on 32 cross-copy test cases",
                     fontsize=11, fontweight="bold", pad=4)
    cats = ["Attention\nonly",
            "Attention +\n6-channel SwiGLU"]
    pass_n = [0, 32]
    fail_n = [32, 0]
    x = np.arange(2)
    width = 0.55

    bars_pass = ax_bar.bar(x, pass_n, width, color=COL_PASS, edgecolor="black",
                            linewidth=1.0, label="PASS  (margin $+2$)")
    bars_fail = ax_bar.bar(x, fail_n, width, bottom=pass_n,
                            color=COL_FAIL, edgecolor="black", linewidth=1.0,
                            label="FAIL  (margin $0$)")

    # Inside-bar labels: state PASS/FAIL counts explicitly so the colours
    # don't have to do the talking.
    for bp, bf, p, f in zip(bars_pass, bars_fail, pass_n, fail_n):
        cx = bp.get_x() + bp.get_width() / 2
        if p > 0:
            ax_bar.text(cx, p / 2,
                         f"PASS\n{p}/32\n(margin $+2$)",
                         ha="center", va="center",
                         color="white", fontsize=12, fontweight="bold")
        if f > 0:
            ax_bar.text(cx, p + f / 2,
                         f"FAIL\n{f}/32\n(margin $0$)",
                         ha="center", va="center",
                         color="white", fontsize=12, fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(cats, fontsize=10)
    ax_bar.set_ylim(0, 36)
    ax_bar.set_ylabel("test cases (out of 32)", fontsize=10)
    ax_bar.grid(axis="y", alpha=0.25)

    fig.suptitle(r"The attention/MLP boundary: cross-axis routing",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.savefig(FIGURES / "fig_04_cross_copy.png", dpi=DPI)
    plt.close(fig)
    print("  Saved fig_04_cross_copy.png")


# ── Fig 5 (chapter 5): SVA logits — all four cases + residual sidebar ──

def fig_05_sva_logits():
    """Show all four (subject_number x tense) SVA cases on a 2x2 grid,
    with a sidebar showing the residual-stream structure of the
    AGREE_OP position."""

    # 18-dim residual: [NUMBER(6), LEX_CLASS(6), TENSE(6)]
    # block layout: [sign, mag, any_state, flag, flag, flag]
    def block(sign, mag, any_state=1.0, flags=(0, 0, 0)):
        return [sign, mag, any_state, *flags]

    def build_resid(num_sign, lex_sign, tense_sign):
        return np.array(
            block(num_sign, +1) +
            block(lex_sign, +1, flags=(2.0, 0, 0)) +  # LEX block carries an op flag
            block(tense_sign, +1)
        )

    VERBS = {
        "is":   build_resid(+1, -1, +1),
        "are":  build_resid(-1, -1, +1),
        "was":  build_resid(+1, -1, -1),
        "were": build_resid(-1, -1, -1),
    }

    cases = [
        ("boy",  +1, "present", +1, "is"),
        ("boys", -1, "present", +1, "are"),
        ("boy",  +1, "past",    -1, "was"),
        ("boys", -1, "past",    -1, "were"),
    ]

    fig = plt.figure(figsize=(12.5, 6.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.85],
                          left=0.06, right=0.98, top=0.88, bottom=0.08,
                          hspace=0.62, wspace=0.34)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    ax_side = fig.add_subplot(gs[:, 2])

    for ax, (subj, num, tense_name, tense, expected) in zip(axes, cases):
        resid = build_resid(num, -1, tense)
        logits = {n: int(resid @ v) for n, v in VERBS.items()}
        names = list(logits.keys())
        vals = [logits[n] for n in names]
        winner = max(logits, key=logits.get)
        runner_up = sorted(vals, reverse=True)[1]
        margin = max(vals) - runner_up
        colors = [COL_PASS if logits[n] == max(vals) else GRID
                   for n in names]

        bars = ax.bar(names, vals, color=colors, edgecolor="black",
                       linewidth=1.0, width=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f"{v:+d}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.set_title(f"[{subj}, AGREE_BE_{tense_name.upper()}]\n"
                     f"$\\rightarrow$  '{winner}'   (margin $+{margin}$)",
                     fontsize=10.5, fontweight="bold", pad=4)
        ax.set_ylim(0, max(vals) + 2)
        ax.set_ylabel("logit (integer)", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=10)

    # ---- Sidebar: residual-stream structure at the AGREE_OP position. ----
    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(0, 1)
    ax_side.axis("off")
    ax_side.set_title("residual at AGREE_OP\n(after attention)",
                       fontsize=10.5, fontweight="bold")

    blocks = [
        ("NUMBER",    "subject's NUMBER",    "(copied by attention)", COL_AXIS_A),
        ("LEX_CLASS", r"$-1$ (verb class)",   "(committed by op embed)", COL_AXIS_B),
        ("TENSE",     "op's TENSE value",     "(committed by op embed)", COL_AXIS_C),
    ]
    by = 0.72
    bh = 0.18
    for i, (name, val, src, col) in enumerate(blocks):
        y = by - i * (bh + 0.03)
        ax_side.add_patch(FancyBboxPatch((0.05, y), 0.90, bh,
                                          boxstyle="round,pad=0.02",
                                          facecolor=col, alpha=0.20,
                                          edgecolor=col, linewidth=1.2))
        ax_side.text(0.50, y + bh - 0.04, name, ha="center", va="top",
                      fontsize=10, fontweight="bold", color=col)
        ax_side.text(0.50, y + bh / 2 - 0.01, val, ha="center", va="center",
                      fontsize=9, color="#222")
        ax_side.text(0.50, y + 0.02, src, ha="center", va="bottom",
                      fontsize=7.8, color="#555", style="italic")

    ax_side.text(0.50, 0.05,
                  "the 3-tuple uniquely\nselects one of\n{is, are, was, were}",
                  ha="center", va="bottom", fontsize=9, color="#222")

    fig.suptitle("Subject-verb agreement: exact integer logits, all 4 cases",
                 fontsize=13, fontweight="bold", y=0.97)
    fig.savefig(FIGURES / "fig_05_sva_logits.png", dpi=DPI)
    plt.close(fig)
    print("  Saved fig_05_sva_logits.png")


# ── Figs 6, 7, 8 (chapter 8): T5d trajectory, restart, axis timeline ───

# These are produced from the frozen T5d log living in the parent
# olmo2_geometric repo.  We copy/regenerate them here so the t4 paper
# is self-contained.

T5D_LOG = Path(
    "/home/thorin/olmo2_work/olmo2_geometric/experiments/output/t5d_final_v1/log.jsonl"
)


def _load_t5d_events():
    import json
    if not T5D_LOG.exists():
        return None
    return [json.loads(l) for l in T5D_LOG.read_text().splitlines() if l.strip()]


def _t5d_trajectory(events):
    iters = sorted([e for e in events if e.get("event") == "iter"],
                   key=lambda e: e["iteration"])
    xs, vocab, sd_rate, axes, coll = [], [], [], [], []
    last = {"vocab": None, "axes": None, "self_decode": None,
            "largest_collision": None}
    for e in iters:
        m = (e.get("metrics_after") if e.get("result") == "commit"
             else e.get("metrics_before")) or {}
        for k in last:
            if m.get(k) is not None:
                last[k] = m[k]
        if last["vocab"] is None:
            continue
        xs.append(e["iteration"])
        vocab.append(last["vocab"])
        axes.append(last["axes"])
        coll.append(last["largest_collision"] or 0)
        sd = last["self_decode"] or [0, 1]
        sd_rate.append(sd[0] / max(sd[1], 1))
    return xs, vocab, sd_rate, axes, coll


def _t5d_axis_events(events):
    out = []
    for e in events:
        if e.get("event") != "iter":
            continue
        a = e.get("action") or {}
        if a.get("kind") == "propose_axis" and e.get("result") == "commit":
            out.append((e["iteration"], a.get("axis_name", "?"),
                         a.get("target_group", []) or [],
                         e.get("metrics_before", {}),
                         e.get("metrics_after", {})))
    return out


def _t5d_saturation_iter(events):
    for e in events:
        if e.get("event") == "saturation":
            return e["iteration"]
    return None


def fig_06_07_t5d_trajectory():
    events = _load_t5d_events()
    if events is None:
        print("  [skip] T5d log not found at", T5D_LOG)
        return
    xs, vocab, sd_rate, axes, coll = _t5d_trajectory(events)
    axis_pts = [(it, name) for (it, name, *_rest) in _t5d_axis_events(events)]
    sat_iter = _t5d_saturation_iter(events)

    # Fig 6: full trajectory.
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.55],
                            wspace=0.30, hspace=0.32,
                            left=0.06, right=0.98, top=0.90, bottom=0.08)
    ax_v  = fig.add_subplot(gs[0, 0])
    ax_sd = fig.add_subplot(gs[0, 1], sharex=ax_v)
    ax_ax = fig.add_subplot(gs[1, 0], sharex=ax_v)
    ax_co = fig.add_subplot(gs[1, 1], sharex=ax_v)
    ax_legend = fig.add_subplot(gs[:, 2])
    ax_legend.axis("off")

    axis_line = MUTED
    sat_line  = INK
    ax_v.plot(xs, vocab, color=INK, lw=1.6)
    ax_v.set_ylabel("vocab size")
    ax_v.set_title("Vocabulary growth")
    ax_v.grid(alpha=0.3)

    ax_sd.plot(xs, [r * 100 for r in sd_rate], color=COL_PASS, lw=1.6)
    ax_sd.set_ylabel("self-decode (%)")
    ax_sd.set_title("Self-decode rate")
    ax_sd.set_ylim(0, 100)
    ax_sd.grid(alpha=0.3)

    ax_ax.step(xs, axes, where="post", color=GOLD_DARK, lw=1.6)
    ax_ax.set_ylabel("# semantic axes")
    ax_ax.set_xlabel("iteration")
    ax_ax.set_title("Axis count (step changes = autonomous discoveries)")
    ax_ax.grid(alpha=0.3)

    ax_co.plot(xs, coll, color=VIOLET, lw=1.6)
    ax_co.set_ylabel("largest collision group")
    ax_co.set_xlabel("iteration")
    ax_co.set_title("Largest collision (smaller = better)")
    ax_co.grid(alpha=0.3)

    for axp in (ax_v, ax_sd, ax_ax, ax_co):
        for it, _ in axis_pts:
            axp.axvline(it, color=axis_line, lw=0.6, alpha=0.6,
                         linestyle="--", zorder=0)
        if sat_iter is not None:
            axp.axvline(sat_iter, color=sat_line, lw=1.0, alpha=0.7, zorder=0)

    if sat_iter is not None:
        ax_v.text(sat_iter, max(vocab) * 0.98, " SATURATION ",
                   fontsize=8, va="top", ha="right", color=sat_line,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=PAPER,
                              edgecolor=sat_line, lw=0.5))

    ax_legend.set_title("Auto-discovered axes", fontsize=11,
                          loc="left", fontweight="bold")
    rows = ["#   iter   axis name", "─" * 38]
    for i, (it, name) in enumerate(axis_pts, 1):
        rows.append(f"{i:>2}  {it:>5d}   {name}")
    rows.append("")
    rows.append("Vertical dashed lines on each panel")
    rows.append("mark these events.  Solid black line")
    rows.append("= saturation auto-pause (rc=7).")
    ax_legend.text(0.0, 0.95, "\n".join(rows), family="monospace",
                    fontsize=9, va="top", ha="left",
                    transform=ax_legend.transAxes)

    fig.suptitle("T5d auto-loop trajectory — 22,432 iterations to saturation",
                 fontsize=12)
    fig.savefig(FIGURES / "fig_06_t5d_trajectory.png", dpi=DPI)
    plt.close(fig)
    print("  Saved fig_06_t5d_trajectory.png")

    # Fig 7: restart phase zoom.
    fig2, ax = plt.subplots(figsize=(11, 5))
    mask = [i for i, x in enumerate(xs) if x >= 19000]
    if mask:
        x2 = [xs[i] for i in mask]
        sd2 = [sd_rate[i] * 100 for i in mask]
        v2  = [vocab[i] for i in mask]
        ax.plot(x2, sd2, color=TEAL, lw=1.8, label="self-decode (%)")
        ax2 = ax.twinx()
        ax2.plot(x2, v2, color=INK, lw=1.8, alpha=0.85, label="vocab")
        ax.set_xlabel("iteration")
        ax.set_ylabel("self-decode (%)", color=TEAL)
        ax2.set_ylabel("vocab size", color=INK)
        ax.set_ylim(50, 100)
        ax.grid(alpha=0.3)
        phase_axes = [(it, name) for it, name in axis_pts if it >= 19000]
        if phase_axes:
            for it, _ in phase_axes:
                ax.axvline(it, color=axis_line, lw=0.7,
                           alpha=0.5, linestyle="--")
            it_min = min(it for it, _ in phase_axes)
            it_max = max(it for it, _ in phase_axes)
            ax.annotate(
                "4 axes added\n(personhood, initiative_authority,\n"
                " group_affiliation_strength, status_origin)",
                xy=((it_min + it_max) / 2, 92),
                xytext=((it_min + it_max) / 2 + 700, 78),
                fontsize=9, color=INK, ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.7),
            )
        if sat_iter is not None:
            ax.axvline(sat_iter, color=sat_line, lw=1.2, alpha=0.8)
            ax.annotate("SATURATION (rc=7)", xy=(sat_iter, 55),
                        xytext=(sat_iter - 200, 55), fontsize=9,
                        color=sat_line, ha="right",
                        arrowprops=dict(arrowstyle="->", color=sat_line))
        ax.set_title("Restart phase: dictionary filter + adaptive axis-trigger\n"
                     "4 axes discovered, vocab 806→1438, self-decode 55%→84%")
    fig2.tight_layout()
    fig2.savefig(FIGURES / "fig_07_t5d_restart.png", dpi=DPI)
    plt.close(fig2)
    print("  Saved fig_07_t5d_restart.png")


def fig_08_t5d_axes():
    """Horizontal-bar timeline: each of the 7 auto-discovered axes
    plotted by iteration of discovery, with target collision group and
    self-decode delta."""
    events = _load_t5d_events()
    if events is None:
        print("  [skip] T5d log not found at", T5D_LOG)
        return
    axis_pts = _t5d_axis_events(events)
    if not axis_pts:
        print("  [skip] no axis events found")
        return

    fig, ax = plt.subplots(figsize=(11, 5.4))
    iters = [it for it, *_ in axis_pts]
    names = [n for _, n, *_ in axis_pts]
    targets = [t for _, _, t, *_ in axis_pts]
    deltas = []
    for _, _, _, mb, ma in axis_pts:
        sd_b = (mb.get("self_decode") or [0, 1])[0]
        sd_a = (ma.get("self_decode") or [0, 1])[0]
        deltas.append(sd_a - sd_b)

    y = np.arange(len(axis_pts))
    bars = ax.barh(y, deltas, color=GOLD, edgecolor=INK, linewidth=0.8,
                    height=0.55)
    for i, (it, name, tgt) in enumerate(zip(iters, names, targets)):
        ax.text(deltas[i] + 2, i,
                 f"  +{deltas[i]} self-decoded   |   "
                 f"target = {{{', '.join(tgt)}}}",
                 va="center", fontsize=8.5, color=INK)
        ax.text(-2, i, f"iter {it:>5d}", va="center", ha="right",
                 fontsize=8.5, color=INK_SOFT)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("self-decode count gained when axis was added", fontsize=10)
    ax.set_xlim(-50, max(deltas) * 2.0)
    ax.set_title("The 7 auto-discovered axes (in order of discovery)",
                  fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_08_t5d_axes.png", dpi=DPI)
    plt.close(fig)
    print("  Saved fig_08_t5d_axes.png")


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures for T4 Constructive Transformer paper...\n")
    fig_01_architecture()
    fig_02_heatmap()
    fig_03_phi_gate()
    fig_04_cross_copy()
    fig_05_sva_logits()
    fig_06_07_t5d_trajectory()
    fig_08_t5d_axes()
    print(f"\nAll figures saved to {FIGURES}/")
    for f in sorted(FIGURES.glob("fig_*.png")):
        print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
