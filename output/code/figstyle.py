"""
Shared visual identity for TruthSpace paper figures.
Imported as 'figstyle' via a small shim to avoid the fig*.py glob.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)

INK        = "#1B2D4A"
INK_SOFT   = "#4A5A75"
GOLD       = "#C9962B"
GOLD_DARK  = "#8C6517"
GOLD_SOFT  = "#F0DDA8"
RED        = "#C0392B"
TEAL       = "#1F8A82"
VIOLET     = "#6E3FA3"
SAND       = "#E9DFC9"
PAPER      = "#FBFAF6"
GRID       = "#D8DEE9"
MUTED      = "#9AA4B2"

SEQ = ["#1F4E79", "#2E75B6", "#5BAEDB", "#94C9E8",
       "#D6BD8B", "#C9962B", "#8C6517"]
CAT = [INK, GOLD, RED, TEAL, VIOLET, "#D67D2C", "#5B7A99", MUTED]


def apply_style() -> None:
    plt.rcdefaults()
    plt.rcParams.update({
        "figure.facecolor":   PAPER,
        "axes.facecolor":     PAPER,
        "savefig.facecolor":  PAPER,
        "savefig.edgecolor":  "none",

        "font.family":        "DejaVu Sans",
        "font.size":          10.5,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.titlepad":      10,
        "axes.labelsize":     10.5,
        "axes.labelcolor":    INK,
        "axes.edgecolor":     INK,
        "axes.linewidth":     1.0,
        "axes.spines.top":    False,
        "axes.spines.right":  False,

        "xtick.color":        INK,
        "ytick.color":        INK,
        "xtick.labelsize":    9.5,
        "ytick.labelsize":    9.5,
        "xtick.major.size":   4,
        "ytick.major.size":   4,

        "legend.frameon":     False,
        "legend.fontsize":    9.5,

        "grid.color":         GRID,
        "grid.linewidth":     0.7,
        "grid.alpha":         0.8,

        "text.color":         INK,
        "mathtext.fontset":   "dejavusans",

        "figure.titlesize":   15,
        "figure.titleweight": "bold",
        "figure.dpi":         110,
        "savefig.dpi":        200,
    })


def soft_grid(ax, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, alpha=0.7, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def clean_spines(ax, keep=("left", "bottom")) -> None:
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    for s in keep:
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.9)


def hide_axes(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def panel_label(ax, text: str, *, x=0.0, y=1.04, **kw) -> None:
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=INK,
            ha="left", va="bottom", **kw)


def caption(ax, text: str, *, y=-0.18, **kw) -> None:
    ax.text(0.5, y, text, transform=ax.transAxes,
            fontsize=9.5, color=INK_SOFT, style="italic",
            ha="center", va="top", **kw)


def save_fig(name: str, *, pad=0.35) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out  = os.path.normpath(os.path.join(here, "..", f"{name}.png"))
    # Strip Software/Source/CreationTime so identical visual output ->
    # identical bytes -> no git churn on rebuilds.
    plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=pad,
                facecolor=PAPER,
                metadata={"Software": None, "Source": None,
                          "CreationTime": None})
    plt.close()
    print(f"saved {os.path.basename(out)}")
