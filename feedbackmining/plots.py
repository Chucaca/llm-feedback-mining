"""Charts for docs/img (matplotlib, Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _ax(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def top_features(rows: list[tuple[str, int, int]], path: Path, top: int = 15) -> None:
    """rows: (feature name, predicted deals, ground-truth deals), sorted desc by predicted."""
    rows = rows[:top][::-1]
    names = [r[0] for r in rows]
    fig, ax = _ax((9, 0.36 * len(rows) + 1.4))
    ax.barh(names, [r[1] for r in rows], color=BLUE, height=0.62, label="extracted by pipeline")
    ax.scatter([r[2] for r in rows], names, color=ORANGE, s=36, zorder=3, label="ground truth",
               edgecolors=SURFACE, linewidths=1.5)
    for i, r in enumerate(rows):
        ax.text(max(r[1], r[2]), i, f"  {r[1]}", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("lost deals mentioning the missing feature", color=INK2)
    ax.set_title("Top missing features behind lost deals (synthetic data)", loc="left", fontsize=11, color=INK)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def funnel(stages: list[tuple[str, int]], path: Path) -> None:
    fig, ax = _ax((8.5, 2.9))
    ax.grid(axis="x", visible=False)
    labels = [s[0] for s in stages][::-1]
    vals = [s[1] for s in stages][::-1]
    total = stages[0][1]
    ax.barh(labels, vals, color=BLUE, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:,} ({v / total:.0%})", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, total * 1.22)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Funnel: notes -> flagged by Stage 1 -> specific features from Stage 2 (synthetic)",
                 loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
