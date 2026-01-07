from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


def set_paper_style() -> None:
    """A conservative matplotlib style (small fonts, clean lines)."""
    mpl.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def make_figure(width: str = "single", *, height: float | None = None, nrows: int = 1, ncols: int = 1):
    """Create a figure with typical paper column widths.

    Parameters
    ----------
    width
        "single" (~3.5 in) or "double" (~7.2 in)
    height
        Height in inches. If None, uses a default aspect ratio.
    """
    set_paper_style()
    if width == "single":
        w = 3.5
    elif width == "double":
        w = 7.2
    else:
        raise ValueError("width must be 'single' or 'double'")
    h = (w * 0.75) if height is None else float(height)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), constrained_layout=True)
    return fig, axes


def save_figure(fig, outpath: str | Path) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax, label: str, *, x: float = -0.08, y: float = 1.02) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", va="bottom", ha="right")
