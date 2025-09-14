"""Figure saving utilities extracted from ds.py.

Usage:
  from fig_utils import set_dataset_save_dir, save_fig, set_fig_formats
  set_fig_formats(["png"])  # default already png
  set_dataset_save_dir("highly_nonlinear")
  ... plot ...
  save_fig("plot.png")

Design:
- Only manages paths and formats; NO plotting logic.
- FIG_FORMATS always non-empty; defaults to ["png"].
"""
from __future__ import annotations
import os
from datetime import datetime
import matplotlib.pyplot as plt

RUN_TAG = datetime.now().strftime('%Y%m%d_%H%M%S')
FIG_SAVE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
CURRENT_FIG_SAVE_DIR: str | None = None
FIG_FORMATS = ["png"]  # mutable list

def set_fig_formats(formats):
    """Configure output formats. formats: iterable of strings (e.g. ['png','svg']).
    Empty / invalid input falls back to ['png']. Duplicates removed preserving order.
    """
    global FIG_FORMATS
    seen = set()
    cleaned = []
    for f in (formats or []):
        f = str(f).lower().strip()
        if not f:
            continue
        if f not in {"png","svg","pdf"}:  # restrict to supported for simplicity
            continue
        if f in seen:
            continue
        seen.add(f)
        cleaned.append(f)
    if not cleaned:
        cleaned = ["png"]
    FIG_FORMATS = cleaned
    return FIG_FORMATS

def set_dataset_save_dir(dataset_name: str):
    """Set current dataset figure directory and return its path."""
    global CURRENT_FIG_SAVE_DIR
    dataset_tag = f"{dataset_name}_{RUN_TAG}"
    CURRENT_FIG_SAVE_DIR = os.path.join(FIG_SAVE_ROOT, dataset_tag)
    return CURRENT_FIG_SAVE_DIR

def _ensure_fig_dir():
    if CURRENT_FIG_SAVE_DIR is None:
        raise ValueError("Call set_dataset_save_dir() before saving figures.")
    os.makedirs(CURRENT_FIG_SAVE_DIR, exist_ok=True)

def save_fig(name: str, dpi: int = 300):
    """Save current matplotlib figure to the configured formats.

    name can end with .png; base name is derived automatically.
    """
    _ensure_fig_dir()
    safe = name.replace(' ', '_')
    base = safe[:-4] if safe.lower().endswith('.png') else safe
    assert CURRENT_FIG_SAVE_DIR is not None, "Figure directory not set."
    for fmt in FIG_FORMATS:
        out_path = os.path.join(CURRENT_FIG_SAVE_DIR, f"{base}.{fmt}")
        try:
            plt.savefig(out_path, format=fmt, dpi=dpi, bbox_inches='tight')
            print(f"[Saved {fmt.upper()}] {out_path}")
        except Exception as e:
            print(f"[Warn] Save failed ({fmt}): {e}")

__all__ = [
    'RUN_TAG', 'FIG_SAVE_ROOT', 'CURRENT_FIG_SAVE_DIR', 'FIG_FORMATS',
    'set_fig_formats', 'set_dataset_save_dir', 'save_fig'
]
