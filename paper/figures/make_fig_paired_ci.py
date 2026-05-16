"""Generate paper/figures/fig_paired_ci.{pdf,png}.

Side-by-side paired 95% bootstrap CIs:

* Left panel  — Phase 2 (decision/006): every Sinkhorn variant vs.
  Hungarian baseline. All bars sit below zero, hence the
  Hungarian-beats-Sinkhorn headline.
* Right panel — Phase 3 (decision/007 follow-up): Oklch hue-rotation
  pool augmentation vs. no-aug, for the two matchers. Hungarian gains,
  Sinkhorn loses.

Run: ``python paper/figures/make_fig_paired_ci.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
PHASE2 = REPO / "experiments" / "results" / "phase2_baseline_2026-05-16.json"
PHASE3_TIGHT = REPO / "experiments" / "results" / "phase3_baseline_tight_2026-05-16.json"
PHASE3 = REPO / "experiments" / "results" / "phase3_baseline_2026-05-16.json"
OUT_DIR = REPO / "paper" / "figures"

# Color palette: Okabe-Ito colour-blind-safe scheme.
#   NEG (vermillion) = bars whose 95% CI is strictly below zero.
#   POS (bluish green) = bars whose 95% CI is strictly above zero.
#   TIE (neutral grey) = bars whose CI straddles zero.
NEG = "#D55E00"
POS = "#009E73"
TIE = "#7f8c8d"


def _bar_color(lo: float, hi: float) -> str:
    if hi < 0:
        return NEG
    if lo > 0:
        return POS
    return TIE


def _annotation(lo: float, hi: float) -> str:
    if hi < 0:
        return "WORSE"
    if lo > 0:
        return "BEATS"
    return "TIE"


def _load_phase2() -> list[tuple[str, float, float, float]]:
    d = json.loads(PHASE2.read_text())
    ci = d["vs_hungarian_bootstrap_ci"]
    out: list[tuple[str, float, float, float]] = []
    for name, stats in ci.items():
        out.append(
            (
                name,
                float(stats["mean_diff_vs_hungarian"]),
                float(stats["ci95_lo"]),
                float(stats["ci95_hi"]),
            )
        )
    out.sort(key=lambda r: r[1])
    return out


def _bootstrap_diff_ci(
    x: np.ndarray, y: np.ndarray, *, n: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap CI for mean(x) - mean(y); x[i] and y[i] are paired runs."""
    rng = np.random.default_rng(seed)
    diffs = x - y
    boots = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, diffs.shape[0], size=diffs.shape[0])
        boots[i] = float(diffs[idx].mean())
    return float(diffs.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _phase3_pairs(path: Path) -> dict[str, tuple[float, float, float]]:
    d = json.loads(path.read_text())
    rows = d["results"]
    by_cond_keyed: dict[str, dict[tuple[int, int], float]] = {}
    for r in rows:
        by_cond_keyed.setdefault(r["name"], {})[(int(r["seed"]), int(r["target_idx"]))] = float(
            r["final_ssim"]
        )
    out: dict[str, tuple[float, float, float]] = {}
    for matcher in ("hungarian", "sinkhorn"):
        no_aug = by_cond_keyed[f"{matcher}_no_aug"]
        with_aug = by_cond_keyed[f"{matcher}_oklch_aug"]
        common = sorted(set(no_aug) & set(with_aug))
        y = np.array([no_aug[k] for k in common], dtype=np.float64)
        x = np.array([with_aug[k] for k in common], dtype=np.float64)
        out[matcher] = _bootstrap_diff_ci(x, y)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    phase2_rows = _load_phase2()
    phase3_pairs = _phase3_pairs(PHASE3_TIGHT)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(12.0, 5.6), gridspec_kw={"width_ratios": [3, 1.4]}
    )

    # ----- left: Phase 2 ε-sweep -----
    labels = [r[0].replace("_unique=0", "").replace("_unique=1", "*") for r in phase2_rows]
    means = np.array([r[1] for r in phase2_rows])
    los = np.array([r[2] for r in phase2_rows])
    his = np.array([r[3] for r in phase2_rows])
    yidx = np.arange(len(means))
    colors = [_bar_color(lo, hi) for lo, hi in zip(los, his, strict=True)]
    ax_l.errorbar(
        means,
        yidx,
        xerr=[means - los, his - means],
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=2,
    )
    ax_l.scatter(means, yidx, c=colors, s=40, zorder=3, edgecolor="black", linewidth=0.5)
    ax_l.axvline(0.0, color="black", lw=0.7)
    ax_l.set_yticks(yidx)
    ax_l.set_yticklabels(labels, fontsize=8)
    ax_l.set_xlabel("ΔSSIM vs. Hungarian (mean + 95% paired bootstrap CI)")
    ax_l.set_title("Phase 2 — every Sinkhorn variant loses to Hungarian", fontsize=11)
    ax_l.grid(axis="x", alpha=0.3)
    ax_l.invert_yaxis()

    # ----- right: Phase 3 oklch-aug split -----
    matchers = ["hungarian", "sinkhorn"]
    means3 = np.array([phase3_pairs[m][0] for m in matchers])
    los3 = np.array([phase3_pairs[m][1] for m in matchers])
    his3 = np.array([phase3_pairs[m][2] for m in matchers])
    colors3 = [_bar_color(lo, hi) for lo, hi in zip(los3, his3, strict=True)]
    annots3 = [_annotation(lo, hi) for lo, hi in zip(los3, his3, strict=True)]
    yidx3 = np.arange(len(matchers))
    ax_r.errorbar(
        means3,
        yidx3,
        xerr=[means3 - los3, his3 - means3],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
    )
    ax_r.scatter(means3, yidx3, c=colors3, s=80, zorder=3, edgecolor="black", linewidth=0.7)
    for i, (m, a) in enumerate(zip(means3, annots3, strict=True)):
        ax_r.text(
            m,
            yidx3[i] - 0.18,
            a,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=colors3[i],
        )
    ax_r.axvline(0.0, color="black", lw=0.7)
    ax_r.set_yticks(yidx3)
    ax_r.set_yticklabels(matchers)
    ax_r.set_xlabel("ΔSSIM (oklch_aug − no_aug)")
    ax_r.set_title(
        "Phase 3 — oklch-aug splits the matchers (N=32 = 8 seeds × 4 targets)",
        fontsize=10,
    )
    ax_r.grid(axis="x", alpha=0.3)
    ax_r.invert_yaxis()

    fig.suptitle("Paired bootstrap 95% CIs — Hungarian dominates, oklch-aug splits", fontsize=12)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_paired_ci.{ext}"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"SAVED {out}")


if __name__ == "__main__":
    main()
