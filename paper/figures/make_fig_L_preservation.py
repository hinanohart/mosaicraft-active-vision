"""Generate paper/figures/fig_L_preservation.{pdf,png}.

Diagnostic: shape of the per-pixel |ΔL*| distribution after Oklch hue
rotation at every angle in ``HueRotatePool``'s default schedule on a
**real, high-chroma** sample image. In float, Oklch hue rotation is
L-preserving by construction; deviations come from two sources, neither
of which is sub-LSB:

1. sRGB gamut clipping. Rotating chroma at constant Oklab L can push
   the result outside the displayable sRGB cube; the implementation
   clips, which silently moves the resulting Oklab L. Most of the long
   tail in the histogram below comes from this.
2. uint8 round-trip quantisation. Small, dominated by (1) on this
   sample.

The headline number is therefore *not* a clean upper bound but the
empirical max |ΔL*| per angle, which we report in each subplot title.

Run: ``python paper/figures/make_fig_L_preservation.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from oklch_aug import bgr_to_oklab, rotate_hue_oklch
from oklch_aug.pool import DEFAULT_HUE_SCHEDULE

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    img_path = REPO / "external" / "mosaicraft" / "docs" / "images" / "tiles_sample.jpg"
    if not img_path.is_file():
        print(f"ERROR: sample image missing at {img_path}", file=sys.stderr)
        sys.exit(1)

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"ERROR: cv2.imread returned None for {img_path}", file=sys.stderr)
        sys.exit(1)

    L_orig = bgr_to_oklab(bgr)[..., 0]

    schedule = list(DEFAULT_HUE_SCHEDULE)

    # Two-pass: compute first, then plot so x-range can be data-driven.
    dLs: list[np.ndarray] = []
    for angle in schedule:
        rotated_bgr = rotate_hue_oklch(bgr, hue_shift_deg=angle, channel_order="bgr")
        dLs.append((bgr_to_oklab(rotated_bgr)[..., 0] - L_orig).ravel())

    overall_max = max(float(np.abs(d).max()) for d in dLs)
    # Tail-aware symmetric range; round up to the next sensible step.
    span = float(np.ceil(overall_max * 100.0) / 100.0)

    fig, axes = plt.subplots(1, len(schedule), figsize=(3.0 * len(schedule), 2.8), sharey=True)
    if len(schedule) == 1:
        axes = [axes]

    for ax, angle, dL in zip(axes, schedule, dLs, strict=True):
        ax.hist(dL, bins=120, range=(-span, span), color="#3a6ea5", edgecolor="none")
        ax.axvline(0.0, color="black", lw=0.6)
        max_abs = float(np.abs(dL).max())
        p99 = float(np.quantile(np.abs(dL), 0.99))
        ax.set_title(
            f"+{int(angle)}° hue\nmax|ΔL*|={max_abs:.3f}, 99th-pct|ΔL*|={p99:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("ΔL* per pixel")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("pixel count")
    fig.suptitle(
        "Oklab L deviation under HueRotatePool (sample: mosaicraft tiles_sample.jpg)\n"
        "tail driven by sRGB gamut clipping at constant Oklab L; uint8 round-trip is sub-LSB",
        fontsize=10,
    )
    fig.tight_layout()

    out_dir = REPO / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"fig_L_preservation.{ext}"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"SAVED {out}")


if __name__ == "__main__":
    main()
