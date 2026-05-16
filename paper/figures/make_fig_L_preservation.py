"""Generate paper/figures/fig_L_preservation.{pdf,png}.

Diagnostic: how flat is the per-pixel |ΔL*| distribution after Oklch
hue rotation at every angle in ``HueRotatePool``'s default schedule?
The claim (`paper/outline.md` C4) is that the rotation is
*L-preserving* — the only deviations should come from sRGB → linear →
Oklab → linear → sRGB round-trip and uint8 quantisation, both
sub-perceptual.

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


def main() -> None:
    img_path = Path(
        "/home/runza/oss/mosaicraft-active-vision/external/mosaicraft/docs/images/tiles_sample.jpg"
    )
    if not img_path.is_file():
        print(f"ERROR: sample image missing at {img_path}", file=sys.stderr)
        sys.exit(1)

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"ERROR: cv2.imread returned None for {img_path}", file=sys.stderr)
        sys.exit(1)

    L_orig = bgr_to_oklab(bgr)[..., 0]  # shape (H, W)

    schedule = list(DEFAULT_HUE_SCHEDULE)
    fig, axes = plt.subplots(1, len(schedule), figsize=(3.0 * len(schedule), 2.6), sharey=True)
    if len(schedule) == 1:
        axes = [axes]

    for ax, angle in zip(axes, schedule, strict=True):
        rotated_bgr = rotate_hue_oklch(bgr, hue_shift_deg=angle, channel_order="bgr")
        L_rot = bgr_to_oklab(rotated_bgr)[..., 0]
        dL = (L_rot - L_orig).ravel()
        ax.hist(dL, bins=80, range=(-0.02, 0.02), color="#3a6ea5", edgecolor="none")
        ax.axvline(0.0, color="black", lw=0.6)
        ax.set_title(f"+{int(angle)}° hue\nmax|ΔL*|={float(np.abs(dL).max()):.4f}")
        ax.set_xlabel("ΔL* per pixel")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("pixel count")
    fig.suptitle(
        "Oklab L-preservation under HueRotatePool (sample: mosaicraft tiles_sample.jpg)",
        fontsize=11,
    )
    fig.tight_layout()

    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"fig_L_preservation.{ext}"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"SAVED {out}")


if __name__ == "__main__":
    main()
