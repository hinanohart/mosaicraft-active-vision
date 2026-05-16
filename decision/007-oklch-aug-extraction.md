# 007 — Oklch hue-rotation pool augmentation extracted to `oklch-aug`

**Status:** REPORTED — 2026-05-16.
**Trigger:** 3-agent review (`document-specialist` / `architect` / `critic`)
of the most innovative transferable technique surviving in the
`mosaicraft` family after `decision/006-phase2-findings.md` falsified
the Sinkhorn-OT advantage.

## What the agents found

1. **`document-specialist`** — 2025-2026 survey returned zero hits
   for "Oklch hue rotation × bipartite-matching pool augmentation".
   `albumentations`, `albumentationsX`, `kornia`, `torchvision`, and
   DALI all lack Oklab as a first-class color space. The closest
   adjacent work is `HVI: A New Color Space for Low-light Enhancement`
   (CVPR 2025), which proposes its own perceptual space and confirms
   that "knowingly perceptually-uniform color space → PSNR/SSIM gain"
   is a referee-acceptable claim. Hot zones for 2026 are
   **VLA / diffusion policy color augmentation** (`pi0.5`, OpenVLA,
   RoboTwin 2.0, RoboLight `arXiv:2603.04249`) and
   **differentiable NBV via OT plan gradients**
   (OTPR `arXiv:2502.12631`).

2. **`architect`** — Verified with `file:line` evidence in
   `mosaicraft/color_augment.py:76-258` and `mosaicraft/color.py:48-132`
   that the technique decomposes into:
   - `rotate_hue_oklch` (lines 76-131) — pure numpy, cv2-free.
   - `expand_color_variants` (lines 146-258) — pulls in `TileSet`,
     `extract_features`, and cv2.
   - Oklab conversion group (`color.py:48-132`) — cv2-free already.
   Conclusion: a "purely numpy `HueRotatePool` + `rotate_hue_oklch` +
   Oklab transforms + Oklab metric" extraction is ~200 lines and stays
   dependency-light. The `expand_color_variants` `TileSet` glue is
   mosaicraft-specific and stays in mosaicraft.

3. **`critic`** — Reverse-proposed three alternative paths and flagged
   that **`albumentations` itself is unmaintained as of 2025-06**
   (migration in progress to `AlbumentationsX`), so a direct PR is a
   low-leverage move. The high-leverage moves are
   (a) ship the extraction as its own PyPI library, then
   (b) write an `arXiv` preprint that bundles
   `decision/006` + Oklch pool aug into a single "honest negative
   result + transferable Oklab tooling" submission targeting the
   *Insights from Negative Results* / OTML workshops, and
   (c) attempt a `PythonOT/POT` PR `#724` rescue using the
   log-domain stability test scaffolding listed as defensible in
   `decision/006-phase2-findings.md:42-46`.

## Decision

**3-layer architecture, in this order:**

```
Layer 1 — oklch-aug PyPI lib (new repo, /home/runza/oss/oklch-aug/)
  rotate_hue_oklch, HueRotatePool, rgb<->oklab, bgr<->oklab, oklab_distance
  numpy core, torch optional, AlbumentationsX/kornia adapters as extras
Layer 2 — this repo (mosaicraft-active-vision)
  Sinkhorn-OT harness retained, honest negative result retained
  cost.py rewired to import from oklch-aug (PyPI install path)
  Phase-3 benchmark adds Oklch-pool × NBV ablation on real images
Layer 3 — external contributions
  arXiv preprint (Insights workshop / OTML target)
  PythonOT/POT PR #724 rescue (log-domain stability scaffolding)
```

This decision is consistent with the project's hard constraints:

- `decision/003` ("no Hungarian fallback") is not rescinded — the
  matcher in this repo stays Sinkhorn-only. The Oklch extraction lives
  in a separate library and is orthogonal to the matcher question.
- Append-only rule for signed decisions is honored — `decision/003`
  and `decision/006` are unchanged; this is a new file.
- R17 (既存修正優先): the extraction reuses existing mosaicraft code
  rather than rewriting it; the new library is a literal port with
  cv2 stripped and a `channel_order` kwarg added.

## Why this is the right move (one paragraph)

The Sinkhorn-OT thesis lost on this benchmark, but the implementation
artifacts around it (log-domain stability, Oklab perceptual cost,
saliency-as-marginal framing, the Oklch tile-pool expander) are
individually useful in adjacent fields where matching / pooling /
color augmentation are first-class problems but Oklab is not yet
standard tooling. Shipping the Oklab tooling as a standalone library
turns a single-purpose research repo into a building block other
researchers can pick up without inheriting our negative result. The
negative result itself becomes a preprint with sharper specificity
("Hungarian wins for photomosaic-NBV at the 50×50 perceptual-cost
scale") rather than a buried `decision/` doc.

## How to apply

1. Treat `oklch-aug` as the canonical home for `bgr_to_oklab`,
   `oklab_to_bgr`, `rotate_hue_oklch`, `HueRotatePool`, and
   `oklab_distance`. Do not re-implement these locally.
2. `cost.py:55` will switch from
   `from mosaicraft.color import bgr_to_oklab` to
   `from oklch_aug import bgr_to_oklab` once `oklch-aug` is published.
   The math is bit-identical (verified by porting the same Ottosson
   coefficients without modification); the switch is import-only.
3. The Phase-3 benchmark (`experiments/benchmark_phase3.py`) is the
   first downstream consumer — it uses `HueRotatePool` to augment a
   tile pool before measuring NBV gain.
4. Decision/003's "no Hungarian fallback" still applies inside this
   repo; the new library is matcher-agnostic and may be used by
   Hungarian-based pipelines elsewhere.

## Signature

User signature not requested — this is an extraction record, not a
new core adoption. The technique was already accepted upstream in
`mosaicraft`; this file documents the move to a standalone home and
the agent-debate provenance that motivated it.
