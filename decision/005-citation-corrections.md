# 005 — Citation Corrections (post-publication audit)

**Status:** REPORTED — 2026-05-16
**Author:** independent document-specialist verification pass after the 4-agent debate that produced decisions/001-003.
**Trigger:** internal audit found three citations in decisions/000 and 003 that, on re-verification against primary sources, are either misattributed or paraphrase the source incorrectly.

`PROGRESS.md` records that signed decision docs are append-only, so this file logs the corrections without rewriting 000 / 003. Read this file alongside them.

## Why the original citations were wrong

The four-agent debate that produced decisions/001-003 had a `document-specialist` agent whose primary contribution was citing recent literature to justify the Hungarian→Sinkhorn pivot. The agent's claims were taken as inputs to decisions/000 and 003 without independent re-verification against primary sources. A later audit (2026-05-16, post-publication) fetched the cited papers directly and found the discrepancies below.

This is not a retraction of decision/003 — Sinkhorn-OT remains the matching layer, the differentiability argument still stands, and Cuturi 2013 / Peyré-Cuturi 2019 are correctly cited. What changes is **how this repo describes the prior-art landscape it claims to occupy.**

## Corrections

### C1 — "Rabin 2014" arXiv ID was wrong

- **Where it appeared:** internal agent debate transcript (now superseded). `decision/000-charter.md` L62 correctly cites the **HAL** ID and is not affected.
- **The error:** the debate transcript also referenced `arXiv:1404.3892` as a duplicate ID for the same paper. `arXiv:1404.3892` is actually *"Effective Fluid FLRW Cosmologies of Minimal Massive Gravity"* by N.T. Yilmaz — a cosmology paper unrelated to color transfer.
- **The correct citation:**
  Rabin, J., Ferradans, S., Papadakis, N. (2014). *Adaptive Color Transfer With Relaxed Optimal Transport.* ICIP 2014. HAL: [hal-01002830](https://hal.science/hal-01002830).
- **Impact on decisions:** none — the HAL citation in `decision/000-charter.md` is correct. The arXiv ID never made it into a decision file; it lived only in the debate transcript.

### C2 — "Beyond Hungarian" is real but does **not** argue Pareto-optimality

- **Where it appeared:** `decision/003-matching-algorithm.md` §"Why Hungarian is the wrong tool here" item 4, which states *"Beyond Hungarian (2026-03) shows Hungarian is not Pareto-optimal anymore for assignment problems with marginal constraints — exactly our setting."*
- **The error:** the paper is real (arXiv:2603.08514, *"Beyond Hungarian: Match-Free Supervision for End-to-End Object Detection"*, Qiu et al., 2026-03), but on direct read it does **not** discuss Pareto optimality, and its setting is **not** marginal-constrained assignment. Its argument is the opposite of what we cited: it proposes removing Hungarian entirely in favor of **match-free supervision** (Cross-Attention-based Query Selection) for DETR-style object detectors. The paper's stated objections to Hungarian are "computational overhead and complicates training dynamics," not Pareto-optimality.
- **Impact on decision/003:** the differentiability / GPU / stability arguments (items 1-3 of §"Why Hungarian is the wrong tool here") stand on their own. The "recent literature" item 4 is **withdrawn**. Sinkhorn-OT remains adopted; the empirical Phase-1 result already shows Hungarian wins on the toy (`decision/004`), and the scientific motivation for keeping Sinkhorn is the Phase-2 saliency-as-marginal experiment, not Qiu et al. 2026.

### C3 — GenNBV does not "treat reconstruction error as a black box"

- **Where it appeared:** `decision/000-charter.md` L65-66, *"GenNBV (CVPR 2024) is the dominant NBV baseline on Isaac Gym but treats reconstruction error as a black box, not as a photomosaic placement loss."*
- **The error:** the phrase "black box" does not appear in GenNBV's paper text. GenNBV's reward is **coverage ratio (CR)**, not reconstruction error — so the contrast we want to draw is correct in spirit (GenNBV rewards coverage; we reward photomosaic SSIM gain) but the "black box" framing is our paraphrase, not the paper's own framing.
- **Corrected wording:** GenNBV (Chen et al., CVPR 2024, [arXiv:2402.16174](https://arxiv.org/abs/2402.16174)) rewards **coverage ratio** for active 3D reconstruction. This repo rewards photomosaic SSIM gain, which is a different signal. Whether one signal is preferable is a Phase-2 question, not a settled claim.

### C4 — "Bottosson" → "Ottosson"

- **Where it appeared:** `src/mosaicraft_active_vision/cost.py` L11 and L21 (this commit fixes it).
- **The error:** the Oklab author's name is **Björn Ottosson**, not "Bottosson." The `external/mosaicraft/` submodule has it correct; we introduced the typo when writing the cost.py docstring.

## What this means for the README

`README.md` §"Why this exists" claims the literature lacks "a system that uses the photomosaic reconstruction loss as the NBV signal, with the matching solved by entropic OT whose marginals carry mosaicraft's saliency weights." That claim remains plausible — none of C1-C3 contradicts it. But `decision/000-charter.md` already states this novelty claim is *"unverified until 003-matching-algorithm.md cites concrete search queries with zero hits."* `decision/003` does not list those search queries. So the README will be updated in this commit to add the same "unverified prior-art" caveat that the charter has always required.

## What's still verified

- **Cuturi 2013** ([arXiv:1306.0895](https://arxiv.org/abs/1306.0895)) — Sinkhorn distances. Correct.
- **Peyré-Cuturi 2019** ([arXiv:1803.00567](https://arxiv.org/abs/1803.00567)) — Computational Optimal Transport monograph. Correct.
- **Ottosson 2020** ([blog](https://bottosson.github.io/posts/oklab/)) — Oklab. Correct (author name typo fixed in C4).
- **FisherRF (ECCV 2024)** ([arXiv:2311.17874](https://arxiv.org/abs/2311.17874)) — Fisher Information for NeRF/3DGS NBV. Correct.
- **ActiveSplat (RA-L 2025)** ([arXiv:2410.21955](https://arxiv.org/abs/2410.21955)) — 3DGS active mapping. Correct.

The Phase-2 backlog in `decision/004` should add an explicit task: **comparison with FisherRF and ActiveSplat as NBV baselines**, since both are correctly cited and represent the modern NBV literature this repo claims to extend.

## How to apply

1. Read this file alongside `decision/000-charter.md` and `decision/003-matching-algorithm.md`. Their text is unchanged (append-only), but the C1-C3 corrections override the affected lines.
2. The Phase-2 backlog now includes:
   - **(new)** Run FisherRF / ActiveSplat as NBV baselines and report against M1 (mosaic-SSIM-gain).
   - **(new)** Complete the Phase-0 prior-art search promised in `decision/000-charter.md` L72-73 (concrete search queries with zero hits) before the README's novelty paragraph drops the "unverified" caveat.

## Signature

User signature not requested — corrections do not change the adopted matching algorithm, license, or metric. They only correct the citation record.
