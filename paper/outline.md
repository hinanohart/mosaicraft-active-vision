# Preprint outline — Photomosaic-NBV: an honest negative result for Sinkhorn-OT and a positive one for Oklch pool augmentation

**Status:** OUTLINE — 2026-05-16. Not yet a draft; this is the
skeleton of claims, evidence pointers, and target-venue notes that a
subsequent drafting pass will turn into prose.

**Provisional title (≤ 12 words):**
"Hungarian beats Sinkhorn for photomosaic-NBV — but Oklch pool
augmentation only helps Hungarian"

(Working alternative for a more upbeat angle if the reviewers we get
prefer a positive headline: "Oklch hue-rotation pool augmentation
improves Hungarian-assignment photomosaic active vision".)

## Target venues

| Venue | Fit | Notes |
|---|---|---|
| Insights from Negative Results (ACL/EMNLP workshop series) | High | Built around the exact framing we have: Sinkhorn was the principled choice and we report that it loses. |
| OTML (NeurIPS workshop on Optimal Transport in ML) | Medium-high | Headline matters here — Sinkhorn losing is a Sinkhorn-community insight, not a niche curiosity. |
| arxiv preprint (cs.CV, cs.LG) | Always | Fallback / primary distribution channel even if no workshop accepts. |
| ICLR Reproducibility Track | Possible | If we can find a published paper claiming Sinkhorn-OT for assignment-style problems and reproduce its setup, the falsification would land here. |

## One-paragraph abstract (provisional, 110 words)

We benchmark Sinkhorn-OT against Hungarian assignment as the matching
layer of a photomosaic-driven next-best-view (NBV) loop, and find that
Hungarian outperforms log-domain Sinkhorn at every ε in our sweep
(95 % paired bootstrap CI strictly below zero on N=16 runs). We then
ask whether augmenting the candidate tile pool with perceptually-
uniform Oklch hue rotations — which we extract into a standalone
library, *oklch-aug* — closes the gap. It does not: Oklch pool
augmentation improves Hungarian by +0.020 SSIM 95 % CI [+0.009,
+0.032] and *hurts* Sinkhorn by −0.041 SSIM 95 % CI [−0.049, −0.034].
We argue this is a structural property of entropic OT distributing
mass over near-duplicate candidates, not a tuning artefact. Code,
harness, and 100 % reproducible JSON are released under MIT.

## Headline claims (every claim links to evidence we already have)

| # | Claim | Evidence |
|---|---|---|
| C1 | Log-domain Sinkhorn does **not** NaN/Inf at ε down to 5e-4 on 50×50 cost matrices at cost scale O(50). | `tests/test_matching.py::test_sinkhorn_log_domain_no_nan_at_small_epsilon` |
| C2 | Saliency-as-marginal is the principled OT framing; saliency-as-row-scale is the mosaicraft-era Hungarian hack. | `decision/006` §"What we believed entering Phase 2"; both framings benchmarked in `experiments/benchmark_phase2.py`. |
| C3 | Hungarian beats every Sinkhorn variant we tried at 95 % paired bootstrap CI. | `decision/006-phase2-findings.md` headline table; `experiments/results/phase2_baseline_2026-05-16.json`. |
| C4 | Oklch hue-rotation pool augmentation is L-preserving by construction (Ottosson 2020) and turns an N-tile pool into N·(1+k). | `oklch-aug/src/oklch_aug/rotate.py`, `oklch-aug/src/oklch_aug/pool.py`; `tests/test_rotate.py::test_L_is_preserved_within_uint8_quantisation`. |
| C5 | Oklch pool augmentation **improves** Hungarian-assignment NBV final SSIM (+0.020, 95 % CI [+0.009, +0.032]). | `experiments/benchmark_phase3.py` `hungarian_oklch_aug` row; `experiments/results/phase3_20260516T215751.json`. |
| C6 | Oklch pool augmentation **hurts** Sinkhorn-OT NBV final SSIM (−0.041, 95 % CI [−0.049, −0.034]). | Same harness, `sinkhorn_oklch_aug` row. |
| C7 | The Sinkhorn-vs-Hungarian split under pool expansion is structural: entropic regularisation spreads transport mass over the near-duplicate Oklch variants, splitting the supply of useful tiles across redundant candidates. Hungarian, which solves for a 1:1 assignment, picks the single best variant. | Mechanism argument in §Discussion; supported by C5/C6 sign. |

## Negative results we own

- C3 (Sinkhorn loses to Hungarian on the toy at all ε / framings /
  uniqueness toggles).
- C6 (Oklch pool aug hurts Sinkhorn).
- The novelty motivation that originally drove decision/003 was
  citation-thin (see `decision/005-citation-corrections.md`); we
  report that publicly rather than burying it.

## Positive results we own

- C1 (numerical stability — a contract independent of the active
  vision claim; this is the part of the work that maps cleanly onto
  PythonOT/POT PR #724).
- C4 + C5 (Oklch pool augmentation as an L-preserving, perceptually
  uniform candidate-set expander that improves the simplest 1:1
  matcher).
- *oklch-aug* the library itself — extracted, tested, MIT-licensed,
  cv2-free, numpy-core, torch-optional.

## §1 Introduction (sketch)

Two paragraphs.

(1) NBV + assignment is the natural framing for any matching-driven
active vision problem (photomosaic, retrieval-augmented planning,
embodied scene reconstruction). The literature has Sinkhorn-OT
(Cuturi 2013, Peyré-Cuturi 2019) as the modern assignment layer, and
plenty of NBV literature (FisherRF ECCV 2024, ActiveSplat RA-L 2025,
GenNBV CVPR 2024, POp-GS CVPR 2025, NBV-Splat arXiv:2512.22771). But
the photomosaic-NBV combination — using photomosaic reconstruction
loss as the NBV signal — was unexplored when we started.

(2) We report two things that surprised us, with reproducible
evidence. (i) Sinkhorn-OT, the principled-looking choice, *loses* to
Hungarian on this toy. (ii) Oklch hue-rotation pool augmentation, a
mosaicraft-era trick we extracted into a standalone library, only
helps the matcher that loses to it. We argue why this is structural,
not a bug.

## §2 Background (sketch)

- Photomosaic construction (mosaicraft).
- Hungarian vs Sinkhorn-OT for assignment (with the niche specificity
  pointed out by critic: arXiv:2005.01182 reports the analogous
  finding in embedding domain, our contribution is *photomosaic +
  saliency-as-marginal + 50×50 perceptual cost*).
- Oklab / Oklch (Ottosson 2020) and why L-preservation matters.
- NBV literature pointers (above).

## §3 Method (sketch)

Three subsections, terse:

- 3.1 Sinkhorn-OT matching layer (log-domain, balanced, no Hungarian
  fallback — decision/003).
- 3.2 Oklch hue-rotation pool augmentation (rotate hue at constant L,
  evenly-spaced angles, originals-first stable order).
- 3.3 NBV loop with photomosaic-SSIM as reward.

## §4 Benchmark setup (sketch)

- ToyScene definition (hue-ring views, target = subset).
- N=8 seeds × N=4 targets (Phase 2) or N=4 seeds × N=4 targets
  (Phase 3) paired runs.
- ε sweep, saliency framings, uniqueness toggle (Phase 2).
- Oklch n_variants=4 → 5× pool (Phase 3).
- Paired bootstrap CI (5000 resamples, 95 %).

## §5 Results (sketch)

- 5.1 Phase 1 headline table (decision/004, 4 ablations).
- 5.2 Phase 2 ε sweep + framings + uniqueness — every Sinkhorn variant
  loses (decision/006).
- 5.3 Phase 3 Oklch pool aug — Hungarian gains, Sinkhorn loses. Figure
  with paired-CI bars, table with means/std/CIs.
- 5.4 Numerical stability addendum (the ε ≤ 5e-4 stability tests).

## §6 Discussion (sketch)

- Why does Oklch pool aug split the matchers? Hypothesis: entropic OT
  distributes mass across candidates that look near-identical in
  cost-matrix terms; the L-preserving rotated variants are exactly
  the worst case for that mechanism. Hungarian's hard 1:1 mapping
  doesn't have that failure mode and benefits from the larger
  candidate pool unambiguously.
- Limitations: still a toy scene, real images are immediate next
  work; the photomosaic reward is a specific instantiation of
  matching-based NBV reward and we don't claim broader robotics
  generalisation without further evidence.
- Relationship to OTPR (arXiv:2502.12631) — Sinkhorn's differentiable
  plan is *wished* in this work but not used; the OTPR direction
  (using Sinkhorn plan gradients to train a scoring head) is open
  follow-up.

## §7 Library: *oklch-aug* (sketch)

- API tour: `rotate_hue_oklch`, `HueRotatePool`, `oklab_distance`,
  RGB/BGR conversions.
- Why it is a separate library from this paper: the technique is
  matcher-agnostic and useful outside photomosaic NBV; we ship it so
  the next user does not have to re-derive Oklab from Ottosson 2020.

## §8 Reproducibility checklist

- All code MIT, public on GitHub.
- Every number in the paper is regenerable via
  `python experiments/benchmark_phaseN.py`.
- JSON snapshots of every reported number are committed under
  `experiments/results/phaseN_baseline_*.json`.
- Bootstrap RNG seed is recorded in the JSON.
- `oklch-aug` is pinned via PyPI / git submodule.

## §A Limitations + Future Work (already in §6 but listed here too)

- Real images (mosaicraft test fixtures + Open Images V7 crops).
- Differentiable NBV via Sinkhorn plan gradients (OTPR-style).
- Cross-saliency formulation experiments.
- POT PR #724 contribution (log-domain partial OT scaffolding).
- robotics policy data-augmentation evaluation of `oklch-aug` outside
  the photomosaic context (e.g., RoboLight arXiv:2603.04249).

## §B Provenance / agent-debate audit trail

- 4-agent debate that led to this repo: `decision/000-charter.md`.
- 4 SIGNED decisions: license, metric, matcher, Phase-2 backlog.
- Post-publication citation corrections: `decision/005`.
- Phase 2 findings: `decision/006`.
- *oklch-aug* extraction (3-agent re-debate): `decision/007`.

## TODOs before submission

- [ ] Real-images Phase 3 run on the mosaicraft test fixtures
      (Phase 3 was on the hue-ring toy).
- [ ] Repeat Phase 3 with N=8 seeds × N=4 targets for a tighter CI
      (Phase 3 was N=4×4=16 paired runs).
- [ ] Single figure: paired bootstrap intervals for Phase-2 ε sweep
      and Phase-3 oklch-aug split, side-by-side.
- [ ] Single figure: Oklab L-preservation diagnostic before vs after
      rotation (histogram of |ΔL| over a real image).
- [ ] Glossary of the four "defensible claims" from `decision/006`
      and which ones the preprint advances.
- [ ] Author list / affiliations / acknowledgements.
- [ ] OTML 2026 / Insights workshop deadlines — check.

## Tone / writing principles

- Honest framing. Do not soft-pedal the negative result.
- Niche specificity. The photomosaic-NBV setup is small; the claim
  is correspondingly scoped.
- Mechanism arguments over hand-waving. The Sinkhorn-vs-Hungarian
  split under pool expansion has a clean explanation we should make.
- Every number cites a JSON file and a commit.
