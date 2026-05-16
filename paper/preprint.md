# Hungarian beats Sinkhorn for photomosaic next-best-view — but Oklch pool augmentation only helps Hungarian

**Status:** preprint draft, 2026-05-16. Generated from `paper/outline.md`
and the JSON snapshots in `experiments/results/`. Every number in the
manuscript is regenerable from one of those files.

**Authors:** TBD (see `## Author list` TODO at the bottom of
`paper/outline.md`).

**Target venues:** arXiv (primary distribution); Insights from Negative
Results in NLP @ EMNLP 2026 if scope is accepted; OTML @ NeurIPS 2026
when its CFP opens. See `paper/outline.md` §Target venues for the
deadline survey.

**Code, data, harness:** MIT, public.
GitHub: <https://github.com/hinanohart/mosaicraft-active-vision> and
<https://github.com/hinanohart/oklch-aug>.

---

## Abstract

We benchmark Sinkhorn-OT against Hungarian assignment as the matching
layer of a photomosaic-driven next-best-view (NBV) loop, and find that
Hungarian outperforms log-domain Sinkhorn at every regularisation
strength we tried (95% paired bootstrap CI strictly below zero across
all 12 Sinkhorn conditions, N=16 paired runs per condition). We then
ask whether augmenting the candidate tile pool with perceptually-uniform
Oklch hue rotations — which we extract into a standalone library,
`oklch-aug` — closes the gap. It does not: Oklch pool augmentation
improves Hungarian by +0.026 SSIM (95% CI [+0.018, +0.036]) and *hurts*
Sinkhorn by −0.042 SSIM (95% CI [−0.047, −0.037]) on the same paired
harness (N=32 in the tight Phase 3 run). We argue this is consistent
with a structural property of entropic OT distributing mass over
near-duplicate candidates, though we did not measure the
ε × oklch-aug interaction directly so we cannot rule out an
ε-tuning component (Limitations §6.2). Every claim is regenerable
from a committed JSON file. Code, harness, library, and
decision-by-decision audit trail are released under MIT.

---

## 1. Introduction

The natural framing for any matching-driven active vision problem is to
choose the next observation that most improves a downstream matching
objective. Examples in the recent literature include FisherRF (ECCV
2024), ActiveSplat (RA-L 2025), GenNBV (CVPR 2024), POp-GS (CVPR 2025),
and NBV-Splat (arXiv:2512.22771); all of them measure "how well does the
current observation set let me reconstruct the scene?" and pick the
next viewpoint accordingly. A natural variant — using **photomosaic
reconstruction loss** as the NBV signal — was unexplored when we
started this project: photomosaic construction has a long tradition in
graphics and a clear bipartite-matching structure (cells ↔ tiles), and
recent work in saliency-as-marginal optimal transport (Cuturi 2013;
Peyré & Cuturi 2019) suggested that Sinkhorn-OT might be the
principled choice.

We report two findings that surprised us, both with paired-bootstrap
evidence.

1. **Sinkhorn-OT loses to Hungarian** on the toy in every variant we
   tried — three entropic regularisation strengths (ε ∈ {0.01, 0.05,
   0.1}), two saliency framings (row-scaling and source-marginal), and
   the one-tile-per-cell uniqueness toggle. The 95% paired-bootstrap
   CI of (Sinkhorn − Hungarian) ΔSSIM is strictly **below** zero for
   all 12 Sinkhorn conditions (Figure 1, left panel; same sign
   convention as Abstract and §5.2).
2. **Oklch hue-rotation pool augmentation splits the matchers.** A
   simple L-preserving expansion of the tile pool — extracted from the
   mosaicraft photomosaic library and packaged as a standalone
   library, `oklch-aug` — improves Hungarian by +0.026 SSIM
   (95% CI [+0.018, +0.036]) and *hurts* Sinkhorn by −0.042 SSIM
   (95% CI [−0.047, −0.037]). Figure 1, right panel.

We argue that the second finding is *consistent with* a structural
property of entropic OT: when the candidate pool contains
near-duplicate variants (the Oklch-rotated copies, which by
construction share the same L channel as the originals), the entropic
plan distributes mass across all of them, splitting useful supply over
redundant rows. Hungarian's hard 1:1 assignment does not have that
failure mode and benefits unambiguously from the larger candidate set.
We did not run an ε sweep × oklch-aug ablation (see §6.2), so the
"structural" framing is supported by but not proven from the data.

Our contribution is therefore three things at once:

* A clean negative result on Sinkhorn vs. Hungarian for a niche but
  non-trivial active-vision problem (§5.1–5.2).
* A positive result on perceptual pool augmentation that *only*
  applies to the simpler matcher (§5.3).
* `oklch-aug`, a small standalone library (numpy core, torch optional,
  Albumentations / Torch adapters) so the technique is reusable
  outside photomosaic NBV (§7).

A negative result published with this much detail is rare; we take
care to argue mechanisms and to make every claim independently
regenerable.

---

## 2. Background

### 2.1 Photomosaic reconstruction and bipartite matching

A photomosaic builds a coarse approximation of a *target image* by
tiling a grid of *cells* with small *tile* images, choosing each tile
so the cell-and-tile pair minimise a perceptual distance. The natural
matching problem is bipartite: one tile per cell, minimising total
cost. The mosaicraft codebase has been doing this with the Hungarian
algorithm (`scipy.optimize.linear_sum_assignment`) since v0.1, and the
saliency layer originally used a *row-scaling* hack — multiply the
cost row of every cell by the cell's saliency before assignment — so
high-saliency cells get matched first.

### 2.2 Sinkhorn-OT and the saliency-as-marginal framing

Sinkhorn (Cuturi 2013) replaces hard assignment with an entropic
relaxation: minimise transport cost plus ε times the negative entropy
of the plan. The dual updates are diagonal scalings of the cost
matrix; log-domain Sinkhorn (Schmitzer 2019) keeps the iterates
numerically stable down to ε on the order of 1e-3. The relevant
property for our setting is that saliency, which Hungarian could only
use via the row-scaling hack, has a clean role in OT as the *source
marginal*: the row sum of the transport plan is constrained to equal
the saliency-weighted row mass. Sinkhorn was therefore the
"principled" choice when we started.

### 2.3 Active vision / next-best-view

The NBV literature is large and growing fast. Recent representative
works include FisherRF (ECCV 2024; information-gain NBV in NeRF),
ActiveSplat (RA-L 2025; Gaussian-splat NBV with frontier-based view
proposals), GenNBV (CVPR 2024; generalisable policy across scenes),
POp-GS (CVPR 2025; uncertainty-aware NBV on Gaussian splats), and
NBV-Splat (arXiv:2512.22771; NBV with monocular Gaussian splatting).
All of them use a *reconstruction-quality* signal at the per-pixel or
per-Gaussian level. Our setting is different: we use the
*reconstruction loss of a photomosaic* as the NBV signal, which is
matching-driven, not splat-driven.

### 2.4 Oklab and Oklch

Ottosson (2020) introduced the Oklab perceptual colour space —
roughly, a linearised CIELAB with parameters fit to MacAdam ellipses
on Munsell data, giving better hue-uniformity. Its polar form Oklch
splits chroma and hue, so rotating hue at constant L is a single
trigonometric pass that preserves perceptual lightness exactly in the
float representation. Two effects move L away from the original in
the uint8 pipeline: (i) sRGB gamut clipping — rotating chroma can
push the result outside the displayable cube, the implementation
clips, and the resulting Oklab L drifts; (ii) uint8 round-trip
quantisation, which is sub-LSB. Figure 2 shows the |ΔL\*|
distribution on a real high-chroma sample: across our default
schedule the median |ΔL\*| stays below 0.002 (well sub-perceptual)
but the long tail reaches max |ΔL\*| ≈ 0.04–0.07 — driven by gamut
clipping, not quantisation.

### 2.5 Niche specificity vs. prior art

A reviewer reading "Hungarian beats Sinkhorn" should immediately ask:
*didn't someone show this already?* The closest prior art we are aware
of is Dong et al. (arXiv:2005.01182), which reports the analogous
finding in the *embedding-domain* assignment setting (matching
pre-trained image embeddings to text embeddings). Our claim is
narrower: in the *photomosaic-cell-vs-perceptual-tile-distance* domain
with 50×50 cost matrices and a saliency signal, Hungarian also wins.
We argue this is a complementary observation — a different domain,
the same direction — not a duplicate finding.

---

## 3. Method

### 3.1 Photomosaic-NBV loop

Given a partial set of camera views, a candidate tile pool extracted
from the observations, and a target image, the NBV loop scores every
remaining viewpoint by how much adding it *would* improve the
photomosaic SSIM, and picks the argmax. The mosaic SSIM-gain
strategy uses an inner loop that solves the assignment problem for
each hypothetical observation and returns the resulting full-mosaic
SSIM. Saliency biases the cost via either the row-scaling hack or the
source-marginal framing, depending on the condition.

### 3.2 Assignment variants

* **Hungarian.** `scipy.optimize.linear_sum_assignment` on the
  Oklab perceptual-distance cost matrix. Saliency is applied as a
  row-scale.
* **Sinkhorn-OT.** Log-domain numpy implementation with 100 max
  iterations and ε swept across {0.01, 0.05, 0.1} for the
  baseline JSON. Two saliency framings (row-scale, source-marginal) and
  a uniqueness toggle (force-1:1 by argmax-after-Sinkhorn vs. allow
  tied assignments) give 3 × 2 × 2 = 12 condition labels, all reported
  in the Phase 2 baseline. The numerical-stability addendum in §5.4
  separately sweeps ε down to 5e-4 to characterise the log-domain
  solver itself (independent of the NBV claim).

### 3.3 Oklch hue-rotation pool augmentation

For each tile in the candidate pool, generate `k` additional copies by
rotating hue in Oklch by evenly-spaced angles at constant L. The
canonical schedule is (72°, 144°, 216°, 288°) — four rotations placing
variants in different a/b-plane quadrants — giving a 5× pool expansion
(originals plus four copies). The schedule is parameterised; the
default is what the mosaicraft codebase used prior to extraction.

`oklch-aug` (§7) implements this as both a single-image function
(`rotate_hue_oklch`) and a pool expander (`HueRotatePool`). The
expander preserves original-order stability so reproducibility-sensitive
matchers see the same indexing across runs.

---

## 4. Benchmark setup

### 4.1 Toy scene

A 32-view hue-ring scene: cameras placed on a ring around a synthetic
checkerboard whose colour varies smoothly around the ring. Each view
is rendered at 320×240 px. Targets are drawn from a held-out subset of
the same scene. The cost matrix is 50×50 (50 cells × ≤50 tiles) so
both matchers run in well under a second per call.

### 4.2 Pairing structure

Every condition shares the same `(scene_seed, target_seed, target_idx,
seed)` quadruple as its partner conditions, so paired-bootstrap CIs
remove the scene/target variance. Phase 2 uses N=4 seeds × N=4 targets
= 16 paired runs per condition. Phase 3 was originally also N=4×4=16
runs; we re-ran with N=8×4=32 for tighter CIs (the headline N=32
numbers in §1 and §5 are from the tight run). The tight run extends
the original seed range from 0–3 to 0–7 on the same scene/target
seeds, so the two runs are not statistically independent
replications — see §6.2.

### 4.3 Statistical procedure

Paired bootstrap with 5000 resamples and a fixed RNG seed. We report
mean difference and (2.5%, 97.5%) quantiles as the 95% CI (percentile
method, two-sided). Resampling is i.i.d. over paired pairs; we did not
use a cluster bootstrap over the target-index axis even though
between-target variance dominates between-seed variance — see
Limitations §6.2. Code: `experiments/benchmark_phase{2,3}.py`.

---

## 5. Results

![Figure 1. Paired bootstrap 95% CIs. Left: every Phase-2 Sinkhorn
variant minus the Hungarian baseline (all 12 bars below zero —
Hungarian dominates); N=16 paired runs per condition (4 scene seeds ×
4 target seeds). Right: Phase-3 oklch-aug minus no-aug, for the two
matchers — Hungarian gains, Sinkhorn loses; N=32 paired runs per
matcher (8 seeds × 4 targets). Bars are mean differences; error bars
are 95% paired-bootstrap CIs (10⁴ resamples, percentile method).
Marker colour uses the Okabe-Ito colorblind-safe palette: vermillion
(#D55E00) = CI strictly above-or-below zero in the *worse* direction,
bluish-green (#009E73) = CI strictly in the *better* direction, grey =
CI crosses zero (tie).](figures/fig_paired_ci.png)

### 5.1 Phase 1 — first-pass headline

Random-view selection vs. saliency-biased vs. mosaic-SSIM-gain
strategies. Both saliency-biased and mosaic-SSIM-gain beat the random
baseline on N=4×4 paired runs; see `decision/004` for the full
4-ablation table. Phase 1 did not compute paired-bootstrap CIs — the
verdict ("saliency- and SSIM-biased beat random") is qualitative and
the Phase-2 / Phase-3 quantitative claims are computed independently
from a re-collected run.

### 5.2 Phase 2 — Sinkhorn ε sweep, saliency framings, uniqueness

Twelve Sinkhorn variants × Hungarian baseline. **Every** Sinkhorn
variant loses at 95% paired-bootstrap CI strictly below zero
(Figure 1, left panel). The best Sinkhorn cell is
`rowscale_eps=0.1_unique=0` at Δ = −0.0415 [−0.053, −0.029]; the worst
is `marginal_eps=0.01_unique=0` at Δ ≈ −0.10. Tightening ε does *not*
close the gap — it widens it (consistent with the entropic plan
becoming sharper but the plan-to-assignment recovery still trailing).

Decision/006 records the full table. The TL;DR for the paper: the
principled-looking choice loses on every axis we tried.

### 5.3 Phase 3 — Oklch pool augmentation × matcher (tight, N=32)

| condition | mean final SSIM | std | oklch_aug Δ vs. no-aug (95% CI) |
|---|---:|---:|---|
| hungarian_no_aug | 0.4990 | 0.031 | (baseline) |
| hungarian_oklch_aug | 0.5255 | 0.022 | **+0.0264 [+0.018, +0.036]** BEATS |
| sinkhorn_no_aug | 0.4717 | 0.041 | (baseline) |
| sinkhorn_oklch_aug | 0.4295 | 0.042 | **−0.0421 [−0.047, −0.037]** WORSE |

(All four conditions: SINKHORN_EPSILON=0.1, OKLCH_N_VARIANTS=4 so the
pool is 5× the original, paired N=32.)

The N=4×4 replication (`phase3_baseline_2026-05-16.json`) shows the
same sign for both matchers with overlapping CIs:
Hungarian Δ = +0.020 [+0.009, +0.032] BEATS;
Sinkhorn Δ = −0.041 [−0.049, −0.034] WORSE.

The Sinkhorn-side loss is the more surprising of the two: a strictly
larger candidate pool with strictly more diverse hues should not, on a
naive view, *hurt* a matching algorithm.

### 5.4 Numerical stability addendum

A separate property test, independent of any NBV claim, exercises
log-domain Sinkhorn at ε ∈ {0.1, 0.05, 0.01, 0.005, 0.001, 5e-4} on a
realistic 50×50 cost matrix at cost scale O(50). All produce finite
plans with `np.isfinite(...).all()` and `plan.sum() ≈ m` to atol=1e-6.
Test: `tests/test_matching.py::test_sinkhorn_log_domain_no_nan_at_small_epsilon`.
This is the part of the work that maps cleanly onto PythonOT/POT
PR #724 (§9).

---

## 6. Discussion

### 6.1 Why does Oklch pool augmentation split the matchers?

Our hypothesis is *mass distribution under entropic regularisation*.
The Oklch-rotated copies are perceptually distinct but, *in cost-matrix
terms*, near-duplicate to the original tile: they share the same Oklab
L by construction, so their per-cell cost values cluster tightly around
the original's cost values. Sinkhorn-OT's entropic plan, faced with a
cluster of near-duplicate-cost rows, distributes its mass over the
cluster rather than concentrating on a single row. The effective
"useful" mass per matched row is therefore split across redundant
candidates, which hurts the final assignment quality once the plan is
collapsed back to a 1:1 mapping by argmax.

Hungarian, in contrast, sees the same near-duplicate cluster and picks
exactly one element of it — the one whose cost is minimally lower than
its siblings. The expansion is pure upside: any rotated variant that
happens to fit a cell better than the original (a common case once the
target image is multi-hued) becomes available without dragging the
plan apart.

We do not have a closed-form proof of this mechanism. The signs in
§5.3 and the existing entropic-OT literature on mode collapse and mass
distribution (Peyré & Cuturi 2019 §4.2) are consistent with it. A
formal characterisation — bound on the regularisation strength below
which the split flips, perhaps — is open.

### 6.2 Limitations

* **Toy scene only.** Both Phase 2 and Phase 3 ran on the hue-ring
  scene. Whether the verdict generalises to real images is open; real
  fixtures are the next gate (§A).
* **One pool augmenter family.** We tested L-preserving Oklch hue
  rotation; other pool expanders (HSV jitter, ColorJitter, AugMix)
  would change the near-duplicate-cluster property of §6.1 and might
  not exhibit the split.
* **Single ε for Phase 3.** Phase 3 uses ε = 0.1 (the Phase-2 best for
  Sinkhorn). An ε sweep × oklch-aug interaction is unmeasured, which
  is the main reason the Abstract softens the "structural property"
  claim to "consistent with".
* **i.i.d. paired bootstrap.** The CIs in §5 use i.i.d. resampling over
  paired pairs. The pairing structure is `(scene_seed, target_seed,
  target_idx, seed)` and between-target variance is ≈10× between-seed
  variance in the Phase-3 results JSON. A cluster bootstrap over
  `target_idx` would widen the CIs, though the signs and rough
  magnitudes are stable across **bootstrap-RNG seed** choice (verified
  at 100 distinct bootstrap RNG seeds). Note: this is robustness of
  the resampling distribution, *not* robustness across independent
  experimental seeds — which is the open question the seed-extension
  caveat below addresses.
* **Phase 3 N=32 is a seed extension, not an independent replication.**
  The tight run extends seeds 0–3 (loose) to 0–7 on the same
  `(scene_seed, target_seed)` grid. Sign and CI shape are stable
  between the two; we use the tighter run as the headline but the
  loose run is preserved in `experiments/results/` for audit.
* **No claim about embodied robotics.** The photomosaic reward is a
  specific instantiation of matching-driven NBV reward. Recent
  diffusion-policy and VLA literature (pi0.5, OpenVLA, RoboTwin 2.0,
  OTPR arXiv:2502.12631) uses Sinkhorn plans in much higher-dimensional
  state spaces; we make no claim about whether the same
  pool-augmentation split would appear there.

### 6.3 Relationship to OTPR (arXiv:2502.12631)

OTPR uses Sinkhorn plan gradients to train a scoring head end-to-end;
the differentiability of the plan is the *whole point*. In our
setting, Sinkhorn's plan is *wished* (smooth, differentiable) but only
used via a hard argmax, so we never reap that benefit. A natural
follow-up is to drop the argmax and train a scoring head against a
Sinkhorn-plan reward; this is the OTPR direction applied to
photomosaic-NBV.

---

## 7. The `oklch-aug` library

`oklch-aug` is a standalone library extracted from the mosaicraft
codebase (`mosaicraft.color_augment.expand_color_variants`) so the
technique is reusable outside photomosaic-NBV. Repo:
<https://github.com/hinanohart/oklch-aug>.

### 7.1 API tour

```python
from oklch_aug import rotate_hue_oklch, HueRotatePool, oklab_distance

# Single image — uint8 HWC.
out = rotate_hue_oklch(img, hue_shift_deg=72.0)

# Pool of N images → N * (1 + k) images, originals first.
pool = HueRotatePool(n_variants=4)        # DEFAULT_HUE_SCHEDULE
expanded = pool([img1, img2, img3])        # len(expanded) == 15

# Oklab perceptual distance — numpy pairwise, torch helper available.
d = oklab_distance(grid_means, tile_means)
```

### 7.2 Design choices

* **numpy core, torch optional.** The single-image rotate and the pool
  expander are pure numpy. The torch helper for `oklab_distance` and
  the torch adapter live behind extras.
* **Channel-order parameter.** `channel_order={"rgb","bgr"}` keeps the
  function honest for both OpenCV-native (BGR) and PIL-/Albumentations-
  native (RGB) pipelines without coding two implementations.
* **L preservation: float-exact, gamut-clip-bounded.** The math
  preserves Oklab L exactly in float. In the uint8 pipeline two effects
  shift L away from the original: sub-LSB quantisation (negligible) and
  sRGB gamut clipping (dominant once chroma is high enough that the
  rotated colour leaves the sRGB cube). Figure 2 shows the |ΔL\*|
  distribution on `tiles_sample.jpg` across the default rotation
  schedule: median |ΔL\*| ≲ 0.002, but the tail reaches max
  |ΔL\*| ≈ 0.04–0.07. The technique is therefore L-preserving in the
  median-pixel sense but not as a hard upper bound; downstream users
  who need a strict bound should restrict the chroma range upstream
  or disable `protect_highlights` / `protect_shadows`.
* **Adapters.** `oklch_aug.adapters.albumentations.OklchHueRotation`
  is an `ImageOnlyTransform` for use inside Albumentations or
  AlbumentationsX pipelines (it gates dtype/shape and honours
  `Compose(seed=...)` reproducibility);
  `oklch_aug.adapters.torch.OklchHueRotation` is a plain torch
  `nn.Module` that accepts (B, 3, H, W) float[0,1] tensors and warns
  on `requires_grad=True`. The torch wrapper round-trips through CPU
  numpy; a torch-native autograd path is open follow-up. The torch
  adapter is deliberately not a `kornia.augmentation.AugmentationBase2D`
  subclass — wrap it with your own kornia adapter if you need
  `AugmentationSequential`-style composition.

![Figure 2. Oklab L deviation under the default `HueRotatePool`
schedule (+72°, +144°, +216°, +288°) on `tiles_sample.jpg`. Per-pixel
|ΔL\*| histograms; subplot titles give the empirical max and 99th
percentile per angle. The tail is driven by sRGB gamut clipping at
constant Oklab L, not by uint8 quantisation; median |ΔL\*| stays
≲ 0.002 across all four angles.](figures/fig_L_preservation.png)

### 7.3 Why a separate library

The Oklch hue-rotation technique is matcher-agnostic. The Hungarian /
Sinkhorn split is specific to *this* paper. A user who only wants the
augmentation should not have to import the entire NBV stack to get it.
The separation also let us add formal tests, mypy strict mode, and a
small public surface that is easier to evolve than the in-mosaicraft
version.

---

## 8. Reproducibility checklist

* All code is MIT, public on GitHub.
* Every number in this paper is regenerable via one of:
  * `python experiments/benchmark_phase1.py`,
  * `python experiments/benchmark_phase2.py`,
  * `python experiments/benchmark_phase3.py`.
* JSON snapshots of every reported number live under
  `experiments/results/phase{1,2,3}_baseline_*.json`; the .gitignore
  carves them out explicitly so reviewers can `git diff` future runs
  against the 2026-05-16 baselines.
* Bootstrap RNG seed is recorded in the JSON. Phase 3 RNG seed is the
  loop seed; bootstrap RNG seed is hard-coded in
  `experiments/benchmark_phase2.py::bootstrap_ci_diff`.
* `oklch-aug` is pinned via SHA in
  `.github/workflows/ci.yml` until PyPI upload; once on PyPI, the SHA
  becomes a version pin.
* `mosaicraft` is consumed as a git submodule pinned to commit
  `2918137` (v0.3.2-32-g2918137).

---

## 9. Open follow-ups

### 9.1 Real-images Phase 3

The Phase 3 split is currently only on the hue-ring toy. Two
fixtures are queued: (i) the mosaicraft test fixtures shipped under
`external/mosaicraft/docs/images/`, and (ii) crops from Open Images V7.
The expectation is that the sign of both effects (Hungarian-up,
Sinkhorn-down) survives; the magnitudes may shift.

### 9.2 PythonOT/POT PR #724

The numerical-stability tests in §5.4 are independent of the active
vision claim and map directly onto the open PR
[PythonOT/POT#724](https://github.com/PythonOT/POT/pull/724), which has
been stuck in [WIP] state since 2025-03 on missing tests + docs build
+ a master conflict. Our planned contribution is co-authoring a
follow-up PR that adds those pieces (notes in
`notes/pot-pr-724-rescue.md`). This is gated on user sign-off per the
project's R14 protocol.

### 9.3 Differentiable photomosaic-NBV via Sinkhorn-plan gradients

OTPR-style — train a scoring head against a Sinkhorn-plan reward
instead of argmax over the plan. The pool-augmentation split of §6.1
predicts that Hungarian-style hard assignment will lose the gradient
signal entirely, so this is exactly the regime where Sinkhorn could
finally win.

### 9.4 Pool-augmentation evaluation outside photomosaic

A reasonable downstream test for `oklch-aug` is policy training under
distribution shift — e.g., the photometric robustness benchmarks in
RoboLight (arXiv:2603.04249) and similar 2026 diffusion-policy /
VLA work. We make no claim here; it is the obvious next library
client.

---

## Appendix A — `decision/` audit trail

This paper does not exist in a vacuum; it is the public extract of a
research diary kept in `decision/` and `PROGRESS.md`. Readers who want
the "why" behind methodology choices (license, metric, matcher) should
look at:

* `decision/000-charter.md` — 4-agent debate that established the
  project's scope.
* `decision/001-license.md` — MIT, signed.
* `decision/002-evaluation-metric.md` — Mosaic-SSIM-gain per view as
  the primary metric, signed.
* `decision/003-matching-algorithm.md` — Sinkhorn-OT, with the now-
  falsified rationale, signed; **the binding decision is preserved
  even though the experimental verdict went the other way** — that
  is what makes the audit trail honest.
* `decision/004-phase1-findings.md`, `005-citation-corrections.md`,
  `006-phase2-findings.md`, `007-oklch-aug-extraction.md` — reported
  findings, no signatures (these are reports, not adoptions).

The `005` document is the post-publication citation audit; it documents
where the original Sinkhorn-OT motivation was citation-thin and lists
the corrected primary sources. We chose to keep this in-repo rather
than silently fixing the README.

---

## Appendix B — Glossary of the four "defensible claims" from decision/006

From `decision/006-phase2-findings.md`, the four claims that survive
the Phase 2 sweep:

1. Log-domain Sinkhorn is numerically stable at ε down to 5e-4 on the
   50×50 cost matrices in this setting. **(Advanced as C1 in §5.4.)**
2. Switching saliency from row-scaling to source-marginal does not
   improve Sinkhorn enough to beat Hungarian. **(Advanced as C2 in
   §5.2.)**
3. Tightening ε does not flip the verdict; the Hungarian-minus-Sinkhorn
   gap widens. **(Advanced as part of C3 in §5.2.)**
4. Forcing uniqueness on the Sinkhorn plan by argmax does not help.
   **(Advanced as part of C3 in §5.2.)**

---

## Acknowledgements

To be filled in once author list is decided. Thanks in advance to the
PythonOT/POT maintainers for the open issue scaffolding the §5.4
numerical-stability tests build on.
