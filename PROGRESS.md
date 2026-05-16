# PROGRESS — read this first if you (Claude or human) are picking up cold

**Why this file exists:** `/compact` may erase the assistant's working
memory at any moment. This file, plus `decision/`, plus `git log`, must
contain enough state to fully resume work without re-asking the user
anything. Update this file at the end of every step.

**Last update:** 2026-05-16 — **step 16 LOCAL.** `oklch-aug` lib (Layer 1)
extracted to `/home/runza/oss/oklch-aug/` with 31 tests + ruff/mypy clean,
and `cost.py:bgr_to_oklab` rewired (54/54 tests pass on
`mosaicraft-active-vision`, phase1+phase2 smoke benches green). `oklch-aug`
is **not yet** published to GitHub or PyPI; CI on this repo will need an
install step once it is — see decision/007. Steps 1-10 shipped earlier to
https://github.com/hinanohart/mosaicraft-active-vision (public, MIT).
**Project root:** `/home/runza/oss/mosaicraft-active-vision/`
**Upstream (R17 reuse target):** `/home/runza/oss/mosaicraft/` (will be
git submoduled into `external/mosaicraft/`).

---

## What this repo is, in one paragraph

A research-stage extension of `mosaicraft` that turns the photomosaic
reconstruction loss into an **active vision** signal: given a partial
scene observation, choose the next camera viewpoint that improves the
resulting photomosaic the fastest. Assignment is done with **Sinkhorn-OT**
(decision/003), the **saliency** signal from `mosaicraft/saliency.py`
becomes the OT cell marginal, and the primary metric is **Mosaic-SSIM-gain
per view** (decision/002). License is **MIT** (decision/001), matching
upstream mosaicraft.

## Hard constraints (do not violate even if asked nicely)

These are recorded so that a fresh Claude session can't re-litigate them.

1. **No Hungarian fallback.** Not even as a `--legacy` flag. The whole
   point of this repo is to demonstrate Sinkhorn-OT as the assignment
   layer. A silent fallback would let benchmarks mis-attribute results.
   (decision/003 "What this does NOT include")
2. **No unverified numeric claims in README or docstrings.** After the
   mosaicraft "5x faster" unverified docstring, every number in a
   user-visible string must be backed by a benchmark file path.
3. **No `gh repo create` until** (a) Phase-1 demo runs end-to-end, (b) all
   four ablations from decision/002 produce signal, (c) CI is green
   locally. (charter.md "Public release gates")
4. **No `rm -rf` ever** (R8). Failed experiments go to
   `experiments/_wip/<name>/`. The `.gitkeep` is sacred.
5. **Reuse from mosaicraft is via submodule, not copy.** Charter forbids
   forking mosaicraft. Files like `features.py` and `saliency.py` are
   imported through the submodule path; only **Oklab distance** is
   extracted into this repo's `cost.py` because mosaicraft's code mixes
   it with Hungarian-specific logic at `placement.py:98-103`.
6. **Decision docs are append-only after SIGNED.** If a decision needs
   revision, add a new file `00X-revisited.md` referencing the old one;
   do not rewrite the signed version.

## Decision status

| File | Status | User signature (verbatim) |
|---|---|---|
| `decision/000-charter.md` | reference | n/a |
| `decision/001-license.md` | **SIGNED** | `mit` |
| `decision/002-evaluation-metric.md` | **SIGNED** | `002は最適で` |
| `decision/003-matching-algorithm.md` | **SIGNED** | `003最適でエージェントが最適とゆってるやつで結果出てた` |
| `decision/004-phase1-findings.md` | REPORTED | 2026-05-16 baseline run |
| `decision/005-citation-corrections.md` | REPORTED | 2026-05-16 post-publication audit (no signature needed: corrections, not new adoption) |
| `decision/006-phase2-findings.md` | REPORTED | 2026-05-16 phase-2 ε sweep + saliency-as-marginal + paired bootstrap (no signature needed: reporting, not adoption) |
| `decision/007-oklch-aug-extraction.md` | REPORTED | 2026-05-16 3-agent review → standalone `oklch-aug` lib (no signature needed: extraction, not new adoption) |

## File-by-file status

| Path | Created | Implements | Tests |
|---|---|---|---|
| `LICENSE` | ✅ | MIT text | n/a |
| `README.md` | ✅ | overview + decision pointers | n/a |
| `pyproject.toml` | ✅ | hatchling, py3.10+, MIT, deps | n/a |
| `.gitignore` | ✅ | Python + R8 _wip protection | n/a |
| `PROGRESS.md` | ✅ (this file) | bootstrap doc | n/a |
| `src/mosaicraft_active_vision/__init__.py` | ✅ | version stub only | n/a |
| `src/mosaicraft_active_vision/cost.py` | ✅ | mosaicraft feature+saliency wrapper + Oklab dist | ✅ (12) |
| `src/mosaicraft_active_vision/matching.py` | ✅ | log-domain Sinkhorn (numpy, torch optional via geomloss later) | ✅ (14) |
| `src/mosaicraft_active_vision/metrics.py` | ✅ | M1 mosaic_ssim_gain + sample_efficiency + M2 view_coverage | ✅ (19) |
| `src/mosaicraft_active_vision/nbv.py` | ✅ | NBV loop + Random / SaliencyBiased strategies | ⬜ (covered by Phase-1 bench) |
| `tests/test_matching.py` | ✅ | Sinkhorn marginal property tests | n/a |
| `tests/test_metrics.py` | ✅ | golden hash sha256[:8]=`869bdbdb` on 32x32 scene | n/a |
| `tests/test_cost.py` | ✅ | Oklab + cost matrix sanity | n/a |
| `experiments/benchmark_phase1.py` | ✅ | 4 ablations, runs end-to-end <2 min on CPU | ✅ (manual driver) |
| `experiments/results/phase1_*.json` | ✅ | baseline run snapshot (see decision/004) | n/a |
| `.github/workflows/ci.yml` | ✅ | py3.10/3.11/3.12 · ruff + mypy + pytest + license + smoke bench | n/a |
| `external/mosaicraft/` (submodule) | ✅ | pinned to `2918137` (v0.3.2-32-g2918137) | n/a |

## Step plan (commit-by-commit, atomic so `/compact` is safe)

Every step ends with a git commit so resuming requires only
`git log` + this table.

| # | Subject | Commit message head |
|---|---|---|
| 1 ✅ | Repo skeleton + 3 SIGNED decisions + PROGRESS.md | `init: skeleton, MIT license, signed decisions 001-003` (commit `d1b8886`) |
| 2 ✅ | Add mosaicraft as git submodule (Phase 1 reuse) | `add mosaicraft submodule at external/mosaicraft, pinned to 2918137` |
| 3 ✅ | Implement `cost.py` (Oklab + feature + saliency) | `feat(cost): Oklab perceptual distance + mosaicraft wrappers` |
| 4 ✅ | Implement `matching.py` (log-domain Sinkhorn) | `feat(matching): log-domain Sinkhorn-OT, numpy backend, argmax recovery` |
| 5 ✅ | Implement `metrics.py` (M1) | `feat(metrics): mosaic_ssim_gain primary DoD + M2 view_coverage` |
| 6 ✅ | Implement `nbv.py` (NBV loop) | `feat(nbv): NBV loop + Random / SaliencyBiased baselines` |
| 7 ✅ | Tests (45/45 passing) | `test: Sinkhorn marginal + metrics golden hash + cost sanity` |
| 8 ✅ | Phase-1 benchmark + 4 ablations + decision/004 | `bench: phase-1 + 4 ablations + decision/004 findings` |
| 9 ✅ | CI workflow (ruff + mypy + pytest + smoke bench) | `ci: ruff + mypy + pytest + smoke bench on push/PR` |
| 10 ✅ | `gh repo create` + push + fix CI mypy on py3.10/3.11 | `ci(mypy): follow_imports=skip + explicit NDArray annotation` |
| 11 ✅ | Post-publication citation audit (decision/005) | `docs: post-publication citation audit + README honesty (decision/005)` |
| 12 ✅ | CITATION.cff + verified primary sources | `docs: add CITATION.cff with verified primary-source references` |
| 13 ✅ | Phase-2 bench harness + log-domain stability tests | `bench(phase2): saliency-as-marginal + ε sweep + multi-seed harness` |
| 14 ✅ | Phase-2 findings (decision/006) — Hungarian still wins, statistically | TBD on commit |
| 15 ✅ | Oklch hue-rotation pool aug extracted to `oklch-aug` lib (/home/runza/oss/oklch-aug/, 31 tests pass) | `feat(extract): oklch-aug v0.0.0 — perceptual hue-rotation pool aug` |
| 16 ✅ | `cost.py` rewired to import `bgr_to_oklab` from `oklch_aug` (54/54 tests pass, phase1+phase2 smoke green, ruff/mypy clean) | `refactor(cost): source bgr_to_oklab from oklch-aug (decision/007)` |
| 17 ✅ | Phase-3 benchmark: Oklch pool aug x NBV (toy scene, 16 paired runs) — Hungarian+oklch_aug BEATS, Sinkhorn+oklch_aug WORSE; results/phase3_20260516T215751.json | `bench(phase3): Oklch pool aug x NBV — Hungarian gains +0.02, Sinkhorn loses 0.04` |
| 18 ✅ | arXiv preprint outline (paper/outline.md — 7 headline claims, target venues, TODOs) | `docs(paper): add Insights/OTML preprint outline` |
| 19 ✅ | POT PR #724 rescue plan (notes/pot-pr-724-rescue.md — investigation + 9-step execution plan, awaiting R14 user sign-off) | `docs(notes): POT PR #724 rescue investigation` |
| 20 ⬜ | (gated) Push oklch-aug to GitHub + PyPI (R14 trigger) | TBD on user approval |
| 21 ⬜ | (gated) Execute POT PR #724 rescue (R14 trigger) | TBD on user approval |

## Where each user instruction lives (R16 audit trail)

These are the verbatim instructions that drove the project. Quoted as-is.

1. `「最終導入アーキテクチャを議論して」` → triggered 4-agent debate.
2. `「最適でossスタートしよう最高に革新的なものを作ろう」` → set the
   goal: "start the optimal OSS, make the most innovative thing."
3. `「それもエージェントで議論して決めるべき」` → forced agent debate
   for the 3-path decision (A/B/C).
4. `「最適でエージェントでが出してくれた結論を元に」` → user accepts
   the integrated agent verdict (path C + Sinkhorn).
5. `「mit, 002は最適で, 003最適でエージェントが最適とゆってるやつで結果出てた」`
   → signed 001 / 002 / 003.
6. `「スラッシュコンパクトをできるような圧縮しても大丈夫な進み方で行こう」`
   → motivated this PROGRESS.md + atomic-commit operating model.
7. `「oss化までやってくれよな最高のものを作りたいから」` → terminal
   goal: `gh repo create` + push, with charter gates respected.

## Resume protocol (if you are a fresh Claude session reading this)

1. Read this file top-to-bottom. (1 minute.)
2. Read `decision/000-charter.md`, then 001-003 in order. (5 minutes.)
3. Run `git -C /home/runza/oss/mosaicraft-active-vision/ log --oneline`
   to see what's been committed.
4. Find the first ⬜ row in the "File-by-file status" table — that is
   the resume point.
5. Do **not** ask the user "where did we leave off?". This file is the
   answer.
6. Run `kluster_code_review_auto` on every file create/edit (chat_id
   inheritance — find it in the last `kluster.ai Review Summary` in the
   chat or default to a fresh call).

## Open risks (recorded so they don't get forgotten on `/compact`)

- **kluster.ai trial has ended.** Reviews go through but return no signal.
  Quality enforcement falls back on local tests + CI + this PROGRESS doc.
- **mosaicraft submodule pinning.** Phase-1 must pin to a specific
  commit, not `main`. Otherwise mosaicraft's next breaking change breaks
  our reproducibility.
- **Sinkhorn epsilon sweep is part of M1.** We must not lock epsilon
  before the ablation sweep, or the metric becomes circular.
- **`opencv-python-headless` vs `opencv-python`.** The pyproject pins
  the headless version to keep CI lean. Local dev with GUI needs may
  swap; do not change the pyproject for that — use a personal venv.
