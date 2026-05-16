# POT PR #724 rescue — investigation notes (2026-05-16)

**Status:** scoped, not yet executed. R14 (議論プロトコル) trigger — a
follow-up PR to a 2.8k-star upstream repo is "external OSS contribution"
and needs user sign-off before any push.

## Current state of the upstream PR

- **Issue:** [PythonOT/POT#723](https://github.com/PythonOT/POT/issues/723) —
  `entropic_partial_wasserstein` returns NaN at small ε (e.g. 0.1) on a
  realistic 50×50 cost matrix. Opened 2025-03-12 by `wzm2256`. Labels
  `bug`, `help wanted`. **OPEN.**
- **PR:** [PythonOT/POT#724](https://github.com/PythonOT/POT/pull/724) —
  `[WIP] Add a stabilized function entropic_partial_wasserstein_logscale`.
  Opened 2025-03-13 by the same author. **OPEN, mergeable=CONFLICTING.**
- **Activity:**
  - 2025-03-13 — PR opened, two initial commits adding the function +
    `RELEASES.md` entry.
  - 2025-03-26 — `cedricvincentcuaz` merged master into the branch.
  - 2025-05-18 — `cedricvincentcuaz` merged master again.
  - 2025-05-23 — `rflamary` (POT maintainer) merged master.
  - 2025-09-07 — last `updatedAt` ping; no commits since.
- **Author self-flagged blockers (in PR body, verbatim):**
  > "I could not build the document in my laptop due to some errors:
  > `no theme named 'sphinx_rtd_theme' found`"
  > "I do not know how to use pytest to test my code. If this is
  > necessary, I may need some help here."

Translation: the maintainers want this in, the author can't get the
last 10 % over the line (tests + docs + conflict resolution), and the
PR has been stuck for 9 months on those last 10 %.

## What we can contribute that isn't already there

1. **pytest test suite for the new function.** The two checks the
   maintainers will demand are exactly the two we already have in
   `tests/test_matching.py`:
   - **No NaN/Inf at small ε.** Mirror
     `test_sinkhorn_log_domain_no_nan_at_small_epsilon` but on
     `entropic_partial_wasserstein_logscale`. Parametrise ε across
     [0.1, 0.05, 0.01, 0.005, 0.001, 5e-4]. Assert
     `np.isfinite(plan).all()` and `plan.sum() ≈ m` (the mass
     parameter).
   - **Bit-for-bit agreement with the old function at large ε.** The
     author already wrote this as `compare_logscale_POT.py` in their
     branch; turn it into a proper pytest case that passes
     `assert_allclose(..., atol=1e-6)` for ε ∈ [1, 10].
2. **Convert `compare_logscale_POT.py` to an `examples/` script.**
   POT uses Sphinx-Gallery; the file needs a top docstring with `# %%`
   cell markers and to land in `examples/plot_*.py` so the docs build
   picks it up.
3. **Resolve the master conflict.** Three master-merges have already
   happened; the current `CONFLICTING` state is most likely a small
   set of `ot/partial.py` lines that drifted while the PR sat. Rebase
   on master, resolve manually, push back to the same branch (if we
   get write access via co-author invite) or to a follow-up branch.
4. **Docs sentence.** One paragraph in `docs/source/all.rst` and a
   `.. autofunction::` entry — small, but blocked by the docs build
   error the author hit.

## Execution plan (M4 in the decision/007 timeline)

```
1.  Fork PythonOT/POT to hinanohart/POT.
2.  git fetch the WIP branch from wzm2256/POT (PR #724 head ref).
3.  Local rebase on PythonOT/POT master; resolve conflicts in
    ot/partial.py.
4.  Add tests/test_partial.py::
      test_entropic_partial_wasserstein_logscale_no_nan_at_small_epsilon
      test_entropic_partial_wasserstein_logscale_matches_old_at_large_epsilon
    using the same parametrize idiom as
    mosaicraft-active-vision/tests/test_matching.py.
5.  Move/rename compare_logscale_POT.py to
    examples/plot_entropic_partial_wasserstein_logscale.py with
    Sphinx-Gallery cell markers.
6.  Add docs/source/all.rst entry.
7.  Run POT's own test suite locally to confirm nothing else broke
    (pip install -e .[all], pytest ot/).
8.  Comment on PR #724:
      @wzm2256 @rflamary @cedricvincentcuaz — I'm a downstream user
      of `entropic_partial_wasserstein` from `mosaicraft-active-vision`
      and hit the same NaN regime you describe. I have a test
      scaffolding I'd like to bring upstream. Happy to either send a
      follow-up PR that builds on this branch (with @wzm2256 retained
      as co-author), or push directly to this branch if you'd prefer.
      Let me know which path you want.
9.  Wait for maintainer response. Default to follow-up PR unless they
    ask otherwise.
```

## Why not just open a competing PR

- Author `wzm2256` clearly wants this in; offering to ship the last
  10 % is more pro-social than racing them with a competing PR.
- The maintainers (rflamary, cedricvincentcuaz) have already done
  three master-merges on the existing branch, so they have invested
  in *this* PR specifically. Competing would forfeit that goodwill.
- Co-authoring or following-up keeps the original author's name on a
  recognised core library, which matters for them and costs us
  nothing.

## R-rule compliance check

- **R8** (rm 禁止): the rescue is additive (new tests, renamed
  example, docs sentence); nothing is deleted from upstream.
- **R11** (Secret 管理): no GitHub PAT will be passed through Claude;
  any `gh` calls run in the user's interactive shell when push is
  attempted.
- **R13** (セキュリティ隔離): contributing to PythonOT/POT is benign
  open-source contribution, no offensive surface.
- **R14** (大規模前 議論プロトコル): user sign-off required before
  step 1 (fork) and step 8 (PR comment). This file is the議論
  document.
- **R17** (既存修正優先): the rescue extends an existing PR rather
  than starting from scratch.

## Open questions for the user

1. Should the contribution be a **follow-up PR** (default, safer) or
   should we ask for write access to wzm2256's branch and push there?
2. Should the rescue PR cite mosaicraft-active-vision in the commit
   message as motivation, or stay scoped to POT's own concerns?
3. Timing: M4 in the decision/007 timeline is roughly 3 months out.
   Pull it forward if the maintainers signal "we need this in the
   next release"?
