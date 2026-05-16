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

## 2026-05-16 — local rescue branch built (no fork/push yet)

Built the additive rescue locally at `/tmp/POT` on top of upstream
`master` (HEAD `41a4d57`). Branch name: `rescue-pr-724`. Patch file
exported to `notes/pot-pr-724-rescue.patch` (single commit, +367/-2
across 6 files). No fork, no push, no PR comment yet — that step is
R14-gated and waits for user sign-off.

### Cleanup pass (2026-05-16, same evening)

After an architect audit found two push-blockers, the patch was
re-exported with:

- **partial_wasserstein_1d auto-formatter drift removed**. The first
  patch contained an unrelated `assert (...), "msg"` ↔
  `assert ..., ("msg")` rewrite that an editor-side ruff hook had
  introduced. The file is now byte-identical to master except for the
  new function block.
- **Co-authored-by trailer corrected** from
  `<original PR #724 author>` placeholder to `<wzm2256@qq.com>` (the
  email visible in upstream PR #724's commit `b067a68`), so GitHub's
  co-author auto-link will fire when the PR is opened.
- **RELEASES.md entry added** under `0.9.7.dev0` (POT convention —
  the existing PRs all have one).
- **Example author line** trimmed to a single `# Author: wzm2256
  (original PR #724)` to avoid leaking the downstream `oklch-aug`
  project name into an upstream POT example.
- **Commit prefix** brought in line with POT convention
  (`[MRG] ...` instead of `[rescue-PR-#724] ...`).

The local author/committer are still placeholders
(`rescue-pr-724-prep <rescue@example.invalid>`) because the actual
identity belongs to whoever pushes. Before push:

```
cd /tmp/POT
git -c user.name="<your name>" -c user.email="<your email>" \
    commit --amend --no-edit --reset-author
```

(or amend with `--author="..."`) so the GitHub PR shows the correct
submitter.

What the patch adds:

| File | Change |
|---|---|
| `ot/partial/partial_solvers.py` | `+ entropic_partial_wasserstein_logscale` after `entropic_partial_wasserstein` |
| `ot/partial/__init__.py` | export the new function + add to `__all__` |
| `test/test_partial.py` | 4 new test functions, 10 parametrised cases total |
| `examples/unbalanced-partial/plot_entropic_partial_wasserstein_logscale.py` | Sphinx-Gallery example reproducing issue #723 + the fix |
| `docs/source/user_guide.rst` | one-paragraph mention next to `entropic_partial_wasserstein` |
| `RELEASES.md` | one-line entry under `0.9.7.dev0` |

Local verification (2026-05-16):

* `pytest test/test_partial.py -q` → 18 passed (8 originals + 10 new).
* `pytest test/` (full suite) → 1052 passed, 88 skipped, 4 xfailed —
  no regression outside `partial`.
* Example script runs end-to-end with `MPLBACKEND=Agg`; standard
  solver returns NaN at `reg ∈ {0.05, 0.01}` while the logscale
  solver stays finite over the whole sweep — issue #723's failure
  mode reproduced and fixed.
* `git diff master -- ot/partial/partial_solvers.py` shows exactly
  one hunk: the new function block. Nothing else drifts.

Reason this is *additive*, not a rebase of the existing PR branch:
the original PR was opened when `ot/partial.py` was a single file
(March 2025); since then the maintainers split it into the
`ot/partial/` package, so the diff against current master is +2k −13k
lines of mostly-unrelated movement. Re-applying just the new
function on the new layout is cleaner than fighting that rebase.

User decisions needed before step 8:

1. Fork to `hinanohart/POT` or to a personal account?
2. Open a *new* PR (default) or push to the existing PR #724 branch?
3. Author attribution — list as `Co-authored-by: wzm2256 <...>` (default,
   pro-social) or only credit in the PR description?

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
