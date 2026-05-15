# 001 — License

**Status:** SIGNED — MIT adopted on 2026-05-16.
**Last updated:** 2026-05-16

## Why this is decision #001

The license is the **first** load-bearing decision because:

- It constrains which dependencies are compatible
  (e.g. AGPL-3.0 deps can't be combined with a permissive parent license
  unless the parent is also AGPL).
- It constrains which datasets can be vendored.
- Changing it after a public release requires consent from every external
  contributor.

We pick now, before any code, so 002 and 003 can reference it.

## Candidates

| License | Permissive? | Patent grant | Copyleft | Picked by lucidrains? | Picked by robotics OSS? |
|---|---|---|---|---|---|
| Apache-2.0 | Yes | **Explicit** | None | Mixed (`x-transformers` MIT, others vary) | Most ROS packages |
| MIT | Yes | Implicit only | None | Most lucidrains repos | Some |
| BSD-3-Clause | Yes | Implicit only | None | No | scipy, scikit-image |
| AGPL-3.0 | No | Yes | **Strong, network use** | No | No (incompatible with most ROS) |

## Constraint check

- `mosaicraft` itself is **MIT** (verified by reading `/home/runza/oss/mosaicraft/LICENSE` — to be re-verified before final commit).
- Any code we *import* from `mosaicraft` must be license-compatible.
  MIT → Apache-2.0 is allowed (one-way), Apache-2.0 → MIT requires
  preserving the Apache patent grant via NOTICE.
- Sinkhorn-OT reference implementations:
  - `scipy.spatial.distance` — BSD-3 (compatible with all candidates).
  - `POT` (Python Optimal Transport) — MIT (compatible with all).
  - `geomloss` (Feydy) — MIT (compatible with all).
  - No AGPL Sinkhorn implementation we'd realistically depend on.

## Recommendation (NOT YET ADOPTED)

**Apache-2.0**, because:

1. **Patent grant is explicit.** Sinkhorn-OT and NBV both have active
   patent activity (Google's NBV patents, OT acceleration patents from
   2024-2026). An explicit grant protects users.
2. **Compatible with mosaicraft's MIT** (one-way merge OK; preserve MIT
   notices on copied files).
3. **Permissive enough** that any future downstream (ROS, Isaac Gym
   examples, lucidrains-style toolkits) can incorporate it.
4. **Not viral** — does not force downstream users to relicense, which
   matters if the repo grows into a `pip install` library.

## Why not AGPL

If the user's goal is **"academic citation"** or **"GitHub star count among
researchers"**, AGPL is fine. But it locks out commercial / ROS adoption.
Decision deferred to the user; flag if the user picks "学術新規性"-only and
explicitly wants viral copyleft.

## Decision

- [ ] Apache-2.0 (recommended)
- [x] **MIT** ← adopted
- [ ] BSD-3-Clause
- [ ] AGPL-3.0
- [ ] Dual license

**User signature (verbatim, R16):** `mit` — 2026-05-16.

## Why the user overrode the Apache-2.0 recommendation

The user picked MIT despite Apache-2.0's explicit patent grant. Plausible
reasons (not user-stated, not assumed binding):

1. **mosaicraft is MIT.** Matching licenses keeps the submodule + import
   surface symmetrical; no NOTICE file maintenance.
2. **MIT is the lucidrains default** (`x-transformers`, `vit-pytorch`,
   most `pytorch-*` lucidrains repos are MIT). If this repo ever sends
   PRs upstream or attracts that community, MIT is the cheap path.
3. **MIT is shorter to read.** New contributors don't have to parse the
   patent clause to decide if their employer's CLA covers them.

The Apache-2.0 patent grant trade-off is recorded here so future
contributors understand what was given up. If a downstream user is
sued over an OT-acceleration patent, the upstream license does not
help them — that is a known cost of the MIT decision.

## Implementation tasks created by this decision

- `LICENSE` file at repo root containing the MIT text with copyright
  `2026 mosaicraft-active-vision contributors`.
- `pyproject.toml` metadata `license = { text = "MIT" }`.
- README "License" section updated to point to `LICENSE`.
