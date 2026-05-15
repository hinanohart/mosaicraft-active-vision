"""mosaicraft-active-vision: photomosaic-driven next-best-view planning.

Phase 1 has not landed yet — `decision/000-charter.md` blocks code exports
until 001/002/003 are signed AND a Phase-1 demo runs end-to-end.

Decisions 001 (MIT license), 002 (Mosaic-SSIM-gain primary metric), and
003 (Sinkhorn-OT matching, no Hungarian fallback) are signed as of
2026-05-16. Code modules will be added one at a time, each gated by
the charter's "no unverified numeric claims" rule.

This file deliberately exports nothing. Importing the package only gives
you the version string. That is intentional, not a stub-in-progress.
"""

__version__ = "0.0.0"
__all__: list[str] = []
