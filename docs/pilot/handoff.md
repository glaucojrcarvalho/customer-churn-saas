# Customer Churn portfolio pilot — handoff

## Attempt 1 — 2026-10-17

Base main commit: `d162c62e944c274e391c7e02b55bf7c37c1c04ab`.
Branch: `feat/portfolio-churn-release-readiness-20261017`.

Completed: independent textual review of README, CI workflow, Dockerfile, model builder, FastAPI routers, schemas, preprocessing, training, tests, and training CSV. Created `docs/pilot/release-readiness.md` with eight findings and explicit release gates. No deployments or merges.

Evidence: baseline main CI run `18872639805` succeeded on 2025-10-28; it is historical and does not validate this branch. Audit commit `92a7a39f5dcb2c352bb41acb2634dc0360a169b9`. No tests executed in this attempt; docs-only changes.

Open blockers: seven-row unvalidated dataset, unknown data/model provenance and redistribution rights, unbounded CSV/JSON input, silent dummy fallback, unverified pickle integrity, automatic container publication on main, no root LICENSE, insufficient Docker isolation, unreviewed dependency supply chain. Public deployment remains forbidden.

Next safe task: remove the README's unsupported production-ready claim, then harden CI so PR validation is read-only and container publishing requires a reviewed release. Keep all changes on this branch and update this handoff with exact commit/CI evidence. The README update was attempted but blocked by the connector; do not assume it changed.
