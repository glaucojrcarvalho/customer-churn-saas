# Public release readiness review

Status: NOT READY FOR PUBLIC DEPLOYMENT. Source review dated 2026-10-17; baseline `d162c62e944c274e391c7e02b55bf7c37c1c04ab`.

## Findings

1. The training CSV has seven observations; no evaluation, calibration, data provenance or redistribution rights were established. Do not claim operational prediction quality.
2. `create_model.py` silently substitutes a dummy classifier on any training error. This must fail closed.
3. `/upload` reads an entire unbounded file into memory; `/predict` has no batch maximum or finite numeric range constraints. Public input must be bounded and validated, or public upload disabled.
4. The current GitHub workflow publishes container images on every `main` push. Split read-only pull request CI from explicitly authorized release publishing.
5. The committed pickle artifact has unverified provenance and integrity. Only load reviewed artifacts; never deserialize untrusted inputs.
6. The Docker build copies the entire context and defaults to root. Constrain build contents and runtime privileges.
7. No repository LICENSE was observed. Resolve licensing and dataset provenance before redistribution.
8. README claims production readiness without evidence. Label the future portfolio artifact as illustrative, using documented synthetic fixtures.

## Approval gates

- [ ] Source, history, binary, build context and output reviewed for confidential or personal material; automated secret scanning completed.
- [ ] Data and model provenance and legal redistribution rights confirmed; synthetic replacements documented if needed.
- [ ] Model generation fails closed, with integrity checks and explicit demonstration-only language.
- [ ] API payload, file and row limits and error handling covered by tests, or upload removed from public deployment.
- [ ] Read-only pull request CI, reviewed release-only publishing, pinned actions, vulnerability scanning and artifact inspection.
- [ ] Responsive accessible demo and browser regression tests, with no operational predictive claims.
- [ ] Exact-commit green CI, deployment isolation, HTTPS headers, cost limits, observability and rollback reviewed.
- [ ] Human approval before merge, release or deployment.

This was a textual source review of README, workflows, Dockerfile, Python app, tests and training CSV, not a runtime, history, binary or penetration test. No publication is authorized.
