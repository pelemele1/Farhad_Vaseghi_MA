---
title: Overview
updated: 2026-08-31
---

# Overview

Very early — this knowledge base was just scaffolded (2026-08-31), so it
currently knows about exactly one thing: the [[alvium-1800-u-1240c]] camera
this thesis's rig is built on, sourced from the vendor's official user
guide.

Nothing here yet on the ML side (SoilingNet, focal loss, YOLO backbones,
related work) — this knowledge base is meant to grow to cover that too as
sources get ingested, but the thesis's own code/architecture decisions
already live in `architecture.md` and `docs/development_log.md` at the repo
root, not here. This wiki is for *external* reading material and its
synthesis, not a restatement of project decisions already documented
elsewhere in the repo.

## What to add next (open, not prescriptive)

- The remaining 3 Alvium PDFs (Flex accessory guide, modular concept,
  features reference) — manifest entries already exist, not yet ingested.
- Any papers behind the thesis's technique choices — SoilingNet (Stage B's
  namesake per architecture.md), the focal loss paper (Lin et al. 2017,
  already cited in `src/models/losses.py`'s docstring but not synthesized
  here), YOLO backbone papers.
