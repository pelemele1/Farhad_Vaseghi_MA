# Conventions

Working preferences for this specific knowledge base (as opposed to the
general schema in `../CLAUDE.md`).

- **No fabrication from memory.** Even when a fact (e.g. camera specs) was
  already established earlier in this project's own docs, a wiki `sources/`
  or `entities/` page should be grounded in an actual re-read of the raw
  material when first created, not copied from memory/recollection. If a
  number can't be verified cleanly (e.g. a table whose columns didn't
  survive text extraction), say so explicitly in the page rather than
  guessing — see [[alvium-1800-u-1240c]]'s "Open questions" section for the
  pattern.
- **Partial ingests are fine and should say so.** A source document (like
  the ~16,000-line Alvium user guide) doesn't need to be read cover-to-cover
  in one pass. The `sources/` page should list what was actually read and
  what wasn't, so a later ingest knows where to pick up.
- **This wiki does not duplicate `docs/`.** This project's own generated
  reports (`docs/development_log.md`, `docs/stage_a_final_report.md`,
  `docs/data_collection_pipeline.md`) stay where they are and are not
  re-synthesized into wiki pages — they're project output, not external
  reading material.
