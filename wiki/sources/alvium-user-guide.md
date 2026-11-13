---
title: Alvium USB Cameras User Guide (V5.5.0)
sources: [raw/camera_docs/Alvium-USB-Cameras_User-Guide.pdf]
updated: 2026-08-31
---

# Alvium USB Cameras User Guide

Allied Vision's full user guide for the Alvium 1800 U USB3 camera family
(`raw/camera_docs/Alvium-USB-Cameras_User-Guide.pdf`).
~16,000 lines of extracted text (`pdftotext -layout`); this page covers only
the parts read so far — the model actually used in this thesis
([[alvium-1800-u-1240c]]) plus general mechanical/electrical notes that
apply across the family. Most of the guide (I/O wiring, GenICam feature
reference, mounting drawings, ESD handling) has **not** been read yet — see
"Not yet covered" below.

## What was read (this ingest)

- The full specifications table for the 1800 U-1240m/c model (guide p. 194–196,
  "Table 79: Alvium 1800 U-1240m/c specifications") — extracted verbatim into
  [[alvium-1800-u-1240c]].
- The focal-length-vs-field-of-view table for the same model (guide p. 267,
  "Table 134") — extracted into [[alvium-1800-u-1240c]].
- A general USB3 Vision / GenICam note: the camera exposes control via
  **GenICam (GenICam Access)**, standard interface for machine-vision
  cameras (guide line ~3489, "GenICam Applications Programming Interface
  (API). USB3 Vision standard").

## Not yet covered (flag for future ingest)

- Mechanical mounting/lens-flange dimension tables (guide area around line
  11395–11443) — a per-model dimension table exists but its column headers
  weren't captured cleanly by text extraction; needs a direct PDF read
  (page image) rather than `pdftotext` to interpret correctly. Don't trust
  any numbers pulled from that raw region without re-checking against the
  actual page.
- Full GPIO/trigger wiring instructions (relevant to the burst-capture
  trigger setup described in `docs/data_collection_pipeline.md`).
- Sensor cleanliness / optical cleaning procedures (guide has a dedicated
  section — relevant since the rig deliberately soils a *glass pane*, not
  the sensor itself, but cleanliness of the camera housing during rig
  assembly is still in scope).
- Vimba X software setup chapter (the rig already uses Vimba X Viewer per
  earlier project work, but the setup chapter itself hasn't been read here).

## Related

- [[alvium-1800-u-1240c]] — the camera model this thesis actually uses
- `docs/data_collection_pipeline.md` (project doc, not wiki) — already has
  an equipment table built from this same camera family, from earlier
  project work before this knowledge base existed
