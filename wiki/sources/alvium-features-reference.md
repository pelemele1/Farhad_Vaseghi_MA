---
title: Alvium Features Reference
sources: [raw/camera_docs/Alvium_Features_Reference.pdf]
updated: 2026-08-31
---

# Alvium Features Reference

Allied Vision, V3.6.0 (2026-May-20). A full GenICam/GenTL parameter
reference (~360 pages, hundreds of individual features) — organized exactly
like Vimba X Viewer's feature tree: `Transport Layer` / `Interface` /
`Local Device` / `Camera` modules, each broken into categories
(`AcquisitionControl`, `AnalogControl`, `DigitalIOControl`,
`ActionControl`, `TriggerControl`-adjacent features, `AutoModeControl`,
`ChunkDataControl`, `EventControl`, `SequencerControl`, `PtpControl`,
`TransferControl`, etc.), each category listing its individual features
alphabetically with access type, unit, and value range.

**This is a reference manual, not narrative content** — consulted on
individual features as needed, not read cover-to-cover. This ingest read
the table of contents in full (to know what's in here) plus one category
in detail (`DigitalIOControl`, directly relevant to the rig's trigger
wiring). Every other category is unread; use the table of contents
structure above to find the right page when a specific feature is needed.

## DigitalIOControl — read in full, most relevant category for the rig

Controls the camera's physical I/O lines (used for external trigger
wiring, referenced in `docs/data_collection_pipeline.md`'s trigger setup):

- **`LineSelector`**: picks which physical line to configure — `Line0`,
  `Line1`, and (GigE/USB only, not CSI-2) `Line2`, `Line3`.
- **`LineMode`**: `Input` or `Output`, per selected line.
- **`LineSource`** (output lines only): what signal to drive out —
  options include `AcquisitionActive`, `ExposureActive`,
  `FrameTriggerWait` (useful for confirming the camera is actually armed
  and waiting on an external trigger), `Line0Signal`/`Line1Signal` (pass
  another line's state through), and several counter/timer signals.
- **`LineInverter`**: invert the signal polarity of a line.
- **`LineDebounceMode`**/`LineDebounceDuration`: for input lines, filters
  out signal noise shorter than a configurable duration (0.019–39.57 µs
  range) before it's accepted — relevant if the rig's external trigger
  circuit is electrically noisy.
- **`LineStatus`**/`LineStatusAll`: read current line state(s).

Two lines only on Alvium CSI-2 cameras (Line0/Line1); all four available on
GigE/USB — this project's cameras are USB, so all four lines should be
usable, unconfirmed against the actual hardware.

## Not covered here (the other ~350 pages)

`AcquisitionControl` (`ExposureTime`, `TriggerMode`, `TriggerSource`,
`TriggerDelay`, `AcquisitionFrameRate`) and `AnalogControl` (`Gain`,
`BlackLevel`, `Gamma`, `BalanceRatio`) are the two other categories most
likely to matter for this project's capture settings
(`docs/data_collection_pipeline.md`'s exposure/gain log columns) — flagged
here as the next things worth reading if this document gets ingested
further, but not read this pass. Everything else (Action/Event/Sequencer/Ptp
control, ChunkData, GigE-specific transport-layer features) is out of
scope for a USB-interface rig with no burst-sequencing or PTP
synchronization need identified so far.
