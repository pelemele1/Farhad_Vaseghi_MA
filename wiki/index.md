# Index

Catalog of every wiki page. Updated whenever a page is added or removed.

## sources/
- [alvium-user-guide](sources/alvium-user-guide.md) — Allied Vision's Alvium USB camera user guide; only the 1800 U-1240c spec/FOV tables read so far, rest flagged unread
- [alvium-flex-accessory-guide](sources/alvium-flex-accessory-guide.md) — the Alvium Flex hardware option (board-to-board connector, Add-on/Interface Boards); safety/connector-naming sections read, spec tables/assembly instructions not
- [alvium-modular-concept](sources/alvium-modular-concept.md) — full hardware-options/pricing catalog across the Alvium family; short, fully read
- [alvium-features-reference](sources/alvium-features-reference.md) — ~360-page GenICam feature reference; table of contents + DigitalIOControl (trigger/GPIO) category read, rest is a lookup reference, not narrative
- [soilingnet](sources/soilingnet.md) — Uřičář et al. 2019; the paper Stage B's tile/grid design is modeled on; full paper read
- [lin-2017-focal-loss](sources/lin-2017-focal-loss.md) — Lin et al. 2017; source of `FocalLossWithLogits`; full paper read
- [procedural-lens-soiling](sources/procedural-lens-soiling.md) — Zhang et al. 2025; source paper for the vendored `third_party/physical_lens_soiling/` code; full paper read
- [sidl](sources/sidl.md) — Choi/Park/Kim 2025; real physical dirty-lens-film capture methodology, directly comparable to this thesis's own rig concept; full paper read
- [yolov11-architecture](sources/yolov11-architecture.md) — Khanam & Hussain 2024; independent architectural analysis of the `yolo11m.pt` backbone family; full paper read

## entities/
- [alvium-1800-u-1240c](entities/alvium-1800-u-1240c.md) — the camera model used on this thesis's rig; full spec + FOV tables from the vendor guide; open question on Flex vs. standard hardware option

## concepts/
- [focal-loss](concepts/focal-loss.md) — the technique itself, generalized beyond the one paper; why it beats a flat class weight, where this project used it, an untried adjacent idea (bias init)

## analyses/
_(none yet)_

## Not-yet-ingested raw sources

All raw/ documents now have at least a wiki/sources/ page (as of this
ingest pass). Remaining gaps within already-ingested sources — not full
gaps, just flagged incomplete reads:
- [[alvium-user-guide]] — most of the ~16,000-line main user guide (I/O
  wiring, sensor cleanliness, Vimba X setup chapter) beyond the spec/FOV
  tables.
- [[alvium-flex-accessory-guide]] — Interface Board full specs, FPC
  cable/PCB part numbers, illustrated assembly instructions.
- [[alvium-features-reference]] — everything except `DigitalIOControl`;
  `AcquisitionControl` (exposure/trigger) and `AnalogControl` (gain) are
  the next most relevant categories if this gets ingested further.

The MIO-TCD benchmark paper (source of `data/raw/mio_tcd/`) is not on disk
at all — it's paywalled (IEEE Xplore only), see `raw/papers/README.md`.
