---
title: Alvium Flex Design and Accessory Guide
sources: [raw/camera_docs/Alvium-Flex_Design-Accessory-Guide.pdf]
updated: 2026-08-31
---

# Alvium Flex Design and Accessory Guide

Allied Vision, V1.1.1 (2025-Jul-11). Covers the **Alvium Flex** hardware
option — a variant of the Alvium camera line that replaces the standard
interface/I/O connectors with a board-to-board connector for custom
integration. Read: safety/ESD chapter, connector naming, Add-on Board specs
(CSI-2). Not read in detail: Interface Board Screw-on/Compact full specs,
FPC cable part-number tables, PCB/cable numbers, the illustrated assembly
instructions (pages 44–52).

## What "Flex" actually changes

Standard Alvium USB cameras connect via a USB 3.0 Micro-B connector.
**Alvium USB Flex** cameras instead expose a Hirose DF40C-50DP-0.4V
board-to-board connector, meant to attach a small **Add-on Board** (screwed
directly onto the camera) which then breaks out to either:
- a bare **Hirose TF38** FPC connector (roll-your-own cabling), or
- an **Interface Board** (Screw-on or Compact variant) that restores a
  standard USB-A/JST connector at the end of an FPC cable, letting you
  relocate the USB connector away from the camera body.

This exists for cases where the standard USB Micro-B connector's position
or cable stiffness doesn't fit the mechanical design — e.g. separating
connectors from the camera back panel, or avoiding "clumsy USB cables" in a
tight enclosure (the guide's own phrasing).

**Open question — does this thesis's rig use the Flex variant?** Not
confirmed. `docs/data_collection_pipeline.md`'s equipment table just lists
"Alvium 1800 U-1240c" with a USB3 interface, no mention of Flex/Add-on
Board/Interface Board. Worth checking against the physical rig hardware
directly (or the Vimba X Viewer's exact model string) before assuming
either way — see [[alvium-1800-u-1240c]].

## Safety/handling notes relevant to rig assembly

- **ESD**: bare-board/incomplete-housing cameras (which is what an Add-on
  Board sits directly on) are especially ESD-sensitive — the guide warns
  electrostatic discharge can damage specific pixel groups, visible as
  bubbles/blobs in captured images. Recommends: grounded body before
  unpacking, static-dissipative mat, wrist strap, ESD-protective clothing
  and housing during assembly.
- **No hot-plugging**: Alvium Flex cameras must have power disconnected
  before connecting/disconnecting FPC cables — hot-plugging can destroy the
  camera via inrush current.
- **FPC cable handling**: minimum bend radius 10 mm; connectors rated for
  only ~20 mating/unmating cycles (designed for one-time installation, not
  repeated reconnection) — relevant if the rig's rig-building process
  involves repeatedly reseating cameras during setup/testing.
- **Add-on Board identification**: CSI-2 and USB Add-on Boards look
  identical but have different pin assignments — marked with a "C" or "U"
  letter respectively; connecting the wrong type can short-circuit and
  damage the camera.

## Not covered here

- Full electrical/mechanical specs and technical drawings for the
  Interface Board Screw-on/Compact variants (pages 37–42).
- FPC cable part numbers and PCB numbers (pages 43).
- Step-by-step assembly instructions with drawings (pages 44–52) —
  relevant if/when the rig's physical assembly documentation
  (`docs/data_collection_pipeline.md`) needs exact connector orientation
  guidance.
