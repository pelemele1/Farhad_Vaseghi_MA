---
title: Alvium Modular Concept
sources: [raw/camera_docs/Alvium-Modular-Concept_External.pdf]
updated: 2026-08-31
---

# Alvium Modular Concept

Allied Vision, V1.8.0 (2026-Jul-16). Full document read (766 lines — it's
short, mostly a hardware-options/pricing catalog rather than technical
narrative). Lists every optional hardware variant available across the
Alvium camera family and how they combine.

## What it actually is

A cross-reference catalog: for each modular option (lens mount type,
interface variant, filter, sensor treatment, housing color), it states
availability, typical lead time, and how the option combines with every
other option (Table 21: "Combining Alvium modular options" — a full
compatibility matrix). Not a specs document — for actual electrical/optical
specs it repeatedly points to the relevant camera User Guide instead
(e.g. [[alvium-user-guide]] for the 1800 U-1240c).

## Options catalogued (relevant subset)

- **Alvium G1 BL** (Board Level) — different connector position/no lens
  mount, for custom mechanical integration.
- **Alvium FP3/GM2** — extended-cable-length variants (up to 15 m) using
  FPD-Link III or GMSL2 serializers, for cases needing the camera far from
  the host.
- **Alvium CSI-2/USB Flex** — the same Flex option [[alvium-flex-accessory-guide]]
  covers in detail; this page just lists it as one of several modular
  options and confirms it combines with filter/sensor/housing options.
- **Alvium CSI-2/USB Frame** — a bare sensor unit precision-mounted into a
  frame instead of a housing, for custom lens/optical-system alignment;
  available in 180°/90° USB connector orientations.
- **Filter options** — color cameras normally ship with an IR-cut filter
  (removable on request); monochrome cameras normally ship *without* one
  (addable on request, "IRC 625 colored glass filter").
- **Sensors without cover glass** (RCG = Removed Cover Glass for housed
  cameras, TCG = Taped Cover Glass for bare-board, removed by the
  customer) — for higher quantum efficiency or attaching fiber optics
  directly, at the cost of losing the sensor's protective glass.
- **Black housing** — standard Alvium housings are red with a black back
  cover on closed-housing models; fully black housing is an option.

## Relevance to this thesis

Low-to-moderate — this thesis's rig uses standard housed Alvium 1800
U-1240c units (not Frame, not G1 BL, not FP3/GM2). The one relevant
open question this page raises: **is the rig's camera the standard USB
variant or the Flex variant?** (Same open question as
[[alvium-flex-accessory-guide]] — this document doesn't resolve it either,
it just confirms Flex is one of several available modular options, all
independent of the base 1800 U-1240c sensor choice.)

## Not covered here

Nothing significant — the whole document was read; it's short and this
page reflects its full content at the level of detail relevant to this
project (individual pricing/lead-time numbers and product codes are
intentionally omitted here as noise for a research knowledge base — see
the raw PDF directly if procurement details are needed).
