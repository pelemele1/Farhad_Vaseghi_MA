# Data Collection Pipeline (Part 1)

**Status:** ready for review, not yet executed in the field.

This document is the step-by-step field protocol for capturing the real dataset with the
two-camera rig. [`setup.md`](../setup.md) states the design rules (glass pane, sync, mapping,
pilot size) — this document turns those rules into something you can actually follow on-site.

---

## 1. Equipment

**Cameras:** 2× Allied Vision Alvium 1800 U-1240c (confirmed on the actual hardware via Vimba X
Viewer, serial `05HYK`).

| Spec | Value |
|---|---|
| Sensor | Sony IMX226, color, 1/1.7" |
| Resolution | 4024 × 3036 (12.2 MP) |
| Pixel size | 1.85 µm |
| Shutter | Rolling or Global Reset — **use Global Reset** (avoids the two cameras capturing slightly different instants when something in the scene is moving) |
| Max frame rate | 35 fps free-run, ~17 fps in triggered mode |
| Interface | USB3, controlled via GenICam / Vimba X |
| Trigger input | 4 GPIOs, external trigger supported, 0–5.5 V (accepts a "high" signal of 2.0–5.5 V) |
| Weather sealing | IP30 only — dust protection, **no water protection** |
| Lens | not included — must be bought separately (C/CS/S-mount) |

**Lens:** pick a focal length that gives a reasonable field of view at 10–20 m. From the
datasheet's FOV table, scaled to our distance:

| Focal length | Width of view at 10 m | Width of view at 20 m |
|---|---|---|
| 8 mm | ~9.2 m | ~18.4 m |
| 12 mm | ~6.1 m | ~12.2 m |
| 16 mm | ~4.6 m | ~9.1 m |
| 25 mm | ~2.9 m | ~5.8 m |

**Recommendation: 12 mm or 16 mm.** That gives a framing similar to the traffic-camera images
already used in Stage A (a few meters to ~12 m wide). Confirm before buying — see §7.

**Glass pane holder:** a small bracket/filter holder mounted in front of the right camera's lens,
sized to fit clear glass or acrylic discs. Must be swappable in seconds without moving the rig.
Not something the camera comes with — build it.

**Rig:** both cameras mounted rigidly on one shared bar/plate, fixed distance apart. Measure that
distance once the rig is built and record it — never change it without re-doing calibration
(§3, step 1).

**Trigger wiring:** wire both cameras' trigger-in pins to one shared external pulse source (e.g. a
small microcontroller or a signal generator), so both cameras fire on the exact same electrical
pulse. The camera manual only documents single-camera triggering — this wiring is a custom part
you build, not a built-in feature.

**Weatherproofing:** the camera bodies have no water rating. Keep them under a rain shield/housing
at all times. Water distortion is applied only to the pane in front of the lens — never spray or
rain directly on the cameras themselves.

---

## 2. Preparing the distortion panes

Each pane is a separate small piece of glass/acrylic. Keep several clean spares, plus dedicated
panes for each distortion type below. Give every pane a unique ID (write it on the frame/holder).

| Type | How to apply | Light severity | Heavy severity |
|---|---|---|---|
| **Dirt** | Mix a mud/dust slurry (fixed recipe), apply with a brush or sponge, let dry the same amount of time every time | ~10–20% of the pane covered | ~40–60% covered |
| **Water** | Fixed number of sprays from a spray bottle (or a controlled drip), distilled water to avoid residue — capture immediately, before it evaporates or runs | a few isolated droplets | dense droplets or a thin film |
| **Scratch** | Scratch with a fixed sandpaper grit or blade, fixed number/pressure of passes — this is permanent, so use dedicated pre-scratched panes, not the clean ones | 1–3 short, thin scratches | 5+ scratches, or longer/deeper ones |

Keep the recipe (slurry ratio, spray count, sandpaper grit) the same every time you prepare a new
pane of a given severity — that's what makes "light" and "heavy" mean the same thing across
sessions.

---

## 3. Choosing sites, weather, and camera height

**Sites — where to shoot:**

- Look for traffic scenes: streets, junctions, parking lots, campus roads — vehicles/pedestrians
  visible at 10–20 m. Same style of scene as the MIO-TCD images already used in Part 2.
- Public or permitted ground only (public streets, campus grounds, parking lots you're allowed to
  be in). Skip private property without asking first.
- Prefer a mostly static background (buildings, road, parked cars) — matches the "negligible
  parallax" assumption `setup.md` already relies on. Avoid scenes dominated by fast-moving objects
  close to the camera.
- Point away from building entrances / crowds where you can — real people and license plates will
  be in these photos, unlike the synthetic Part 2 data. Confirm what's required (blurring faces
  in post, site choice, etc.) with your supervisor/institution before large-scale capture; this
  isn't something to decide unilaterally.
- Don't treat every scene as a brand-new location: pick **~10–20 fixed sites** you can return to,
  then revisit each one across several sessions (different weather, light, time of day) to build
  up the scene counts in §5. Scouting a new site every single session isn't realistic solo.

**Weather / lighting — which conditions:**

- Cover at least three, spread roughly evenly across sessions: **sunny/clear**, **overcast/
  cloudy**, and one **lower-light** condition (early morning or dusk — check in the field that
  the camera's exposure range still handles it; don't push into full night).
- Skip actual rain, snow, or fog — the cameras are IP30 only (no water rating) and won't survive
  real precipitation without a separate purpose-built enclosure. This does **not** limit "water"
  training data: that comes from water applied to the pane on purpose (§2), not from shooting in
  the rain.
- Log weather as one word from a small fixed set (`sunny` / `overcast` / `dusk` / ...), not a
  free-text description — keeps the log filterable later.

**Camera height — how high off the ground:**

- Real traffic cameras sit 3–6 m up on poles — out of reach for a two-camera rig on a tripod. Use
  two practical heights instead:
  - **~2–2.5 m** (extendable stand, or an elevated spot — balcony, footbridge, parking-garage
    upper level) — closer to a real traffic-camera angle. Use this as the default.
  - **~1.2–1.5 m** (eye-level tripod) as the fallback wherever elevated access isn't available —
    also adds useful viewpoint diversity.
- Height is fixed for the whole session (the rig never moves once calibrated — §4). Record the
  exact height used in the capture log.
- Check the mount is actually stable at that height (weighted base, no tipping risk) — the rig
  moving mid-session invalidates that session's calibration.

---

## 4. Running one capture session

Do these steps in order, without moving the rig once calibration is done:

1. **Calibrate.** Mount a clean pane. Capture checkerboard/Charuco calibration shots with both
   cameras. Do this every session, and any time the rig gets bumped or moved.
2. **Lock camera settings.** With the clean pane still on, set exposure, gain, and white balance
   manually from the clean scene, then leave them fixed for the rest of the session. Do not let
   auto-exposure re-adjust after a pane swap — otherwise a change in brightness caused by blur or
   haze could get mistaken for the actual distortion signal.
3. **Capture the clean reference.** Take a burst of frames (e.g. 5) with the clean pane still on.
4. **Swap panes and repeat.** For each prepared pane (dirt light/heavy, water light/heavy, scratch
   light/heavy), mount it on the right camera only, take a burst of frames, then move to the next
   pane. The rig stays put the whole time.
5. **Log everything** (see §6) before packing up.

Site, weather, and height are chosen before step 1 (see §3) and stay fixed for the whole session.

---

## 5. How many images to capture

**Phase 1 — pilot.** Use the number already set in `setup.md`: about 15–30 scenes, 2–3 distortion
types, 2 severity levels each. Goal: prove the whole pipeline works (pane logistics, calibration,
timing) before committing to more.

**Phase 2 — full dataset.** A rough target sized to be usable for the thesis and a paper, given
this is manual field work (not an automated rig):

- ~150–250 distinct scenes (different locations/sessions)
- 7 pane states per scene: clean + 3 types × 2 severities
- a short burst (e.g. 5 frames) per pane state

That lands around 5,000–9,000 image pairs total. This is a starting number, not a fixed rule —
adjust it to how much field time is actually available (see §7).

**Splitting into train/val/test:** split by scene, never by individual frame, so no two frames of
the same location end up on both sides of a split. Same rule already used for the Stage A dataset.

---

## 6. Capture log

Record one row per burst (not per individual frame — frame index inside the burst is its own
column):

| Field | Example |
|---|---|
| Session ID | `2026-09-01_site3` |
| Scene ID | `scene_07` |
| Site ID | one of the ~10–20 fixed sites, §3 |
| Location | address / GPS |
| Timestamp | |
| Weather | sunny / overcast / dusk |
| Camera height | ~2–2.5 m or ~1.2–1.5 m (§3) |
| Pane ID | `dirt_heavy_02` |
| Distortion type | clean / dirt / water / scratch |
| Severity | light / heavy / n/a |
| Exposure, gain | as locked in step 2 |
| Calibration reference | which session's calibration shots apply |
| Burst frame count | e.g. 5 |

This log is what defines the labels — `architecture.md` already assumes a log like this exists;
this is its actual schema.

---

## 7. File storage

- Save lossless images (PNG, or raw + lossless debayer) — not JPEG, so compression artifacts
  don't get mistaken for distortion.
- Folder layout: `data/raw/rig_capture/<session_id>/...`, matching the existing
  `data/raw/mio-tcd/` convention already used in this repo.

---

## 8. Decisions still open

These need your input before fieldwork starts — each has a recommendation above, but needs
confirming:

- Exact lens (focal length + mount) — recommendation: 12 mm or 16 mm.
- Rig baseline distance — only known once the rig is physically built.
- The actual list of ~10–20 sites (§3) — needs scouting on the ground.
- Trigger hardware (what generates the shared pulse).
- Final phase-2 scale number — depends on how much field time you can commit.
- Privacy/ethics sign-off for capturing identifiable people/plates (§3) — check with your
  supervisor/institution before large-scale capture.
