> [!WARNING]
> ## ⚠️ Realistic contamination severity — do not overdo it
> **The contaminations must reflect real-world conditions.** The network has to learn to detect **weak and subtle** contaminations as well — these are the normal case in operation and the hardest for the model.

## Different patterns/severities per contamination type
Use different contamination patterns in different strength.

## Different distances per contamination type
Vary the distance between objectiv and glas.

### Spray-based experiments (water drops, mud)
In spray-based experiments, droplets are sprayed onto the pane with a toothbrush.
The spraying procedure is as follows: dip the toothbrush into the liquid, hold it approx. 15 cm in front of the pane and run your thumb backwards over the bristles so that small droplets are sprayed forward.
Severity definition (re-dip the toothbrush before each spray):

| Severity | Spray strokes |
|---|---|
| light | 1× |
| medium | 2× |
| heavy | 4× |


1. Water drops: spray water onto the panes at different severities, possibly using a toothbrush

2. Mud: like water drops, but also simply smeared onto the pane. Brown, partly opaque splashes, as caused by rain/road spray
How: finely sieve garden soil and mix it with water into a thin suspension


### Non-spray-based experiments
3. Fogging: vaporizer? Breathing on the pane? Holding hot water in front of it => fogging with water vapour?

4. Cleaning streaks: wet the glass with cleaning agent/soap and wipe it off poorly

5. Scratches: sandpaper for polycarbonate panes; glass: possibly asphalt?

6. Dust: fine flour

## Recording process (stereo setup)
Principle: two cameras film the same scene (approx. 10 m away) **simultaneously**. A clean pane sits in front of one camera, the contaminated pane in front of the other. A homography H maps the clean image onto the dirty image → every dirty image has a pixel-aligned clean counterpart, even when the scene is moving.

**A. Setup (once)**
1. Mount both cameras (same model) rigidly side by side, **baseline as small as possible**, same viewing direction
2. Set both cameras to manual with **identical values**: exposure, gain, white balance, focus (on the scene), resolution, compression, day/night mode. Only readjust if the image clips (sky blown out / dark areas black) — then change both cameras identically
3. Synchronize recording (hardware trigger or common timestamp)
4. Identical pane holders in front of both cameras, **same pane–lens distance** on both (when varying the distance, change both)
5. Optional: a few checkerboard shots per camera — only needed if the alignment residual grows towards the image edges (lens distortion)


**B. Per session (one contamination type, one severity)**
1. Clean pane in front of both cameras (**same material and thickness as the test pane**) → **10 clean/clean pairs**. From these, estimate H (per-camera median image, feature matching + RANSAC) and the residual = error floor. Also checks the cleaning
2. Remove the test pane and contaminate it **away from the camera** (do not touch the holder in front of the reference camera)
3. Take a phone photo of the test pane in transmitted light against a bright surface
4. Re-insert the test pane
5. Record **10 synchronized image pairs** (or 30 s of video for fogging); the scene may move (person walking through the frame), but **keep the near range < 5 m clear** (parallax)
6. Fill in the log (see below); clean the pane, set it aside to dry, or — for cumulative types (scratches) — continue directly to the next severity

**Pane swap:** Alternate the dirty camera regularly (even session no. → camera B dirty, odd → camera A dirty) so that the network does not learn the characteristics of one camera as "dirty".

**Fogging:** instead of 10 image pairs, record synchronized video from both cameras until the pane is clear again.

**Log per session:** date/time, session no., dirty camera (A/B), material, pane distance, type, severity, repetition, pane ID, day/IR, remarks (weather, lighting, errors)

**Different severities:** The 6 contamination types should be recorded at different severities

**Domain:** Recordings should capture typical top-down surveillance camera scenes. Different weather, time of day.
