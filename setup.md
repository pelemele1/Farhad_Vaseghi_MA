# Setup: Hardware & Mounting

- Two cameras, close together, rigidly mounted on a shared rig (fixed baseline for the entire data collection)
- Hardware-triggered synchronization (both cameras fire at exactly the same time)
- Left camera = reference (ground truth), right camera = distortion camera
- Scenes are captured at 10–20 m distance (object distance)

## Reference vs. Distortion

- The right camera always has a glass pane in front of the lens
  - clean → reference / clean capture
  - prepared (dirt, water, scratches) → distortion capture
- This prevents the pure glass effect (refraction, slight blur) from being falsely learned as "distortion"

## Camera-to-Camera Mapping (right → left)

1. Per camera individually: lens undistortion (checkerboard/Charuco calibration, remove lens distortion)
2. Then: a single global homography between the undistorted images, estimated via feature matching (SIFT/ORB + RANSAC) on real scenes at working distance
3. Because of the large object distance (10–20 m), parallax is negligible → a single static homography is sufficient, no depth-dependent/dense mapping needed
4. Validate mapping accuracy quantitatively (reprojection error / photometric residual) before generating distortion data

## Data Generation

- The right image (with distortion) is transformed into the left coordinate system via the learned mapping
- Result: an almost pixel-aligned image pair → (clean, distorted)
- No additional mask/segmentation label needed, image pairs only

## Target Model

- Training objective: both distortion detection and restoration

## Procedure

- First a pilot dataset (~15–30 scenes × 2–3 distortion types × 2 severity levels) to validate the entire pipeline
- Then scale up to a larger, more diverse dataset (more scenes, lighting/weather conditions, distortion variants)
- If the setup is ever rebuilt: new calibration + validation per session, tag the captures accordingly
