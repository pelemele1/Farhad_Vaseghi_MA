# papers

Academic papers that matter to this thesis's research and design decisions
— e.g. the source for a technique already used in the code (SoilingNet,
focal loss, the YOLO backbone) or background/related work for the writeup.

| File | Authors, year | Description | Ingested into wiki |
|---|---|---|---|
| `12883-ChoiS.pdf` | Sooyoung Choi, Sungyong Park, Heewon Kim (Soongsil University) — AAAI-25 (2025) | "SIDL: A Real-World Dataset for Restoring Smartphone Images with Dirty Lenses" — 1,588 real degraded/clean image pairs across 300 scenes, contaminants incl. water drops, fingerprints, dust, scratch; benchmarks several restoration models (AirNet, NAFNet, Restormer, etc.) on it. Relevant as related work: a real-world (not synthetic) dirty-lens dataset, complementary to this thesis's synthetic approach, and its physical film-based capture rig is directly comparable to this thesis's own Part 1 rig. | [[sidl]] |
| `Zhang et al. - 2025 - Procedural generation of lens soiling data via physics-based simulation.pdf` | Song Zhang, Rui Zhang, Jian Li, Xuefeng Li — *The Visual Computer* 41:10433–10449 (2025), Springer | "Procedural generation of lens soiling data via physics-based simulation" — physics-based synthetic lens-soiling generation (mud/water stains, droplet refraction, lens flare) via OpenCV. **This is the source paper for `third_party/physical_lens_soiling/`** — the vendored code this thesis's own `add_dirt`/`add_water` effects (`src/soiling/effects.py`) are built on (see `third_party/physical_lens_soiling/NOTICE.md`). High-value ingest: documents the physical/optical reasoning behind code already in production use here. | [[procedural-lens-soiling]] |
| `Uricar et al. - 2019 - SoilingNet Soiling Detection on Automotive Surround-View Cameras.pdf` | Michal Uřičář, Pavel Křížek, Ganesh Sistu, Senthil Yogamani — ITSC 2019 (arXiv:1905.01492) | "SoilingNet: Soiling Detection on Automotive Surround-View Cameras" — the paper Stage B's tile/grid classification design (architecture.md §2) is explicitly modeled on ("SoilingNet approach", per this project's own planning). CNN + multi-task learning + GAN-based data augmentation for coarse-localization soiling detection. | [[soilingnet]] |
| `Lin et al. - 2017 - Focal Loss for Dense Object Detection.pdf` | Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár — ICCV 2017 / RetinaNet paper (arXiv:1708.02002) | "Focal Loss for Dense Object Detection" — the loss function behind `FocalLossWithLogits` in `src/models/losses.py` (Session 17's fix for Stage B's scratch-class precision collapse under extreme tile imbalance). Already cited by name in that module's docstring; this is the actual source. | [[lin-2017-focal-loss]] + [[focal-loss]] (concept) |
| `Khanam and Hussain - 2024 - YOLOv11 An Overview of the Key Architectural Enhancements.pdf` | Rahima Khanam, Muhammad Hussain (arXiv:2410.17725) | "YOLOv11: An Overview of the Key Architectural Enhancements" — independent architectural analysis of YOLO11 (C3k2 block, SPPF, C2PSA), the backbone family `FrozenYOLOBackbone` loads (`weights/yolo11m.pt`). Note: Ultralytics never published an official YOLO11 paper (unlike v8/v9/v10); this is the closest available technical reference, not a primary source from the model's authors. | [[yolov11-architecture]] |

**Not added — paywalled:** the MIO-TCD benchmark paper ("MIO-TCD: A New
Benchmark Dataset for Vehicle Classification and Localization", Luo et al.,
*IEEE Transactions on Image Processing* 27(11):5129–5141, 2018 —
[IEEE Xplore](https://ieeexplore.ieee.org/document/8387876/)) — the dataset
`data/raw/mio_tcd/` is sampled from — has no free/open-access copy; only
available via IEEE Xplore (institutional access, e.g. FAU's library, would
be needed). Not downloaded from an unofficial source.

Add a paper here directly — drop the PDF in this folder and add a row
above (filename, authors + year, one-line description of why it matters to
this thesis, and its ingest status). See `../../CLAUDE.md` for the full
raw/ convention and the ingest workflow.
