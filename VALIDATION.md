# Source release validation

## Core update · 2026-11-25

Export source: `bd5d8b56ac32de4f645b06c5d28adf948044fff5`, branch `round1-mapping-body-search`. This export keeps the existing source-only CLI and separate local state; it does not export the web app or private galleries. Existing navigation safety limits and pretrained gait remain unchanged.

| Check in this open-source checkout | Actual outcome |
|---|---|
| Core safety, acoustics, candidate-room, body-hint and room-visit regressions | **64 tests passed** |
| Pretrained files | All 10 identity/VAD/KWS and optional body-model hashes verified |
| Actual model loading | CAMPPlus, YuNet, SFace, Chinese KWS, OSNet and YOLO11n-pose loaded in the prepared Jazzy / Python 3.12 runtime |
| Body registration entry point | Registered 12 independent rendered views of one avatar in temporary local storage, then recognized a held-out 13° / 3.4 m synthetic view; no live gallery modified |
| CLI | Help and new `enroll-body` arguments checked |
| Example resources | Household and community MJCF staged and loaded by MuJoCo, and acoustic YAML parsed; no physics run |
| Launch construction | Body gallery absent/present branches constructed; optional body process selected correctly; no processes launched |
| Model downloader | Local fixture verified SHA-checked archive extraction and optional-body exclusion; corrected Chinese KWS archive-hash metadata |
| Syntax | Python compilation, shell syntax and whitespace checks passed |

The registration check is a synthetic smoke test, not a person-search success rate. Full empty-map exploration and cross-room search were **not rerun in this export**; the currently running private workbench was not restarted. Native simulator code did not change in this update. No dependencies were upgraded or models downloaded.

### Source-project experiments retained separately

Compact metrics and source-result SHA-256 values: [`results/round1-summary.json`](results/round1-summary.json). The measurements below are historical source-project runs, not additional tests performed by this export.

| Mapping batch | Normal physical wall time: launch → exploration → saving → actual localization ready | Distance | Independently known reachable floor | Nodes |
|---|---|---|---|---|
| `home-final`, 168 m² / 3 rooms | 491.14 s; 4.87 min/100 m²; 2.73 min/room | 92.64 m | 99.9248% | 333 |
| `community-final`, 200 m² / 4 rooms | 440.23 s; 3.67 min/100 m²; 1.83 min/room | 71.77 m | 99.3633% | 264 |

Each layout has one final full mapping run, with physical room entry, saving and localization verified. Household mapping used the `1075a4c` exploration tree; community mapping added four initial looks and TF warm-up. They are not two repetitions of an identical revision. Floor coverage is not complete 3D surface coverage. Community's earlier configuration took 915.33 s / 133.02 m; the final run reduced time by 51.9% and distance by 46.0%, a single-run comparison across configurations.

| Search batch and source revision | Recorded result |
|---|---|
| `community-final-search`, `b92ac03` | 3/3 completed; 43.48 / 26.90 / 136.69 s; initial top3 2/3 |
| `community-candidates-search`, `1e106f8` | 3/3 completed; 39.80 / 19.70 / 55.56 s; initial remote top1 2/3 and top3 3/3. Third call had prior face memory |
| `home-candidates-search`, `5b2577f` | Initial top1/top3 3/3, but only 2/3 physical tasks succeeded; third task timed out |
| `home-body-view-search`, `114c8a5` | Two tasks succeeded (42.71 / 23.48 s); third audio expired before task start |
| `home-body-view-retry`, `114c8a5` | Isolated third-call retry exercised body-guided viewpoint and obtained a face, then failed bounded face recovery at 113.43 s |
| `home-body-view-nearby-retry`, `114c8a5` | Subsequent call from the already nearby stopped position succeeded in 13.60 s, separation 1.773 m. Not an original-start single-call success |

Do not combine these batches into one success rate. Latest household cold-start three-person acceptance is incomplete. The JSON also preserves older failed search batches.

Source-project appearance evaluation: 8 synthetic resources × 4 held-out directions × 2 distances = 64/64 known identities correct; 0/64 wrong accepts in the unknown-gallery condition. Pure OSNet also got known 64/64, with 3/64 unknown wrong accepts before colour assistance. This does not validate real people, clothing changes or hairstyle/height identification. Online body inference over 186 distinct frames had median 0.230 s and P95 0.362 s; median observation age 0.368 s, with no diagnostics beyond the 1.2 s age gate. These are body-node timings, not whole-system frame rates.

Known limits include occasional FastDDS/Nav2 startup failures recovered by restart, audio deadlines under load, bounded face recovery failures, room-segmentation errors and ambiguous corridor acoustics. The acoustic simulator remains an approximation; no real-building or physical-G1 validation is claimed. The public runtime's observed-room check is independent of simulator truth, and cannot establish full-scene coverage by itself.

## Original source extraction · 2026-11-03

Validation uses the existing prepared Ubuntu 24.04 / ROS 2 Jazzy dependency root on an Ubuntu 22.04 host. No host dependencies were upgraded.

| Check | Actual outcome |
|---|---|
| Python syntax and shell syntax | Passed |
| Native MuJoCo actor/sensor adapter | Compiled successfully from pinned upstream sources |
| Core safety/scheduling/acoustic regressions | **51 tests passed** |
| Recognition-model hashes | Four identity/VAD ONNX files and four Chinese KWS files verified |
| Actual pretrained model initialization | CAMPPlus, YuNet, SFace and Chinese wake word loaded |
| CLI face + voice enrollment | Passed with one detected rendered-avatar face and a 2.544 s existing call recording; speech duration 2.402 s, recording RMS −26.5 dBFS |
| Saved-map session, without any web service | Real RTAB-Map registration passed; Nav2 and navigation-map lifecycle active; fresh map and estimated map-frame pose available |
| Actual short navigation | Nav2 succeeded at goal `[0.8, 0.0]`; estimated final XY `[0.59582, −0.11204]`; error **0.23290 m**, FAST-LIO XY integral **0.68473 m**; 34 path poses, path length **0.93336 m** |
| Upper-body control | Existing arm trajectory controller activated; measured during this short session, maximum requested shoulder swing 0.2645 rad; leg policy unchanged |

The navigation metrics come from the unmodified core goal checker, using estimated TF/odometry. A short local navigation test is not a cross-room endurance result. Waveform regressions cover three controlled free-field bearings, level adjustment and silence rejection; these are not real-building acoustic tests.

During extraction, an empty optional `initial_pose` was initially passed as a malformed ROS launch argument. The CLI now omits the argument when empty; the subsequent actual session and navigation succeeded. The added acoustic test initially used continuous noise, which the speech-frame gate correctly rejected; the fixture now has quiet intervals. Algorithm thresholds were not relaxed.

The private source revision preceding extraction was `10e8d16e2ec6a6a30dacabd18d9c4e871a741f46`. Its final exhibition experiment had already completed fresh mapping and a specified three-person saved-map search sequence. Those historical runs are not added to the counts above. The source-only extraction has **not** rerun a full empty-map exploration or a full three-person end-to-end sequence. Fresh-machine installation has also not been fully reproduced.

Runtime exploration no longer depends on offline simulator-ground-truth scoring scripts. It preserves the upstream exploration tree, map saving and localization gate; it explicitly reports coverage as independently unmeasured. This keeps evaluation data outside operational planning.

No private voice waveform, face image, biometric gallery, map database, model weight, complete rosbag or old report is published with these checks.
