# LiDAR to event camera extrinsics

This project corrects the extrinsic between a LiDAR and an event camera without a calibration target, by aligning depth edges in the LiDAR scan with edges in the event stream. The extrinsic is the rigid transform $`T_{\mathrm{camera} \leftarrow \mathrm{lidar}}`$ that moves a LiDAR point into the camera frame. A small error in it shifts projected depth away from the edges the event camera sees.

## Results

Two sequences of [VECtor](https://star-datasets.github.io/vector/) were used, `school_scooter1` and `corridors_dolly1`. The time offset between the LiDAR and the event camera and the rotation correction to the FARO extrinsic were estimated together, with the translation held at FARO.

| Sequence | LiDAR time offset | Rotation from FARO (x, y, z) | Score at 0 ms and FARO → best |
|---|---|---|---|
| `school_scooter1` | +92 ms | -0.50°, -1.13°, -0.38° | 2.152 → 2.838 |
| `corridors_dolly1` | +44 ms | -0.50°, -1.13°, -1.13° | 3.157 → 3.722 |

The main findings are these.

1. **The LiDAR clock is off.** The LiDAR stamps run behind the event camera by about 90 ms on `school_scooter1` and about 44 ms on `corridors_dolly1`, although VECtor lists the sensors as hardware synchronized. The offset differs between recordings, so it has to be estimated for each one. Correcting it raised the edge score by 32% and 18%, more than any rotation change did at 0 ms.
2. **FARO is off by about 0.5° in pitch and 1.1° in yaw.** Both sequences give the same pitch and yaw once the offset is corrected, and the even and odd halves of each sequence agree within 0.13° in yaw.
3. **Roll and translation are not observable from edges here.** Roll lands between -0.38° and -1.13° depending on the sequence and the half. Shifts of up to 30 cm change the score by at most 12%, so the translation is left at FARO.

The offset on `corridors_dolly1` is less certain than on the scooter. Its score curve has a second peak at +120 ms that scores within 1.5% of the best, and the gyro check below puts the two sequences only 15 ms apart.

The sections below follow the order in which the work was done. Everything before [Time offset](#time-offset) was computed with the LiDAR stamps taken as published, so the rotations reported there carry the clock error.

## Data

The data is the `school_scooter1` and `corridors_dolly1` sequences of VECtor, which has two Prophesee event cameras (640 x 480) and an Ouster OS0-128 LiDAR. The files go in `data/raw/vector/`, and the calibration results in `data/raw/vector/calib/`.

## Loading VECtor

`lidar_event_extrinsics.vector` reads the event stream, the LiDAR sweeps, the IMU, the ground truth, and the calibration. Times are integer microseconds on the Unix clock, which all VECtor files share.

Each LiDAR sweep keeps the 128 x 2048 grid the Ouster reports, with a capture time for every point. The sweeps are raw, so one revolution spreads over 100 ms and the points are not motion compensated. The events are compressed with Zstandard, which `h5py` reads once `hdf5plugin` is imported.

VECtor calibrates the LiDAR and the IMU against the left regular camera, so both are chained to the left event camera through the joint camera calibration:

```math
T_{\mathrm{event} \leftarrow \mathrm{lidar}} = T_{\mathrm{event} \leftarrow \mathrm{regular}} \, T_{\mathrm{regular} \leftarrow \mathrm{lidar}}
```

There are two LiDAR extrinsics, one from a FARO laser scan and one from the CAD model. They differ by 0.78° in rotation and 5.97 cm in translation.

## Frame check

The bag stores points in the Ouster `os_sensor` frame. The other Ouster frame, `os_lidar`, is turned 180° about z, and the calibration files only say "LiDAR". `scripts/frame_check.py` decides which one the extrinsics expect. For every 10th sweep it marks depth edges, the near side of any range jump above 0.5 m between neighboring columns, projects them into the event camera, and scores the event density at those pixels relative to the image mean.

| Extrinsic | Median score, `os_sensor` | Median score, turned | Sweeps where `os_sensor` wins |
|---|---|---|---|
| FARO | 1.83 | 1.09 | 31 of 42 |
| CAD | 1.49 | 1.00 | 24 of 43 |

The turned frame scores about 1, which is what edges placed at random would get, so the extrinsics apply to the points as published. The score is noisy on single sweeps, and many sweeps sit near 1 with either frame. The CAD extrinsic scores higher than FARO on the last 4 s of the sequence, which the edge alignment step will have to explain.

## Edge maps

The visible part of a sweep is captured over about 29 ms while the sensor moves, so each edge point is compared with events from 2 ms bins around its own capture time, using the events within 3 ms of the bin center.

A blurred event count only rewards edges that land within a pixel or two of an event, so the score at the FARO extrinsic was no better than at extrinsics rotated by 1° or shifted by 5 cm. The fix follows Levinson and Thrun (RSS 2013). Each LiDAR point is weighted by how much nearer it is than its horizontal neighbors,

```math
w_i = \max(r_{\mathrm{left}} - r_i,\; r_{\mathrm{right}} - r_i)^{1/2},
```

keeping jumps above 0.2 m. The capped event count $`E`$ is spread into an inverse distance map

```math
D(p) = \alpha E(p) + (1 - \alpha) \max_q E(q)\, \gamma^{\lVert p - q \rVert_1},
```

with $`\alpha = 1/3`$ and $`\gamma = 0.9`$, so a point a few pixels off an edge still scores. The score of a sweep is $`\sum_i w_i D(p_i) / \sum_i w_i`$, with each map divided by its mean.

`scripts/compare_edge_maps.py` rotates the FARO extrinsic about each camera axis and shifts it along each axis, and averages the score over every 10th sweep.

| Rotation about x (deg) | -8 | -4 | -2 | -1 | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Blurred count | 1.34 | 1.68 | 2.09 | 2.03 | 2.25 | 1.97 | 1.87 | 2.05 | 2.04 |
| Inverse distance | 1.58 | 1.89 | 2.06 | 2.06 | 2.08 | 2.03 | 1.96 | 1.92 | 1.73 |

| Rotation about y (deg) | -8 | -4 | -2 | -1 | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Blurred count | 2.13 | 2.34 | 2.58 | 3.00 | 2.25 | 2.63 | 2.81 | 2.60 | 1.71 |
| Inverse distance | 1.84 | 1.99 | 2.11 | 2.14 | 2.08 | 2.04 | 2.03 | 1.99 | 1.61 |

With the inverse distance map both curves have one peak, at 0° about x and at -1° about y, and fall off on both sides. The blurred count jumps up and down between neighboring angles. Rotation about the optical axis z is flat between -2° and 5°, and shifts of up to 30 cm change the score by at most 12%. The shift along z peaks at 15 to 20 cm rather than at 0. A few centimeters at 5 to 10 m move a point by only 2 to 3 pixels, so the translation is weakly observable from edges alone. The time surface scored like the blurred count.

## Search

A candidate extrinsic is a correction applied in the event camera frame, a rotation vector and a translation, on top of a starting extrinsic. The score of a candidate is the edge score averaged over every 10th sweep (45 sweeps), since single sweeps are too noisy. The search first tries every rotation on a grid of ±4° in 1° steps around the start (729 candidates), because a local search from a poor start stops on the first bump in the score. From the best grid point a coordinate search nudges each parameter up and down, keeps any move that raises the score, and halves the steps when none does. It refines rotation first, then translation, then all six together.

`scripts/run_search.py` runs the calibration from the FARO and CAD extrinsics on all 45 sweeps, and from FARO on the even and odd halves. Results are corrections to the FARO extrinsic in event camera axes.

| Run | Rotation x, y, z (deg) | Translation x, y, z (cm) | Score start → end |
|---|---|---|---|
| All, from FARO | -1.25, -1.75, +0.63 | 0.0, 0.0, +0.6 | 2.085 → 2.155 |
| All, from CAD | -0.99, -0.56, -0.49 | -0.1, -5.3, +2.6 | 2.172 → 2.214 |
| Even half, from FARO | -1.00, -2.00, +1.00 | 0.0, 0.0, +1.3 | 2.066 → 2.229 |
| Odd half, from FARO | -1.13, -0.38, -2.88 | +3.7, -1.2, +6.3 | 2.104 → 2.265 |

All four runs move the pitch (rotation about x) by -1.0° to -1.25° from FARO. Yaw lands anywhere from -2.0° to -0.4°, roll from -2.9° to +1.0°, and the translation mostly stays wherever it started. The CAD extrinsic differs from FARO by 0.78° and 5.9 cm, which is inside this spread, so the method cannot yet tell the two apart. The CAD translation also scores higher than the FARO one, and the local search from FARO does not reach it, since it needs a move along y and z together.

Adding range jumps between neighboring beams, to get horizontal edges that might constrain roll, made the score flat: the best grid point moved to the edge of the grid. A beam hitting the floor or ceiling at a grazing angle lands far from the beam next to it even on a flat surface, so those jumps mark smooth surfaces as edges.

## Perturb and recover

`scripts/perturb_and_recover.py` starts the calibration from the FARO extrinsic moved by a known amount and records where it lands. With all six parameters free there are 40 trials: rotations of 0.5°, 1°, 2°, and 3°, translations of 2, 5, and 10 cm, and 2° with 5 cm together, five of each with random directions.

None of the 40 trials lands within 0.5° of the run started from FARO itself. Most of them gather somewhere else, at a median rotation of (-0.97°, -0.55°, -1.01°) from FARO. The translation does not come back. It ends 0.6 to 27 cm from where the reference run ends, whatever the start was, and its drift pulls the rotation around with it.

With the translation held at its start (`ROTATION_ONLY=1`), the 20 rotation trials gather much more tightly.

| Axis | Median from FARO | Middle half of trials | Range |
|---|---|---|---|
| x (pitch) | -1.16° | -1.25° to -0.62° | -1.34° to +0.27° |
| y (yaw) | -0.52° | -0.64° to -0.46° | -1.85° to -0.36° |
| z (roll) | -1.20° | -1.61° to -0.77° | -1.98° to +1.23° |

Their final scores have a median of 2.198, above the 2.152 of the run started from FARO, so that run had stopped on a local bump and the cluster is the real peak of the score. Starts up to 3° away find the same peak, and yaw is pinned to within about 0.2° in most trials.

The peak sits about 1.8° from FARO and 1.5° from CAD. At the event camera focal length of 327 px, that moves edges by 3 to 7 px in an overlay, too little to judge by eye which one fits the events better. This test shows that the search finds the same answer from different starts. It cannot show whether that answer is the true extrinsic, because the score could favor a slightly wrong rotation, for example through the weighting of edges that the event camera never sees. The translation has to come from somewhere other than edge alignment in this sequence.

## Second sequence

To check whether the peak on `school_scooter1` is a FARO error or a quirk of the score, the same 20 rotation-only trials ran on `corridors_dolly1` (920 sweeps over 92 s, every 10th used). It is a slower dolly run, at a median of 0.9 m/s against 2.6 m/s on the scooter, with the same rig and the same calibration files.

| Axis | `school_scooter1` | `corridors_dolly1` | CAD relative to FARO |
|---|---|---|---|
| x (pitch) | -1.16° | +0.02° | -0.36° |
| y (yaw) | -0.52° | -0.76° | -0.69° |
| z (roll) | -1.20° | -1.63° | +0.01° |

The table gives the median rotation from FARO over each set of trials. On `corridors_dolly1` the middle half of the trials spans 0.39° in pitch, 0.24° in yaw, and 0.93° in roll, and at 3° starts 3 of 5 trials land within 0.5° of the run started from FARO.

Yaw and roll move the same way on both sequences. Yaw lands at -0.5° to -0.8°, close to the CAD value of -0.69°, which suggests the FARO yaw is off by about that much. Roll lands at -1.2° to -1.6° on both, but it is the loosest axis in every test, so that agreement is weaker evidence. Pitch disagrees by 1.2°, so the -1.2° pitch on `school_scooter1` is not a property of the rig. The faster, bumpier scooter run is a likely cause, since motion that the score does not model would show up there first.

## Time offset

The score compares each LiDAR point with events from around its own capture time, so the 100 ms a sweep takes is already handled and deskewing the sweep with the IMU would not change it. What can still go wrong is the clock. If the LiDAR times are off against the event camera, every point is compared with events from the wrong moment, and the error grows with speed.

`scripts/time_offset.py` checks the clock in three ways.

The ground-truth poses carry the LiDAR timestamps. Their turning rate is compared with the IMU gyro averaged over the same 100 ms, shifted by lags from -300 to +300 ms. This uses neither the extrinsic nor the events.

| Sequence | Best lag | Gyro error at 0 ms | Gyro error at best lag |
|---|---|---|---|
| `school_scooter1` | +135 ms | 6.21 °/s | 2.26 °/s |
| `corridors_dolly1` | +120 ms | 3.48 °/s | 3.24 °/s |

Part of this lag may be a convention of the ground truth, since a pose fitted to a whole sweep can describe its middle or end while carrying the stamp of its start. That would account for 50 to 100 ms at most.

The edge score at the FARO extrinsic, with the LiDAR point times shifted by -100 to +250 ms, peaks at +90 ms on `school_scooter1` (2.371 against 2.066 at 0 ms) and falls off on both sides. On the slower `corridors_dolly1` the curve is noisier, with its highest points at +40 and +120 ms.

Calibrating again with the LiDAR times shifted by +90 ms changes the result. Rotations are from FARO, with the translation held at FARO.

| Run | 0 ms | +90 ms |
|---|---|---|
| `school_scooter1`, all | -1.25°, -1.75°, +0.63° (2.152) | -0.50°, -1.13°, -0.38° (2.855) |
| `school_scooter1`, even half | -1.00°, -2.00°, +1.13° (2.214) | -0.50°, -1.13°, -0.38° (2.871) |
| `school_scooter1`, odd half | -1.00°, -0.38°, -3.00° (2.201) | -0.50°, -1.13°, -0.38° (2.839) |
| `corridors_dolly1`, all | 0.00°, -0.75°, -2.00° (3.157) | +0.50°, -0.88°, -1.00° (3.547) |
| `corridors_dolly1`, even half | -0.25°, -1.00°, -2.75° (3.156) | -0.88°, -1.00°, -1.00° (3.596) |
| `corridors_dolly1`, odd half | 0.00°, -0.75°, +1.13° (3.286) | +0.13°, -0.88°, -1.25° (3.536) |

Scores are in brackets. At +90 ms the two halves of `school_scooter1` land on the same rotation, where at 0 ms they were 3.6° apart in roll, and the score rises by a third. The scooter moves at about 2.6 m/s against 0.9 m/s for the dolly, so the clock error hurt it most, and that explains the sweep-dependent results in the sections above. With the offset, yaw agrees across both sequences at -0.9° to -1.1° from FARO, which suggests the FARO yaw is off by about 1°. Roll lands at -0.4° on the scooter and -1.0° to -1.25° on the dolly, and pitch on the dolly still moves between -0.9° and +0.5° across halves.

VECtor describes its sensors as hardware synchronized, but the LiDAR stamps in both sequences behave as if they run roughly 90 ms, close to one revolution, behind the event camera and the IMU. For this data the clock matters more than any 1° error in the extrinsic.

Splitting the sweeps by speed or by turning rate (`scripts/motion_checks.py`, at 0 ms) did not show a clean trend. The scooter pitch went from -1.00° on the slowest third to -0.25° on the fastest, and yaw and roll jumped by several degrees between thirds of 30 sweeps.

## Joint estimate of offset and rotation

A wrong offset also pulls the rotation, so fixing the offset at 90 ms by hand and then fitting the rotation can carry one error into the other. `scripts/joint_offset.py` calibrates the rotation from scratch at every candidate offset and keeps the pair with the highest score. A coarse pass covers -20 to +200 ms in 20 ms steps, and a fine pass covers ±16 ms around the best in 4 ms steps, on all sweeps and on each half. The translation stays at FARO.

| Sequence | Offset, all / even / odd | Rotation from FARO, all sweeps | Score at 0 ms → best |
|---|---|---|---|
| `school_scooter1` (45 sweeps) | +92 / +92 / +88 ms | -0.50°, -1.13°, -0.38° | 2.152 → 2.838 |
| `corridors_dolly1` (46 sweeps) | +44 / +44 / +44 ms | -0.50°, -1.13°, -1.13° | 3.157 → 3.722 |

Pitch and yaw now come out the same on both sequences, -0.50° and -1.13° from FARO, and the halves of each sequence stay within 0.13° in yaw. Roll still disagrees, at -0.38° on the scooter and -0.38° to -1.13° across the dolly halves, so the rotation about the optical axis is the part edge alignment does not pin down.

The offsets do not match. The scooter run, recorded on 2022-02-20, settles at about 90 ms, and the dolly run, recorded on 2022-01-25, at about 44 ms. The dolly curve also has a second peak at +120 ms (3.614 against 3.663 at +40 ms in the coarse pass). The gyro check gave lags of 135 and 120 ms, 15 ms apart rather than 46, so the size of the offset on the dolly run is less certain than on the scooter. Either way the LiDAR offset does not look like one fixed constant of the rig, which means it has to be estimated per recording.

## Tests

```
pip install -e ".[dev]"
pytest
```

The tests in `tests/test_vector_sequence.py` need the sequence in `data/raw/vector/` (or a folder set with `VECTOR_DIR`) and are skipped without it.

## License

The code is MIT licensed. The repository holds no dataset files. VECtor is released under the BSD 3-Clause Clear License.
