# LiDAR to event camera extrinsics

This is a work in progress. It corrects the extrinsic between a LiDAR and an event camera without a calibration target, by aligning depth edges in the LiDAR scan with edges in the event stream.

The extrinsic is the rigid transform $`T_{\mathrm{camera} \leftarrow \mathrm{lidar}}`$ that moves a LiDAR point into the camera frame. A small error in it shifts projected depth away from the edges the event camera sees. The method searches over a 6-DoF correction for the one that best lines the two up, and it is tested by perturbing a known extrinsic and measuring how much of the perturbation is recovered.

## Data

The data is the `school_scooter1` sequence of [VECtor](https://star-datasets.github.io/vector/), which has two Prophesee event cameras (640 x 480) and an Ouster OS0-128 LiDAR. The files go in `data/raw/vector/`, and the calibration results in `data/raw/vector/calib/`.

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

## Tests

```
pip install -e ".[dev]"
pytest
```

The tests in `tests/test_vector_sequence.py` need the sequence in `data/raw/vector/` (or a folder set with `VECTOR_DIR`) and are skipped without it.

## License

The code is MIT licensed. The repository holds no dataset files. VECtor is released under the BSD 3-Clause Clear License.
