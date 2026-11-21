# LiDAR to event camera extrinsics

This is a work in progress. It corrects the extrinsic between a LiDAR and an event camera without a calibration target, by aligning depth edges in the LiDAR scan with edges in the event stream.

The extrinsic is the rigid transform $`T_{\mathrm{camera} \leftarrow \mathrm{lidar}}`$ that moves a LiDAR point into the camera frame. A small error in it shifts projected depth away from the edges the event camera sees. The method searches over a 6-DoF correction for the one that best lines the two up, and it is tested by perturbing a known extrinsic and measuring how much of the perturbation is recovered.

## Data

The data is the `school_scooter1` sequence of [VECtor](https://star-datasets.github.io/vector/), which has two Prophesee event cameras (640 x 480) and an Ouster OS0-128 LiDAR. The files go in `data/raw/vector/`, and the calibration results in `data/raw/vector/calib/`.

## License

The code is MIT licensed. The repository holds no dataset files. VECtor is released under the BSD 3-Clause Clear License.
