"""Loaders for one large-scale VECtor sequence (https://star-datasets.github.io/vector/).

Files used, with VECtor's own names:
    <sequence>.synced.left_event.hdf5    left event camera, DSEC layout
    <sequence>.synced.lidar.bag          Ouster OS0-128 sweeps, ROS1 bag
    <sequence>.synced.imu.txt            IMU at 200 Hz
    <sequence>.synced.gt.txt             LiDAR trajectory, TUM format
    calib/*.yaml                         calibration results

All times are integer microseconds on the Unix clock, which every file shares.
Positions are in meters.

Frames:
    lidar    Ouster os_sensor frame, x forward, z up
    regular  left regular (frame-based) camera, which VECtor calibrates everything to
    event    left event camera, z forward, x right, y down
    imu      the IMU
    world    the frame of the ground-truth trajectory
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from lidar_event_extrinsics.camera import Camera
from lidar_event_extrinsics.geometry import (
    check_rigid,
    invert,
    make_transform,
    rotation_from_quaternion,
)

SEQUENCE = "school_scooter1"


def sequence_files(data_dir, sequence=SEQUENCE):
    data_dir = Path(data_dir)
    return {
        "events": data_dir / f"{sequence}.synced.left_event.hdf5",
        "lidar": data_dir / f"{sequence}.synced.lidar.bag",
        "imu": data_dir / f"{sequence}.synced.imu.txt",
        "ground_truth": data_dir / f"{sequence}.synced.gt.txt",
        "calib": data_dir / "calib",
    }


def seconds_to_us(seconds):
    return np.round(np.asarray(seconds, dtype=np.float64) * 1e6).astype(np.int64)


# Calibration


def read_yaml(path):
    with open(path) as file:
        return yaml.safe_load(file)


def load_event_camera(calib_dir):
    calib = read_yaml(Path(calib_dir) / "left_event_camera_intrinsic_results.yaml")
    if calib["camera_model"] != "plumb_bob":
        raise ValueError(f"expected a plumb_bob camera, got {calib['camera_model']}")

    camera_matrix = np.array(calib["camera_matrix"]["data"]).reshape(3, 3)
    distortion = np.array(calib["distortion_coefficients"]["data"], dtype=np.float64)
    return Camera(
        fx=camera_matrix[0, 0],
        fy=camera_matrix[1, 1],
        cx=camera_matrix[0, 2],
        cy=camera_matrix[1, 2],
        width=calib["image_width"],
        height=calib["image_height"],
        distortion=distortion,
    )


def load_extrinsics(calib_dir):
    """Read the VECtor extrinsics and chain them to the left event camera.

    VECtor measures the LiDAR and the IMU against the left regular camera, not the
    event camera, so each one is chained through the joint camera calibration:

        T_event_from_lidar = T_event_from_regular @ T_regular_from_lidar

    There are two LiDAR extrinsics. "faro" comes from registering LiDAR scans and
    checkerboard images to a FARO laser scan of the room. "cad" comes from the
    mechanical drawing of the rig.
    """
    calib_dir = Path(calib_dir)

    joint = read_yaml(calib_dir / "large_scale_joint_camera_extrinsic_results.yaml")
    T_regular_from_event = np.array(joint["cam2"]["T_cam0_cam2"]).reshape(4, 4)
    T_event_from_regular = invert(check_rigid(T_regular_from_event, "T_regular_from_event"))

    lidar_faro = read_yaml(calib_dir / "camera_lidar_extrinsic_results.yaml")
    T_regular_from_lidar_faro = np.array(lidar_faro["cam0"]["T_cam_lidar"]).reshape(4, 4)
    check_rigid(T_regular_from_lidar_faro, "T_regular_from_lidar (faro)")

    lidar_cad = read_yaml(calib_dir / "camera_lidar_extrinsic_cad_readings.yaml")
    T_regular_from_lidar_cad = np.array(lidar_cad["cam0"]["T_cam_lidar"]).reshape(4, 4)
    check_rigid(T_regular_from_lidar_cad, "T_regular_from_lidar (cad)")

    camera_imu = read_yaml(calib_dir / "large_scale_camera_imu_extrinsic_results1.yaml")
    T_regular_from_imu = np.array(camera_imu["cam0"]["T_cam_imu"])
    check_rigid(T_regular_from_imu, "T_regular_from_imu")

    T_event_from_lidar_faro = T_event_from_regular @ T_regular_from_lidar_faro
    T_event_from_lidar_cad = T_event_from_regular @ T_regular_from_lidar_cad
    T_event_from_imu = T_event_from_regular @ T_regular_from_imu
    T_lidar_from_imu = invert(T_regular_from_lidar_faro) @ T_regular_from_imu

    return {
        "T_event_from_lidar_faro": T_event_from_lidar_faro,
        "T_event_from_lidar_cad": T_event_from_lidar_cad,
        "T_event_from_imu": T_event_from_imu,
        "T_lidar_from_imu": T_lidar_from_imu,
    }


# Events


@dataclass
class Events:
    x: np.ndarray  # (N,) int pixel column
    y: np.ndarray  # (N,) int pixel row
    t_us: np.ndarray  # (N,) int64, Unix time
    polarity: np.ndarray  # (N,) int8, +1 brighter or -1 darker

    def __len__(self):
        return len(self.t_us)

    def slice(self, first, last):
        return Events(
            x=self.x[first:last],
            y=self.y[first:last],
            t_us=self.t_us[first:last],
            polarity=self.polarity[first:last],
        )


def open_event_file(event_path):
    # VECtor compresses events with Zstandard. Importing hdf5plugin registers that
    # filter with h5py, otherwise every read fails.
    import h5py
    import hdf5plugin  # noqa: F401

    return h5py.File(event_path, "r")


def event_time_range_us(event_path):
    with open_event_file(event_path) as h5:
        t_offset_us = int(h5["t_offset"][0])
        first_us = t_offset_us + int(h5["events/t"][0])
        last_us = t_offset_us + int(h5["events/t"][-1])
    return first_us, last_us


def load_events(event_path, t_start_us, t_end_us):
    """Read the events with t_start_us <= t < t_end_us.

    A sequence holds over 100 million events, so only the window is read. The file
    stores t relative to t_offset, and ms_to_idx[k] is the index of the first event
    at or after k milliseconds, which gives the slice without scanning t.
    """
    with open_event_file(event_path) as h5:
        t_offset_us = int(h5["t_offset"][0])
        ms_to_idx = h5["ms_to_idx"]
        n_events = h5["events/t"].shape[0]

        start_ms = max((t_start_us - t_offset_us) // 1000, 0)
        end_ms = (t_end_us - t_offset_us) // 1000 + 1
        start = int(ms_to_idx[start_ms]) if start_ms < len(ms_to_idx) else n_events
        end = int(ms_to_idx[end_ms]) if end_ms < len(ms_to_idx) else n_events

        t_us = h5["events/t"][start:end].astype(np.int64) + t_offset_us
        x = h5["events/x"][start:end]
        y = h5["events/y"][start:end]
        polarity_01 = h5["events/p"][start:end]

    # The millisecond slice is a little wider than the window, so trim it exactly.
    keep = (t_us >= t_start_us) & (t_us < t_end_us)
    polarity = np.where(polarity_01[keep] > 0, 1, -1).astype(np.int8)
    return Events(
        x=x[keep].astype(np.int32),
        y=y[keep].astype(np.int32),
        t_us=t_us[keep],
        polarity=polarity,
    )


# LiDAR


@dataclass
class LidarSweep:
    """One full revolution of the Ouster, as the 128 x 2048 grid the sensor reports.

    Row k is beam k from top to bottom and columns follow the firing order, so
    neighbors in the grid are neighbors in space. Empty returns have xyz = 0.
    """

    stamp_us: int  # time of the first column
    xyz: np.ndarray  # (128, 2048, 3) float64, lidar frame
    point_time_us: np.ndarray  # (128, 2048) int64, Unix time of each point
    valid: np.ndarray  # (128, 2048) bool, True where there was a return


# PointCloud2 datatype codes to NumPy types.
POINTFIELD_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def decode_point_cloud(message):
    """Turn a sensor_msgs/PointCloud2 into a structured array of shape (height, width)."""
    names = [field.name for field in message.fields]
    formats = [POINTFIELD_TYPES[field.datatype] for field in message.fields]
    offsets = [field.offset for field in message.fields]
    point_type = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": message.point_step}
    )
    points = np.frombuffer(bytes(message.data), dtype=point_type)
    return points.reshape(message.height, message.width)


def load_lidar_sweep(bag_path, stamp_us):
    """Read the sweep whose header stamp is stamp_us.

    The stamps of all sweeps are the ground-truth times (one pose per sweep), so a
    caller picks a sweep from GroundTruth.t_us. The bag is bz2 compressed and a full
    pass takes about 90 s, so only the chunk near the sweep is read. In these bags the
    record time equals the header stamp, and the ground truth rounds it to whole
    microseconds, so a 1 ms window around the stamp finds exactly one sweep.
    """
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    search_start_ns = (int(stamp_us) - 1000) * 1000
    search_end_ns = (int(stamp_us) + 1000) * 1000

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        for connection, _, raw in reader.messages(start=search_start_ns, stop=search_end_ns):
            message = reader.deserialize(raw, connection.msgtype)
            header_us = message.header.stamp.sec * 1_000_000 + message.header.stamp.nanosec // 1000
            if abs(header_us - stamp_us) > 1000:
                continue

            points = decode_point_cloud(message)
            xyz = np.stack([points["x"], points["y"], points["z"]], axis=-1).astype(np.float64)

            # The field t counts nanoseconds from the first column of this sweep.
            offset_us = points["t"].astype(np.int64) // 1000
            point_time_us = header_us + offset_us

            valid = np.linalg.norm(xyz, axis=-1) > 0
            return LidarSweep(stamp_us=header_us, xyz=xyz, point_time_us=point_time_us, valid=valid)

    raise KeyError(f"no LiDAR sweep with stamp {stamp_us} us in {bag_path}")


# IMU and ground truth


@dataclass
class Imu:
    t_us: np.ndarray  # (N,) int64
    gyro: np.ndarray  # (N, 3) rad/s, imu frame
    accel: np.ndarray  # (N, 3) m/s^2, imu frame


def load_imu(imu_path):
    # Columns: timestamp gx gy gz ax ay az qx qy qz qw. The orientation is not used.
    table = np.loadtxt(imu_path, comments="#")
    return Imu(t_us=seconds_to_us(table[:, 0]), gyro=table[:, 1:4], accel=table[:, 4:7])


@dataclass
class GroundTruth:
    t_us: np.ndarray  # (N,) int64, one per LiDAR sweep
    T_world_from_lidar: np.ndarray  # (N, 4, 4)

    def __len__(self):
        return len(self.t_us)

    def positions(self):
        return self.T_world_from_lidar[:, :3, 3]


def load_ground_truth(ground_truth_path):
    """Read the TUM file: timestamp tx ty tz qx qy qz qw.

    A TUM row is the pose of the sensor in the world, so it is T_world_from_lidar.
    """
    table = np.loadtxt(ground_truth_path, comments="#")
    poses = np.zeros((len(table), 4, 4))
    for row_index, row in enumerate(table):
        rotation = rotation_from_quaternion(row[4], row[5], row[6], row[7])
        poses[row_index] = make_transform(rotation, row[1:4])
    return GroundTruth(t_us=seconds_to_us(table[:, 0]), T_world_from_lidar=poses)
