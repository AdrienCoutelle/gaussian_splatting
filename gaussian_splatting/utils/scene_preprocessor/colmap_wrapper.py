import os
import tempfile
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import pycolmap

from gaussian_splatting.utils.logger import Logger

logger = Logger("COLMAP_WRAPPER")


def suppress_output_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with open(os.devnull, "w") as devnull:
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            try:
                return func(*args, **kwargs)
            finally:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)

    return wrapper


@dataclass
class ColmapPose:
    image_id: int
    camera_id: int
    name: str
    position: dict
    rotation: dict


@dataclass
class ColmapIntrinsic:
    camera_id: int
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class ColmapPoint:
    point_id: int
    xyz: list[float]
    rgb: list[int]
    error: float
    track_length: int


@dataclass
class ColmapResults:
    intrinsics: ColmapIntrinsic
    poses: list[ColmapPose]
    points: list[ColmapPoint]


class ColmapWrapper:
    CAMERA_MODEL = "PINHOLE"
    MATCHER = "exhaustive"

    def __init__(
        self,
        images_path: str,
    ) -> None:
        self.device = pycolmap.Device.auto
        self.temp_dir = tempfile.TemporaryDirectory()

        self.workspace_path = Path(self.temp_dir.name)
        self.workspace_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.sparse_path = self.workspace_path / "sparse"
        self.sparse_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.database_path = self.workspace_path / "database.db"

        self.text_model_path = self.workspace_path / "text"
        self.text_model_path.mkdir(parents=True, exist_ok=True)

        self.images_path = Path(images_path)
        if (
            not self.images_path.exists()
            or not self.images_path.is_dir()
        ):  # fmt:skip
            raise FileNotFoundError(
                f"COLMAP images_path must point to an existing image directory, got '{images_path}'."
            )

    def run(self) -> ColmapResults:
        logger.info("Running COLMAP [1/4]: Extracting features...")
        self._run_feature_extraction()

        logger.info("Running COLMAP [2/4]: Matching features...")
        self._run_feature_matching()

        logger.info("Running COLMAP [3/4]: Running mapping...")
        self._run_mapping()

        logger.info("Running COLMAP [4/4]: Running reconstruction...")
        self._run_reconstruction()

        return ColmapResults(
            poses=self._parse_images(self.text_model_path / "images.txt"),
            intrinsics=self._parse_cameras(self.text_model_path / "cameras.txt"),
            points=self._parse_points(self.text_model_path / "points3D.txt"),
        )

    @suppress_output_wrapper
    def _run_feature_extraction(self) -> None:
        reader_options = pycolmap.ImageReaderOptions(camera_model=self.CAMERA_MODEL)
        pycolmap.extract_features(
            database_path=self.database_path,
            image_path=self.images_path,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
            device=self.device,
        )

    @suppress_output_wrapper
    def _run_feature_matching(self) -> None:
        if self.MATCHER == "exhaustive":
            pycolmap.match_exhaustive(
                database_path=self.database_path,
                device=self.device,
            )
        elif self.MATCHER == "sequential":
            pycolmap.match_sequential(
                database_path=self.database_path,
                device=self.device,
            )
        else:
            raise ValueError(f"Unsupported matcher: {self.MATCHER}")

    @suppress_output_wrapper
    def _run_mapping(self) -> None:
        maps = pycolmap.incremental_mapping(
            database_path=self.database_path,
            image_path=self.images_path,
            output_path=self.sparse_path,
        )

        if len(maps) == 0:
            raise RuntimeError("COLMAP mapper did not produce any reconstruction.")

    @suppress_output_wrapper
    def _run_reconstruction(self) -> None:
        reconstruction = pycolmap.Reconstruction(self.sparse_path / "0")
        reconstruction.write_text(self.text_model_path)

    def _parse_cameras(
        self,
        cameras_path: Path,
    ) -> list[ColmapIntrinsic]:
        cameras: list[ColmapIntrinsic] = []
        with open(cameras_path) as file:
            for line in file:
                line = line.strip()
                if (
                    not line
                    or line.startswith("#")
                ):  # fmt:skip
                    continue

                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]

                if model != self.CAMERA_MODEL:
                    raise RuntimeError(f"Only '{self.CAMERA_MODEL}' is supported yet, got '{model}'.")

                cameras.append(
                    ColmapIntrinsic(
                        camera_id=camera_id,
                        model=model,
                        width=width,
                        height=height,
                        fx=params[0],
                        fy=params[1],
                        cx=params[2],
                        cy=params[3],
                    )
                )

        if len(cameras) != 1:
            raise RuntimeError("COLMAP results got more than 1 camera, not supported yet.")

        return cameras[0]

    def _parse_images(
        self,
        images_path: Path,
    ) -> list[ColmapPose]:
        images: list[ColmapPose] = []
        with open(images_path) as file:
            lines = [
                line.strip()
                for line in file
                if (
                    line.strip()
                    and not line.startswith("#")
                )
            ]  # fmt:skip

        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            image_id = int(parts[0])
            qw, qx, qy, qz = [float(parts[j]) for j in range(1, 5)]
            tx, ty, tz = [float(parts[j]) for j in range(5, 8)]
            camera_id = int(parts[8])
            name = parts[9]

            images.append(
                ColmapPose(
                    image_id=image_id,
                    camera_id=camera_id,
                    name=name,
                    position={
                        "x": tx,
                        "y": ty,
                        "z": tz,
                    },
                    rotation={
                        "qw": qw,
                        "qx": qx,
                        "qy": qy,
                        "qz": qz,
                    },
                )
            )

        return images

    def _parse_points(
        self,
        points_path: Path,
    ) -> list[ColmapPoint]:
        points: list[ColmapPoint] = []
        with open(points_path) as file:
            for line in file:
                line = line.strip()
                if (
                    not line
                    or line.startswith("#")
                ):  # fmt:skip
                    continue

                parts = line.split()
                points.append(
                    ColmapPoint(
                        point_id=int(parts[0]),
                        xyz=[float(parts[1]), float(parts[2]), float(parts[3])],
                        rgb=[int(parts[4]), int(parts[5]), int(parts[6])],
                        error=float(parts[7]),
                        track_length=(len(parts) - 8) // 2,
                    )
                )

        return points
