import json
import os
import tempfile
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import pycolmap

from gaussian_splatting.utils.logger import Logger

logger = Logger("COLMAP")


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
class ColmapConfig:
    images_path: str
    output_folder: str
    poses_filename: str
    intrinsics_filename: str
    points_filename: str

    @classmethod
    def from_dict(
        cls,
        configuration: dict,
    ) -> "ColmapConfig":
        if not isinstance(configuration, dict):
            raise ValueError(f"ColmapConfig must be a dictionary, got '{type(configuration).__name__}'.")

        mandatory_fields = {
            "images_path",
            "output_folder",
        }
        if not set(configuration.keys()).issuperset(mandatory_fields):
            missing_fields = mandatory_fields - set(configuration.keys())
            raise ValueError(
                f"ColmapConfig is missing the following mandatory fields: {', '.join(missing_fields)}, "
                f"got {', '.join(configuration.keys())}."
            )

        images_path = configuration["images_path"]
        output_folder = configuration["output_folder"]
        poses_filename = configuration["poses_filename"]
        intrinsics_filename = configuration["intrinsics_filename"]
        points_filename = configuration["points_filename"]

        if not isinstance(images_path, str):
            raise ValueError(f"ColmapConfig 'images_path' must be a string, got '{images_path}'.")
        if not isinstance(output_folder, str):
            raise ValueError(f"ColmapConfig 'output_folder' must be a string, got '{output_folder}'.")
        if not isinstance(poses_filename, str):
            raise ValueError(f"ColmapConfig 'poses_filename' must be a string, got '{poses_filename}'.")
        if not isinstance(intrinsics_filename, str):
            raise ValueError(f"ColmapConfig 'intrinsics_filename' must be a string, got '{intrinsics_filename}'.")
        if not isinstance(points_filename, str):
            raise ValueError(f"ColmapConfig 'points_filename' must be a string, got '{points_filename}'.")

        return ColmapConfig(
            images_path=images_path,
            output_folder=output_folder,
            poses_filename=poses_filename,
            intrinsics_filename=intrinsics_filename,
            points_filename=points_filename,
        )


class ColmapRunner:
    CAMERA_MODEL = "PINHOLE"
    MATCHER = "exhaustive"

    def __init__(
        self,
        configuration: ColmapConfig,
    ) -> None:
        self.configuration = configuration

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

        self.images_path = Path(self.configuration.images_path)
        if (
            not self.images_path.exists()
            or not self.images_path.is_dir()
        ):  # fmt:skip
            raise FileNotFoundError(
                f"COLMAP images_path must point to an existing image directory, got '{self.configuration.images_path}'."
            )

        self.output_folder = Path(self.configuration.output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        logger.info("Running COLMAP [1/4]: Extracting features...")
        self._run_feature_extraction()

        logger.info("Running COLMAP [2/4]: Matching features...")
        self._run_feature_matching()

        logger.info("Running COLMAP [3/4]: Running mapping...")
        self._run_mapping()

        logger.info("Running COLMAP [4/4]: Running reconstruction...")
        self._run_reconstruction()

        self._save_poses_json()

        self._save_intrinsics_json()

        self._save_points_ply()

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

    def _save_poses_json(self) -> None:
        output_path = self.output_folder / self.configuration.poses_filename

        images = self._parse_images(self.text_model_path / "images.txt")

        with open(output_path, "w") as file:
            json.dump(images, file, indent=2)

        logger.info(f"Saved camera poses to {output_path}")

    def _save_intrinsics_json(self) -> None:
        output_path = self.output_folder / self.configuration.intrinsics_filename

        cameras = self._parse_cameras(self.text_model_path / "cameras.txt")

        with open(output_path, "w") as file:
            json.dump(cameras, file, indent=2)

        logger.info(f"Saved camera intrinsics to {output_path}")

    def _parse_cameras(
        self,
        cameras_path: Path,
    ) -> list[dict]:
        cameras = []
        with open(cameras_path) as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]

                camera_data = {
                    "camera_id": camera_id,
                    "model": model,
                    "width": width,
                    "height": height,
                }

                if model == "PINHOLE":
                    camera_data["fx"] = params[0]
                    camera_data["fy"] = params[1]
                    camera_data["cx"] = params[2]
                    camera_data["cy"] = params[3]
                elif model == "SIMPLE_RADIAL":
                    camera_data["f"] = params[0]
                    camera_data["cx"] = params[1]
                    camera_data["cy"] = params[2]
                    camera_data["k"] = params[3]
                else:
                    camera_data["params"] = params

                cameras.append(camera_data)

        return cameras

    def _parse_images(
        self,
        images_path: Path,
    ) -> list[dict]:
        images = []
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
                {
                    "image_id": image_id,
                    "camera_id": camera_id,
                    "name": name,
                    "position": {
                        "x": tx,
                        "y": ty,
                        "z": tz,
                    },
                    "rotation": {
                        "qw": qw,
                        "qx": qx,
                        "qy": qy,
                        "qz": qz,
                    },
                }
            )

        return images

    def _save_points_ply(self) -> None:
        output_path = self.output_folder / self.configuration.points_filename
        points3d_path = self.text_model_path / "points3D.txt"
        if not points3d_path.exists():
            logger.warning(f"No points3D.txt found at {points3d_path}, skipping PLY export")
            return

        points = []
        with open(points3d_path) as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])

                points.append((x, y, z, r, g, b))

        if len(points) == 0:
            logger.warning("No 3D points found in points3D.txt")
            return

        with open(output_path, "w") as file:
            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write(f"element vertex {len(points)}\n")
            file.write("property float x\n")
            file.write("property float y\n")
            file.write("property float z\n")
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")
            file.write("end_header\n")

            for x, y, z, r, g, b in points:
                file.write(f"{x} {y} {z} {r} {g} {b}\n")

        logger.info(f"Saved 3D points to {output_path}")
