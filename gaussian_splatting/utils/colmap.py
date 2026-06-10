import json
import os
import tempfile
from functools import wraps
from pathlib import Path

import numpy as np
import pycolmap
import torch
from pydantic import BaseModel

from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply.ply_saver import PLYSaver

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


class ColmapConfig(BaseModel):
    images_path: str
    output_folder: str
    poses_filename: str
    intrinsics_filename: str
    points_filename: str


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

        gaussian_collection = self._create_gaussian_collection()

        ply_output_path = self.output_folder / self.configuration.points_filename
        ply_saver = PLYSaver(ply_output_path)
        ply_saver.save_gaussians(gaussian_collection)

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

    def _create_gaussian_collection(self) -> GaussianCollection:
        points3d_path = self.text_model_path / "points3D.txt"
        if not points3d_path.exists():
            raise FileNotFoundError(f"No points3D.txt found at {points3d_path}")

        positions = []
        colors_uint8 = []
        with open(points3d_path) as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                colors_uint8.append([int(parts[4]), int(parts[5]), int(parts[6])])

        if len(positions) == 0:
            raise ValueError("No 3D points found in points3D.txt")

        positions = torch.tensor(positions, dtype=torch.float32)
        colors_rgb = torch.tensor(colors_uint8, dtype=torch.float32) / 255.0

        C0 = 1 / (2 * np.sqrt(np.pi))
        sh_dc = (colors_rgb - 0.5) / C0
        sh_coeffs = sh_dc.unsqueeze(1)

        n_points = positions.shape[0]
        scene_extent = torch.norm(positions.max(dim=0).values - positions.min(dim=0).values)
        base_sigma = torch.clamp(scene_extent / np.sqrt(float(n_points)), min=1e-4)
        scales = base_sigma.expand(n_points, 1).repeat(1, 3)

        quaternions = torch.zeros((n_points, 4), dtype=torch.float32)
        quaternions[:, 0] = 1.0

        opacities = torch.full(
            (n_points, 1),
            fill_value=1,
            dtype=torch.float32,
        )

        return GaussianCollection.from_tensors(
            positions=positions,
            quaternions=quaternions,
            scales=scales,
            sh_coeffs=sh_coeffs,
            opacities=opacities,
        )
