import json
import os
import tempfile
from functools import wraps
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np
import pycolmap
import torch
from pydantic import BaseModel

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderer.renderer import Renderer, RendererConfig
from gaussian_splatting.structures.renderer.utils import _quaternions_to_rotation_matrices
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


class ScenePreprocessorConfig(BaseModel):
    images_path: str
    output_folder: str
    poses_filename: str
    intrinsics_filename: str
    points_filename: str
    example_image_filename: str | None = None


class ScenePreprocessor:
    CAMERA_MODEL = "PINHOLE"
    MATCHER = "exhaustive"

    def __init__(
        self,
        configuration: ScenePreprocessorConfig,
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

        self._render_example_image(gaussian_collection)

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
                if (
                    not line
                    or line.startswith("#")
                ):  # fmt:skip
                    continue

                parts = line.split()
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                colors_uint8.append([int(parts[4]), int(parts[5]), int(parts[6])])

        if len(positions) == 0:
            raise ValueError("No 3D points found in points3D.txt")

        positions = torch.tensor(positions, dtype=torch.float32)
        colors_rgb = torch.tensor(colors_uint8, dtype=torch.float32) / 255.0

        n_points = positions.shape[0]

        SH_DEGREE = 3  # TODO: Make this configurable
        NUM_SH_COEFFS = (SH_DEGREE + 1) ** 2
        C0 = 1 / (2 * np.sqrt(np.pi))
        sh_dc = (colors_rgb - 0.5) / C0
        sh_rest = torch.zeros((n_points, NUM_SH_COEFFS - 1, 3), dtype=torch.float32)
        sh_coeffs = torch.cat([sh_dc.unsqueeze(1), sh_rest], dim=1)

        # TODO: Explain this
        positions_np = positions.numpy()
        K = 3
        diff = positions_np[:, None, :] - positions_np[None, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        knn_dist = np.sqrt(np.sort(dist_sq, axis=1)[:, :K])
        avg_dist = np.mean(knn_dist, axis=1).clip(min=1e-7)
        log_scales = np.log(avg_dist * 0.5).astype(np.float32)
        scales = torch.tensor(log_scales, dtype=torch.float32).unsqueeze(1).repeat(1, 3)

        quaternions = torch.zeros((n_points, 4), dtype=torch.float32)
        quaternions[:, 0] = 1.0

        opacities = torch.full(
            (n_points, 1),
            fill_value=1,
            dtype=torch.float32,
        )

        return GaussianCollection.from_tensors(
            positions=mx.array(positions.numpy()),
            quaternions=mx.array(quaternions.numpy()),
            scales=mx.array(scales.numpy()),
            sh_coeffs=mx.array(sh_coeffs.numpy()),
            opacities=mx.array(opacities.numpy()),
        )

    def _render_example_image(
        self,
        gaussian_collection: GaussianCollection,
    ) -> None:
        if self.configuration.example_image_filename is None:
            return

        intrinsics_path = self.output_folder / self.configuration.intrinsics_filename
        poses_path = self.output_folder / self.configuration.poses_filename

        if (
            not intrinsics_path.exists()
            or not poses_path.exists()
        ):  # fmt:skip
            logger.warning("Intrinsics or poses file not found. Skipping example rendering.")
            return

        with open(intrinsics_path, "r") as f:
            intrinsics = json.load(f)

        with open(poses_path, "r") as f:
            poses = json.load(f)

        example_pose = poses[0]
        camera_id = example_pose["camera_id"]

        camera_meta = next(
            (
                c
                for c in intrinsics
                if c["camera_id"] == camera_id
            ),
            intrinsics[0]
        )  # fmt:skip

        height = camera_meta["height"]
        width = camera_meta["width"]

        if "fx" in camera_meta and "fy" in camera_meta:
            focal_length = (camera_meta["fx"] + camera_meta["fy"]) / 2.0
        else:
            focal_length = camera_meta.get("fx", camera_meta.get("f", 1.0))

        tx = example_pose["position"]["x"]
        ty = example_pose["position"]["y"]
        tz = example_pose["position"]["z"]

        qw = example_pose["rotation"]["qw"]
        qx = example_pose["rotation"]["qx"]
        qy = example_pose["rotation"]["qy"]
        qz = example_pose["rotation"]["qz"]

        q_w2c = mx.array([[qw, qx, qy, qz]], dtype=mx.float32)
        t_w2c = mx.array([tx, ty, tz], dtype=mx.float32)

        r_w2c = _quaternions_to_rotation_matrices(q_w2c)[0]

        r_c2w = r_w2c.T
        t_c2w = -r_c2w @ t_w2c

        top_rows = mx.concatenate([r_c2w, mx.expand_dims(t_c2w, axis=1)], axis=1)
        bottom_row = mx.array([[0.0, 0.0, 0.0, 1.0]], dtype=mx.float32)
        pose_matrix = mx.concatenate([top_rows, bottom_row], axis=0)

        camera = Camera(
            pose=pose_matrix,
            focal_length=focal_length,
            width=width,
            height=height,
        )

        renderer = Renderer(
            RendererConfig(
                width=width,
                height=height,
                focal_length=focal_length,
            )
        )

        try:
            rendered_image = renderer.render(
                camera=camera,
                gaussians=gaussian_collection,
            )

            image_np = (rendered_image.array * 255.0).clip(0, 255).astype(np.uint8)

            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            output_image_path = self.output_folder / self.configuration.example_image_filename
            cv2.imwrite(str(output_image_path), image_bgr)

            logger.info(f"Saved rendered example image to {output_image_path}")
        except Exception as e:
            logger.error(f"Failed to render example image: {e}")
