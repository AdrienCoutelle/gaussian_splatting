import json
from dataclasses import asdict
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np
from pydantic import BaseModel

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderer.renderer import Renderer, RendererConfig
from gaussian_splatting.structures.renderer.utils import _quaternions_to_rotation_matrices
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply.ply_saver import PLYSaver
from gaussian_splatting.utils.scene_preprocessor.colmap_postprocessor import ColmapPostprocessor
from gaussian_splatting.utils.scene_preprocessor.colmap_wrapper import (
    ColmapIntrinsic,
    ColmapPose,
    ColmapResults,
    ColmapWrapper,
)

logger = Logger("SCENE_PREPROCESSOR")


class ScenePreprocessorConfig(BaseModel):
    images_path: str
    output_folder: str
    poses_filename: str
    intrinsics_filename: str
    points_filename: str
    example_image_filename: str | None = None


class ScenePreprocessor:
    def __init__(
        self,
        configuration: ScenePreprocessorConfig,
    ) -> None:
        self.configuration = configuration
        self.colmap_wrapper = ColmapWrapper(self.configuration.images_path)
        self.colmap_postprocessor = ColmapPostprocessor()

        self.output_folder = Path(self.configuration.output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        colmap_results = self.colmap_wrapper.run()
        colmap_results = self.colmap_postprocessor.run(colmap_results)
        self._save_poses_json(colmap_results.poses)
        self._save_intrinsics_json(colmap_results.intrinsics)

        gaussian_collection = self._create_gaussian_collection(colmap_results)

        ply_output_path = self.output_folder / self.configuration.points_filename
        ply_saver = PLYSaver(ply_output_path)
        ply_saver.save_gaussians(gaussian_collection)

        self._render_example_image(
            gaussian_collection=gaussian_collection,
            colmap_results=colmap_results,
        )

    def _save_poses_json(
        self,
        poses: list[ColmapPose],
    ) -> None:
        output_path = self.output_folder / self.configuration.poses_filename

        with open(output_path, "w") as file:
            json.dump([asdict(pose) for pose in poses], file, indent=2)

        logger.info(f"Saved camera poses to {output_path}")

    def _save_intrinsics_json(
        self,
        intrinsics: ColmapIntrinsic,
    ) -> None:
        output_path = self.output_folder / self.configuration.intrinsics_filename

        with open(output_path, "w") as file:
            json.dump(
                [asdict(intrinsics)],
                file,
                indent=2,
            )

        logger.info(f"Saved camera intrinsics to {output_path}")

    def _create_gaussian_collection(
        self,
        colmap_results: ColmapResults,
    ) -> GaussianCollection:
        if len(colmap_results.points) == 0:
            raise ValueError("COLMAP reconstruction did not produce any 3D points.")

        positions = mx.array([point.xyz for point in colmap_results.points], dtype=mx.float32)
        colors_rgb = mx.array([point.rgb for point in colmap_results.points], dtype=mx.float32) / 255.0

        n_points = positions.shape[0]

        SH_DEGREE = 3  # TODO: Make this configurable
        NUM_SH_COEFFS = (SH_DEGREE + 1) ** 2
        C0 = 1 / (2 * np.sqrt(np.pi))
        sh_dc = (colors_rgb - 0.5) / C0
        sh_rest = mx.zeros((n_points, NUM_SH_COEFFS - 1, 3), dtype=mx.float32)
        sh_coeffs = mx.concatenate([mx.expand_dims(sh_dc, axis=1), sh_rest], axis=1)

        num_nearest_neighbors = 3
        pairwise_offsets = positions[:, None, :] - positions[None, :, :]
        squared_distances = mx.sum(pairwise_offsets**2, axis=-1)
        squared_distances = mx.where(
            mx.eye(n_points, dtype=mx.bool_),
            float("inf"),
            squared_distances,
        )
        neighbor_distances = mx.sqrt(mx.sort(squared_distances, axis=1)[:, :num_nearest_neighbors])
        average_neighbor_distances = mx.mean(neighbor_distances, axis=1)
        log_scales = mx.log(mx.maximum(average_neighbor_distances * 0.5, 1e-7))
        scales = mx.stack([log_scales, log_scales, log_scales], axis=1)

        quaternions = mx.concatenate(
            [
                mx.ones((n_points, 1), dtype=mx.float32),
                mx.zeros((n_points, 3), dtype=mx.float32),
            ],
            axis=1,
        )
        opacities = mx.ones((n_points, 1), dtype=mx.float32)

        return GaussianCollection.from_tensors(
            positions=positions,
            quaternions=quaternions,
            scales=scales,
            sh_coeffs=sh_coeffs,
            opacities=opacities,
        )

    def _render_example_image(
        self,
        gaussian_collection: GaussianCollection,
        colmap_results: ColmapResults,
    ) -> None:
        if self.configuration.example_image_filename is None:
            return

        if len(colmap_results.poses) == 0:
            logger.warning("COLMAP did not produce poses. Skipping example rendering.")
            return

        example_pose = colmap_results.poses[0]
        camera_meta = colmap_results.intrinsics

        height = camera_meta.height
        width = camera_meta.width

        if camera_meta.fx is not None and camera_meta.fy is not None:
            focal_length = (camera_meta.fx + camera_meta.fy) / 2.0
        else:
            focal_length = camera_meta.fx or camera_meta.f or 1.0

        tx = example_pose.position["x"]
        ty = example_pose.position["y"]
        tz = example_pose.position["z"]

        qw = example_pose.rotation["qw"]
        qx = example_pose.rotation["qx"]
        qy = example_pose.rotation["qy"]
        qz = example_pose.rotation["qz"]

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
                draw_axis=True,
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
        except Exception as exception:
            logger.error(f"Failed to render example image: {exception}")
