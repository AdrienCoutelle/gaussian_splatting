import json
from dataclasses import asdict
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np
import torch
from pydantic import BaseModel

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderer.renderer import Renderer, RendererConfig
from gaussian_splatting.structures.renderer.utils import _quaternions_to_rotation_matrices
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply.ply_saver import PLYSaver
from gaussian_splatting.utils.scene_preprocessor.colmap_wrapper import (
    ColmapIntrinsic,
    ColmapPoint,
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

        self.output_folder = Path(self.configuration.output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        colmap_results = self.colmap_wrapper.run()
        colmap_results = self._orient_and_center_scene(colmap_results)
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

    def _orient_and_center_scene(
        self,
        colmap_results: ColmapResults,
    ) -> ColmapResults:
        """Rotate the reconstruction to Z-up and center its sparse points at the origin.

        COLMAP camera poses are stored as world-to-camera transforms. Apply the same world rotation
        and translation to camera centers and reconstructed points so the saved poses and PLY remain
        in the same coordinate system.
        """
        if not colmap_results.poses:
            raise ValueError("COLMAP reconstruction did not produce any camera poses.")
        if not colmap_results.points:
            raise ValueError("COLMAP reconstruction did not produce any 3D points.")

        camera_to_world_rotations = np.stack(
            [self._quaternion_to_rotation_matrix(pose.rotation).T for pose in colmap_results.poses]
        )
        average_camera_up = -camera_to_world_rotations[:, :, 1].mean(axis=0)
        if np.linalg.norm(average_camera_up) < 1e-6:
            raise ValueError("Could not determine an average camera up direction from the COLMAP poses.")

        world_rotation = self._rotation_between_vectors(
            source=average_camera_up,
            target=np.array([0.0, 0.0, 1.0]),
        )
        rotated_point_positions = np.stack([world_rotation @ np.asarray(point.xyz) for point in colmap_results.points])
        world_translation = -rotated_point_positions.mean(axis=0)

        processed_poses = [
            self._rotate_pose(
                pose=pose,
                world_rotation=world_rotation,
                world_translation=world_translation,
            )
            for pose in colmap_results.poses
        ]
        processed_points = [
            ColmapPoint(
                point_id=point.point_id,
                xyz=(rotated_position + world_translation).tolist(),
                rgb=point.rgb,
                error=point.error,
                track_length=point.track_length,
            )
            for point, rotated_position in zip(
                colmap_results.points,
                rotated_point_positions,
                strict=True,
            )
        ]

        logger.info("Rotated and centered the COLMAP reconstruction in a Z-up coordinate system.")
        return ColmapResults(
            poses=processed_poses,
            intrinsics=colmap_results.intrinsics,
            points=processed_points,
        )

    @staticmethod
    def _rotate_pose(
        pose: ColmapPose,
        world_rotation: np.ndarray,
        world_translation: np.ndarray,
    ) -> ColmapPose:
        rotation_world_to_camera = ScenePreprocessor._quaternion_to_rotation_matrix(pose.rotation)
        translation_world_to_camera = np.array(
            [
                pose.position["x"],
                pose.position["y"],
                pose.position["z"],
            ]
        )

        camera_center_world = -rotation_world_to_camera.T @ translation_world_to_camera
        processed_rotation_world_to_camera = rotation_world_to_camera @ world_rotation.T
        processed_camera_center_world = world_rotation @ camera_center_world + world_translation
        processed_translation_world_to_camera = -processed_rotation_world_to_camera @ processed_camera_center_world

        quaternion = ScenePreprocessor._rotation_matrix_to_quaternion(processed_rotation_world_to_camera)
        return ColmapPose(
            image_id=pose.image_id,
            camera_id=pose.camera_id,
            name=pose.name,
            position={
                "x": float(processed_translation_world_to_camera[0]),
                "y": float(processed_translation_world_to_camera[1]),
                "z": float(processed_translation_world_to_camera[2]),
            },
            rotation={
                "qw": float(quaternion[0]),
                "qx": float(quaternion[1]),
                "qy": float(quaternion[2]),
                "qz": float(quaternion[3]),
            },
        )

    @staticmethod
    def _rotation_between_vectors(
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        source = source / np.linalg.norm(source)
        target = target / np.linalg.norm(target)
        cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))

        if np.isclose(cosine, 1.0):
            return np.eye(3)

        if np.isclose(cosine, -1.0):
            rotation_axis = np.cross(source, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(rotation_axis) < 1e-6:
                rotation_axis = np.cross(source, np.array([0.0, 1.0, 0.0]))
            rotation_axis /= np.linalg.norm(rotation_axis)
            skew_symmetric = ScenePreprocessor._skew_symmetric_matrix(rotation_axis)
            return np.eye(3) + 2.0 * skew_symmetric @ skew_symmetric

        cross_product = np.cross(source, target)
        skew_symmetric = ScenePreprocessor._skew_symmetric_matrix(cross_product)
        scale = (1.0 - cosine) / np.dot(cross_product, cross_product)
        return np.eye(3) + skew_symmetric + skew_symmetric @ skew_symmetric * scale

    @staticmethod
    def _skew_symmetric_matrix(vector: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [0.0, -vector[2], vector[1]],
                [vector[2], 0.0, -vector[0]],
                [-vector[1], vector[0], 0.0],
            ]
        )

    @staticmethod
    def _quaternion_to_rotation_matrix(
        quaternion: dict,
    ) -> np.ndarray:
        normalized_quaternion = np.array(
            [
                quaternion["qw"],
                quaternion["qx"],
                quaternion["qy"],
                quaternion["qz"],
            ]
        )
        normalized_quaternion /= np.linalg.norm(normalized_quaternion)
        w, x, y, z = normalized_quaternion
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ]
        )

    @staticmethod
    def _rotation_matrix_to_quaternion(
        rotation: np.ndarray,
    ) -> np.ndarray:
        trace = np.trace(rotation)
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
        else:
            dominant_index = int(np.argmax(np.diag(rotation)))
            next_index = (dominant_index + 1) % 3
            final_index = (dominant_index + 2) % 3
            scale = 2.0 * np.sqrt(
                1.0
                + rotation[dominant_index, dominant_index]
                - rotation[next_index, next_index]
                - rotation[final_index, final_index]
            )
            quaternion = np.zeros(4)
            quaternion[0] = (rotation[final_index, next_index] - rotation[next_index, final_index]) / scale
            quaternion[dominant_index + 1] = 0.25 * scale
            quaternion[next_index + 1] = (
                rotation[next_index, dominant_index] + rotation[dominant_index, next_index]
            ) / scale
            quaternion[final_index + 1] = (
                rotation[final_index, dominant_index] + rotation[dominant_index, final_index]
            ) / scale

        return quaternion / np.linalg.norm(quaternion)

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
        intrinsics: list[ColmapIntrinsic],
    ) -> None:
        output_path = self.output_folder / self.configuration.intrinsics_filename

        with open(output_path, "w") as file:
            json.dump(
                [
                    {key: value for key, value in asdict(intrinsic).items() if value is not None}
                    for intrinsic in intrinsics
                ],
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

        positions = torch.tensor([point.xyz for point in colmap_results.points], dtype=torch.float32)
        colors_rgb = torch.tensor([point.rgb for point in colmap_results.points], dtype=torch.float32) / 255.0

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
        colmap_results: ColmapResults,
    ) -> None:
        if self.configuration.example_image_filename is None:
            return

        if not colmap_results.intrinsics or not colmap_results.poses:
            logger.warning("COLMAP did not produce intrinsics or poses. Skipping example rendering.")
            return

        example_pose = colmap_results.poses[0]

        camera_meta = next(
            (
                c
                for c in colmap_results.intrinsics
                if c.camera_id == example_pose.camera_id
            ),
            colmap_results.intrinsics[0],
        )  # fmt:skip

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
        except Exception as e:
            logger.error(f"Failed to render example image: {e}")
