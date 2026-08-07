import json
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class WorldCoordinateTransform:
    """Rigid transform from the original COLMAP world frame to the normalized world frame."""

    rotation: np.ndarray
    translation: np.ndarray

    def apply_to_positions(
        self,
        positions: np.ndarray,
    ) -> np.ndarray:
        return positions @ self.rotation.T + self.translation

    def apply_to_world_to_camera_pose(
        self,
        rotation_world_to_camera: np.ndarray,
        translation_world_to_camera: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        camera_center_world = -rotation_world_to_camera.T @ translation_world_to_camera
        normalized_rotation_world_to_camera = rotation_world_to_camera @ self.rotation.T
        normalized_camera_center_world = self.apply_to_positions(camera_center_world)
        normalized_translation_world_to_camera = -normalized_rotation_world_to_camera @ normalized_camera_center_world
        return normalized_rotation_world_to_camera, normalized_translation_world_to_camera


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
        """Rotate the reconstruction to Z-up, ground it on Z=0, and align its main axis with X.

        COLMAP camera poses are stored as world-to-camera transforms. Apply the same world rotation
        and translation to camera centers and reconstructed points so the saved poses and PLY remain
        in the same coordinate system.
        """
        if not colmap_results.poses:
            raise ValueError("COLMAP reconstruction did not produce any camera poses.")
        if not colmap_results.points:
            raise ValueError("COLMAP reconstruction did not produce any 3D points.")

        point_positions = np.asarray([point.xyz for point in colmap_results.points])
        world_transform = self._build_world_transform(
            poses=colmap_results.poses,
            point_positions=point_positions,
        )
        processed_point_positions = world_transform.apply_to_positions(point_positions)

        processed_poses = [
            self._rotate_pose(
                pose=pose,
                world_transform=world_transform,
            )
            for pose in colmap_results.poses
        ]
        processed_points = [
            ColmapPoint(
                point_id=point.point_id,
                xyz=processed_position.tolist(),
                rgb=point.rgb,
                error=point.error,
                track_length=point.track_length,
            )
            for point, processed_position in zip(
                colmap_results.points,
                processed_point_positions,
                strict=True,
            )
        ]

        logger.info("Rotated and grounded the COLMAP reconstruction with its main axis aligned to X.")
        return ColmapResults(
            poses=processed_poses,
            intrinsics=colmap_results.intrinsics,
            points=processed_points,
        )

    def _build_world_transform(
        self,
        poses: list[ColmapPose],
        point_positions: np.ndarray,
    ) -> WorldCoordinateTransform:
        """Build one transform shared by every sparse point and camera pose."""
        average_camera_up = self._average_camera_up_direction(poses)
        rotate_to_z_up = self._rotation_between_vectors(
            source=average_camera_up,
            target=np.array([0.0, 0.0, 1.0]),
        )

        z_up_point_positions = point_positions @ rotate_to_z_up.T
        align_main_axis_with_x = self._horizontal_principal_axis_rotation(z_up_point_positions)
        world_rotation = align_main_axis_with_x @ rotate_to_z_up

        rotated_point_positions = point_positions @ world_rotation.T
        world_translation = self._center_horizontally_and_ground(rotated_point_positions)
        return WorldCoordinateTransform(
            rotation=world_rotation,
            translation=world_translation,
        )

    def _average_camera_up_direction(
        self,
        poses: list[ColmapPose],
    ) -> np.ndarray:
        camera_to_world_rotations = np.stack([self._quaternion_to_rotation_matrix(pose.rotation).T for pose in poses])
        # COLMAP's camera Y axis points down, so camera up is its negative Y axis.
        average_camera_up = -camera_to_world_rotations[:, :, 1].mean(axis=0)

        if np.linalg.norm(average_camera_up) < 1e-6:
            raise ValueError("Could not determine an average camera up direction from the COLMAP poses.")

        return average_camera_up

    @staticmethod
    def _center_horizontally_and_ground(
        point_positions: np.ndarray,
    ) -> np.ndarray:
        point_center = point_positions.mean(axis=0)
        lowest_point_z = point_positions[:, 2].min()
        return np.array(
            [
                -point_center[0],
                -point_center[1],
                -lowest_point_z,
            ]
        )

    @staticmethod
    def _horizontal_principal_axis_rotation(
        point_positions: np.ndarray,
    ) -> np.ndarray:
        """Return a Z-axis rotation that maps the dominant horizontal PCA direction to positive X."""
        centered_xy = point_positions[:, :2] - point_positions[:, :2].mean(axis=0)
        covariance = centered_xy.T @ centered_xy
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        if eigenvalues[-1] < 1e-12:
            return np.eye(3)

        principal_axis = eigenvectors[:, -1]
        largest_component = np.argmax(np.abs(principal_axis))
        if principal_axis[largest_component] < 0.0:
            principal_axis = -principal_axis

        angle = np.arctan2(principal_axis[1], principal_axis[0])
        cosine = np.cos(angle)
        sine = np.sin(angle)
        return np.array(
            [
                [cosine, sine, 0.0],
                [-sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    def _rotate_pose(
        pose: ColmapPose,
        world_transform: WorldCoordinateTransform,
    ) -> ColmapPose:
        rotation_world_to_camera = ScenePreprocessor._quaternion_to_rotation_matrix(pose.rotation)
        translation_world_to_camera = np.array(
            [
                pose.position["x"],
                pose.position["y"],
                pose.position["z"],
            ]
        )

        processed_rotation_world_to_camera, processed_translation_world_to_camera = (
            world_transform.apply_to_world_to_camera_pose(
                rotation_world_to_camera=rotation_world_to_camera,
                translation_world_to_camera=translation_world_to_camera,
            )
        )

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
        except Exception as e:
            logger.error(f"Failed to render example image: {e}")
