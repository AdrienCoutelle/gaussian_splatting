from dataclasses import dataclass

import numpy as np

from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.scene_preprocessor.colmap_wrapper import (
    ColmapPoint,
    ColmapPose,
    ColmapResults,
)

logger = Logger("COLMAP_POSTPROCESSOR")


@dataclass(frozen=True)
class WorldCoordinateTransform:
    matrix: np.ndarray

    @classmethod
    def from_rotation_and_translation(
        cls,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> "WorldCoordinateTransform":
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        return cls(matrix=matrix)

    @property
    def inverse_matrix(self) -> np.ndarray:
        rotation = self.matrix[:3, :3]
        translation = self.matrix[:3, 3]

        inverse_matrix = np.eye(4)
        inverse_matrix[:3, :3] = rotation.T
        inverse_matrix[:3, 3] = -rotation.T @ translation
        return inverse_matrix

    def apply_to_positions(
        self,
        positions: np.ndarray,
    ) -> np.ndarray:
        homogeneous_positions = np.concatenate(
            [
                positions,
                np.ones((positions.shape[0], 1)),
            ],
            axis=1,
        )
        transformed_positions = (self.matrix @ homogeneous_positions.T).T
        return transformed_positions[:, :3]

    def apply_to_world_to_camera_matrix(
        self,
        world_to_camera_matrix: np.ndarray,
    ) -> np.ndarray:
        # If X_new = T_new_from_old @ X_old, then X_camera = E_old @ inv(T_new_from_old) @ X_new.
        return world_to_camera_matrix @ self.inverse_matrix


class ColmapPostprocessor:
    def run(
        self,
        colmap_results: ColmapResults,
    ) -> ColmapResults:
        if len(colmap_results.poses) == 0:
            raise ValueError("COLMAP reconstruction did not produce any camera poses.")
        if len(colmap_results.points) == 0:
            raise ValueError("COLMAP reconstruction did not produce any 3D points.")

        world_transform = self.compute_world_transform(colmap_results)
        processed_results = self.apply_world_transform(
            colmap_results=colmap_results,
            world_transform=world_transform,
        )

        logger.info("Aligned the COLMAP reconstruction to world +Z and centered sparse points at the origin.")
        return processed_results

    def compute_world_transform(
        self,
        colmap_results: ColmapResults,
    ) -> WorldCoordinateTransform:
        point_positions = np.asarray([point.xyz for point in colmap_results.points], dtype=np.float64)
        average_camera_up = self._average_camera_up_direction(colmap_results.poses)
        rotate_to_z_up = self._rotation_between_vectors(
            source=average_camera_up,
            target=np.array([0.0, 0.0, 1.0]),
        )
        rotated_point_positions = point_positions @ rotate_to_z_up.T
        translation_to_origin = -rotated_point_positions.mean(axis=0)
        return WorldCoordinateTransform.from_rotation_and_translation(
            rotation=rotate_to_z_up,
            translation=translation_to_origin,
        )

    def apply_world_transform(
        self,
        colmap_results: ColmapResults,
        world_transform: WorldCoordinateTransform,
    ) -> ColmapResults:
        point_positions = np.asarray([point.xyz for point in colmap_results.points], dtype=np.float64)
        processed_point_positions = world_transform.apply_to_positions(point_positions)

        processed_poses = [
            self._transform_pose(
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

        return ColmapResults(
            poses=processed_poses,
            intrinsics=colmap_results.intrinsics,
            points=processed_points,
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
    def _transform_pose(
        pose: ColmapPose,
        world_transform: WorldCoordinateTransform,
    ) -> ColmapPose:
        rotation_world_to_camera = ColmapPostprocessor._quaternion_to_rotation_matrix(pose.rotation)
        translation_world_to_camera = np.array(
            [
                pose.position["x"],
                pose.position["y"],
                pose.position["z"],
            ]
        )

        world_to_camera_matrix = np.eye(4)
        world_to_camera_matrix[:3, :3] = rotation_world_to_camera
        world_to_camera_matrix[:3, 3] = translation_world_to_camera
        processed_world_to_camera_matrix = world_transform.apply_to_world_to_camera_matrix(world_to_camera_matrix)

        processed_rotation_world_to_camera = processed_world_to_camera_matrix[:3, :3]
        processed_translation_world_to_camera = processed_world_to_camera_matrix[:3, 3]

        quaternion = ColmapPostprocessor._rotation_matrix_to_quaternion(processed_rotation_world_to_camera)
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
    def _positions_in_camera_frame(
        positions: np.ndarray,
        pose: ColmapPose,
    ) -> np.ndarray:
        rotation_world_to_camera = ColmapPostprocessor._quaternion_to_rotation_matrix(pose.rotation)
        translation_world_to_camera = np.array(
            [
                pose.position["x"],
                pose.position["y"],
                pose.position["z"],
            ]
        )
        return positions @ rotation_world_to_camera.T + translation_world_to_camera

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
            skew_symmetric = ColmapPostprocessor._skew_symmetric_matrix(rotation_axis)
            return np.eye(3) + 2.0 * skew_symmetric @ skew_symmetric

        cross_product = np.cross(source, target)
        skew_symmetric = ColmapPostprocessor._skew_symmetric_matrix(cross_product)
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
