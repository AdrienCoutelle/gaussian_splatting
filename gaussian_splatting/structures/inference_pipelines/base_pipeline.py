import json
import os
from abc import ABC

import cv2
import mlx.core as mx
import numpy as np
from pydantic import BaseModel

from gaussian_splatting.structures.renderer.utils import _quaternions_to_rotation_matrices


class DatasetConfig(BaseModel):
    training_images_path: str
    poses_json_path: str


class InferencePipelineParams(BaseModel, ABC):
    dataset_config_for_closest_view: DatasetConfig | None = None


class BaseInferencePipeline(ABC):
    def __init__(
        self,
        configuration: InferencePipelineParams,
        output_folder: str,
        epoch: int | None = None,
    ):
        self.configuration = configuration
        self.output_folder = output_folder
        self.epoch = epoch

    def _get_closest_view_image(
        self,
        target_pose: mx.array,
    ) -> np.ndarray:
        if self.configuration.dataset_config_for_closest_view is None:
            raise RuntimeError("dataset_config_for_closest_view is not set in the pipeline configuration")

        config = self.configuration.dataset_config_for_closest_view
        target_position = np.array(target_pose[:3, 3].tolist(), dtype=np.float32)

        with open(config.poses_json_path) as f:
            poses_data: list[dict] = json.load(f)

        min_distance = float("inf")
        closest_entry = None

        for entry in poses_data:
            q = mx.array(
                [
                    [
                        entry["rotation"]["qw"],
                        entry["rotation"]["qx"],
                        entry["rotation"]["qy"],
                        entry["rotation"]["qz"],
                    ]
                ]
            )  # (1, 4)
            R_cw = np.array(_quaternions_to_rotation_matrices(q)[0].tolist(), dtype=np.float64)
            t_cw = np.array(
                [
                    entry["position"]["x"],
                    entry["position"]["y"],
                    entry["position"]["z"],
                ],
                dtype=np.float64,
            )
            camera_position = (R_cw @ t_cw).astype(np.float32)

            distance = np.linalg.norm(camera_position - target_position)
            if distance < min_distance:
                min_distance = distance
                closest_entry = entry

        image_stem = os.path.splitext(closest_entry["name"])[0]
        image_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(config.training_images_path, image_stem + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            raise FileNotFoundError(f"No image found for '{image_stem}' in {config.training_images_path}")

        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            raise FileNotFoundError(f"cv2 failed to load image: {image_path}")

        return bgr_image

    @staticmethod
    def _resize_to_fit(
        image: np.ndarray,
        target_width: int,
        target_height: int,
    ) -> np.ndarray:
        """Resize image to fit target dimensions while preserving aspect ratio, padding with black."""
        src_h, src_w = image.shape[:2]
        scale = min(target_width / src_w, target_height / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return canvas

    def _compute_pose_look_at(
        self,
        position: np.ndarray,
        look_at: np.ndarray,
        world_up: np.ndarray,
    ) -> np.ndarray:
        dir_vec = look_at - position
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm < 1e-6:
            raise ValueError("Position and look_at are too close together")
        dir_vec /= dir_norm

        right_vec = np.cross(dir_vec, world_up)
        right_norm = np.linalg.norm(right_vec)
        if right_norm < 1e-6:
            # dir_vec and world_up are parallel, use alternative up vector
            alternative_up = np.array([1.0, 0.0, 0.0]) if abs(dir_vec[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            right_vec = np.cross(dir_vec, alternative_up)
            right_norm = np.linalg.norm(right_vec)
        right_vec /= right_norm

        up_vec = np.cross(right_vec, dir_vec)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, 0] = right_vec
        pose[:3, 1] = up_vec
        pose[:3, 2] = dir_vec
        pose[:3, 3] = position

        return pose
