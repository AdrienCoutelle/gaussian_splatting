from abc import ABC

import mlx.core as mx
import numpy as np
from pydantic import BaseModel


class InferencePipelineParams(BaseModel, ABC):
    pass


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
        raise NotImplementedError

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
        pose[:3, 2] = -dir_vec
        pose[:3, 3] = position

        return pose
