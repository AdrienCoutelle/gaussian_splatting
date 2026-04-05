from abc import ABC
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class InferencePipelineParams:
    pass


class BaseInferencePipeline(ABC):
    def __init__(
        self,
        configuration: InferencePipelineParams,
        device: torch.device,
        output_folder: str,
        epoch: int | None = None,
    ):
        self.configuration = configuration
        self.device = device
        self.output_folder = output_folder
        self.epoch = epoch

    def _get_closest_view_image(
        self,
        target_pose: torch.Tensor,
    ) -> np.ndarray:
        raise NotImplementedError

    def _compute_pose_look_at(
        self,
        position: np.ndarray,
        look_at: np.ndarray,
        world_up: np.ndarray,
    ) -> np.ndarray:
        dir_vec = look_at - position
        dir_vec /= np.linalg.norm(dir_vec)

        right_vec = np.cross(dir_vec, world_up)
        right_vec /= np.linalg.norm(right_vec)

        up_vec = np.cross(right_vec, dir_vec)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, 0] = right_vec
        pose[:3, 1] = up_vec
        pose[:3, 2] = -dir_vec
        pose[:3, 3] = position

        return pose
