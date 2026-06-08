from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from pydantic import BaseModel

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import Gaussian, GaussianCollection
from gaussian_splatting.utils.profiler import profile


def _quaternions_to_rotation_matrices(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert (N, 4) quaternions [w, x, y, z] to (N, 3, 3) rotation matrices."""
    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    return torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=1,
    ).reshape(-1, 3, 3)


@dataclass
class Image:
    array: np.ndarray

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]


@dataclass
class ScreenSpaceGaussians:
    means_2d: torch.Tensor
    covariances_2d: torch.Tensor
    depths: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor


class RendererParams(BaseModel, ABC):
    pass


@profile
class BaseRenderer(ABC):
    def __init__(
        self,
        configuration: RendererParams,
        device: torch.device,
    ):
        self.config = configuration
        self.device = device

    def render(
        self,
        camera: Camera,
        gaussians: list[Gaussian],
    ) -> Image:
        gaussian_collection = GaussianCollection(gaussians=gaussians)

        camera_space_gaussians = self._transform_to_camera_space(
            camera=camera,
            gaussians=gaussian_collection,
        )

        screen_space_gaussians = self._project_to_screen_space(
            camera=camera,
            gaussians=camera_space_gaussians,
        )

        output_image = torch.zeros(
            (camera.h, camera.w, 3),
            device=self.device,
            dtype=torch.float32,
        )

        if screen_space_gaussians is None:
            return Image(
                array=output_image.clamp(0.0, 1.0).detach().cpu().numpy(),
            )

        sorted_indices = torch.argsort(screen_space_gaussians.depths, descending=False)

        self._splat_gaussians_vectorized(
            image=output_image,
            gaussians=screen_space_gaussians,
            sorted_indices=sorted_indices,
            image_height=camera.h,
            image_width=camera.w,
        )

        return Image(
            array=output_image.clamp(0.0, 1.0).detach().cpu().numpy(),
        )

    def _transform_to_camera_space(
        self,
        camera: Camera,
        gaussians: GaussianCollection,
    ) -> GaussianCollection:
        world_to_camera = camera.pose[:3, :3].to(device=self.device, dtype=torch.float32).transpose(0, 1)
        camera_position = camera.pose[:3, 3].to(device=self.device, dtype=torch.float32)

        means = gaussians.means.to(device=self.device, dtype=torch.float32)
        colors = gaussians.colors.to(device=self.device, dtype=torch.float32)
        opacities = gaussians.opacities.to(device=self.device, dtype=torch.float32)

        # Transform all means at once: p_camera = R^T @ (p_world - t)
        camera_means = (means - camera_position) @ world_to_camera.T

        return GaussianCollection.from_tensors(
            means=camera_means,
            quaternions=gaussians.quaternions.to(device=self.device, dtype=torch.float32),
            scales=gaussians.scales.to(device=self.device, dtype=torch.float32),
            colors=colors,
            opacities=opacities,
        )

    def _project_to_screen_space(
        self,
        camera: Camera,
        gaussians: GaussianCollection,
    ) -> ScreenSpaceGaussians | None:
        principal_point_x = camera.w / 2.0
        principal_point_y = camera.h / 2.0

        # Extract camera means (N, 3)
        camera_means = gaussians.means
        depths = -camera_means[:, 2]

        # Filter out gaussians behind the near plane
        valid_mask = depths > self.config.near_plane
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            return None

        # Filter all tensors
        valid_means = camera_means[valid_indices]
        valid_quaternions = gaussians.quaternions[valid_indices]
        valid_scales = gaussians.scales[valid_indices]
        valid_colors = gaussians.colors[valid_indices]
        valid_opacities = gaussians.opacities[valid_indices]
        valid_depths = depths[valid_indices]

        # Project to 2D (N, 2)
        means_2d = torch.stack(
            [
                camera.f * (valid_means[:, 0] / valid_depths) + principal_point_x,
                principal_point_y - camera.f * (valid_means[:, 1] / valid_depths),
            ],
            dim=1,
        )

        # Compute Jacobian for all gaussians at once (N, 2, 3)
        N = len(valid_indices)
        jacobians = torch.zeros((N, 2, 3), device=self.device, dtype=torch.float32)
        jacobians[:, 0, 0] = camera.f / valid_depths
        jacobians[:, 0, 2] = camera.f * valid_means[:, 0] / (valid_depths**2)
        jacobians[:, 1, 1] = -camera.f / valid_depths
        jacobians[:, 1, 2] = -camera.f * valid_means[:, 1] / (valid_depths**2)

        # Compute 3D world-space covariance from quaternion and scale: R @ S^2 @ R^T
        R = _quaternions_to_rotation_matrices(valid_quaternions)  # (N, 3, 3)
        S_squared = torch.diag_embed(valid_scales.square())  # (N, 3, 3)
        world_covariances = R @ S_squared @ R.transpose(1, 2)  # (N, 3, 3)

        # Transform to camera space: W_rot @ Σ_world @ W_rot^T
        # world_to_camera is (3, 3) and broadcasts over the batch dimension
        world_to_camera_rot = camera.pose[:3, :3].to(device=self.device, dtype=torch.float32).transpose(0, 1)
        camera_covariances = world_to_camera_rot @ world_covariances @ world_to_camera_rot.T  # (N, 3, 3)

        # Compute 2D covariances: J @ Σ_cam @ J^T for all gaussians
        covariances_2d = torch.bmm(torch.bmm(jacobians, camera_covariances), jacobians.transpose(1, 2))

        # Add regularization
        # regularization = torch.eye(2, device=self.device, dtype=torch.float32) * self.config.covariance_regularization
        # covariances_2d += regularization.unsqueeze(0)

        return ScreenSpaceGaussians(
            means_2d=means_2d,
            covariances_2d=covariances_2d,
            depths=valid_depths,
            colors=valid_colors,
            opacities=valid_opacities,
        )

    @abstractmethod
    def _splat_gaussians_vectorized(
        self,
        image: torch.Tensor,
        gaussians: ScreenSpaceGaussians,
        sorted_indices: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> None:
        pass
