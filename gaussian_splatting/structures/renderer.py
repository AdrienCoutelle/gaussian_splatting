from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import Gaussian, GaussianCollection
from gaussian_splatting.utils.profiler import profile


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
class GSRendererConfig:
    width: int
    height: int
    focal_length: float
    near_plane: float = 1e-4
    covariance_regularization: float = 0.3
    gaussian_extent: float = 3.0

    @classmethod
    def from_dict(
        cls,
        config_dict: dict,
    ) -> "GSRendererConfig":
        if not isinstance(config_dict, dict):
            raise ValueError(f"GSRendererConfig must be a dictionary, got '{type(config_dict).__name__}'.")

        mandatory_fields = {
            "width",
            "height",
            "focal_length",
        }
        if not set(config_dict.keys()).issuperset(mandatory_fields):
            missing_fields = mandatory_fields - set(config_dict.keys())
            raise ValueError(
                f"GSRendererConfig is missing the following mandatory fields: {', '.join(missing_fields)}, "
                f"got {', '.join(config_dict.keys())}."
            )

        width = config_dict["width"]
        height = config_dict["height"]
        focal_length = config_dict["focal_length"]
        near_plane = config_dict.get("near_plane", 1e-4)
        covariance_regularization = config_dict.get("covariance_regularization", 0.3)
        gaussian_extent = config_dict.get("gaussian_extent", 3.0)

        return GSRendererConfig(
            width=width,
            height=height,
            focal_length=focal_length,
            near_plane=near_plane,
            covariance_regularization=covariance_regularization,
            gaussian_extent=gaussian_extent,
        )


@dataclass
class ScreenSpaceGaussian:
    mean_2d: torch.Tensor
    covariance_2d: torch.Tensor
    depth: float
    color: torch.Tensor
    opacity: torch.Tensor


@profile
class GSRenderer:
    def __init__(
        self,
        config: GSRendererConfig,
        device: torch.device,
    ) -> None:
        self.config = config
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

        depths = torch.tensor([g.depth for g in screen_space_gaussians], device=self.device)
        sorted_indices = torch.argsort(depths, descending=False)

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
        covariances = gaussians.covariances.to(device=self.device, dtype=torch.float32)
        colors = gaussians.colors.to(device=self.device, dtype=torch.float32)
        opacities = gaussians.opacities.to(device=self.device, dtype=torch.float32)

        # Transform all means at once: p_camera = R^T @ (p_world - t)
        # means: (N, 3), camera_position: (3,) -> (N, 3)
        camera_means = (means - camera_position) @ world_to_camera.T

        # Transform all covariances at once: R^T @ Σ @ R
        # covariances: (N, 3, 3), world_to_camera: (3, 3)
        camera_covariances = world_to_camera @ covariances @ world_to_camera.T

        return GaussianCollection.from_tensors(
            means=camera_means,
            covariances=camera_covariances,
            colors=colors,
            opacities=opacities,
        )

    def _project_to_screen_space(
        self,
        camera: Camera,
        gaussians: GaussianCollection,
    ) -> list[ScreenSpaceGaussian]:
        principal_point_x = camera.w / 2.0
        principal_point_y = camera.h / 2.0

        # Extract camera means (N, 3)
        camera_means = gaussians.means
        depths = -camera_means[:, 2]

        # Filter out gaussians behind the near plane
        valid_mask = depths > self.config.near_plane
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            return []

        # Filter all tensors
        valid_means = camera_means[valid_indices]
        valid_covariances = gaussians.covariances[valid_indices]
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

        # Compute 2D covariances: J @ Σ @ J^T for all gaussians
        # jacobians: (N, 2, 3), valid_covariances: (N, 3, 3)
        covariances_2d = torch.bmm(torch.bmm(jacobians, valid_covariances), jacobians.transpose(1, 2))

        # Add regularization
        # regularization = torch.eye(2, device=self.device, dtype=torch.float32) * self.config.covariance_regularization
        # covariances_2d += regularization.unsqueeze(0)

        # Create list of ScreenSpaceGaussian objects
        output_gaussians = [
            ScreenSpaceGaussian(
                mean_2d=means_2d[i],
                covariance_2d=covariances_2d[i],
                depth=valid_depths[i].item(),
                color=valid_colors[i],
                opacity=valid_opacities[i],
            )
            for i in range(N)
        ]

        return output_gaussians

    def _splat_gaussians_vectorized(
        self,
        image: torch.Tensor,
        gaussians: list[ScreenSpaceGaussian],
        sorted_indices: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> None:
        y_coords, x_coords = torch.meshgrid(
            torch.arange(image_height, device=self.device, dtype=torch.float32),
            torch.arange(image_width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        pixel_positions = torch.stack([x_coords, y_coords], dim=-1) + 0.5

        transmittance = torch.ones((image_height, image_width), device=self.device, dtype=torch.float32)

        for idx in tqdm(sorted_indices, desc="Splatting Gaussians", leave=False):
            gaussian = gaussians[idx.item()]

            mean_2d = gaussian.mean_2d
            cov_2d = gaussian.covariance_2d

            # Compute bounding box based on gaussian extent
            # Use conservative estimate: sqrt(max diagonal element + off-diagonal contribution)
            max_variance = torch.max(cov_2d[0, 0], cov_2d[1, 1]) + torch.abs(cov_2d[0, 1])
            max_extent = self.config.gaussian_extent * torch.sqrt(max_variance)

            min_x = int(torch.floor(mean_2d[0] - max_extent).item())
            max_x = int(torch.ceil(mean_2d[0] + max_extent).item())
            min_y = int(torch.floor(mean_2d[1] - max_extent).item())
            max_y = int(torch.ceil(mean_2d[1] + max_extent).item())

            # Clip to image bounds
            min_x = max(0, min_x)
            max_x = min(image_width, max_x)
            min_y = max(0, min_y)
            max_y = min(image_height, max_y)

            if min_x >= max_x or min_y >= max_y:
                continue

            # Extract relevant pixel positions
            pixels = pixel_positions[min_y:max_y, min_x:max_x]

            # Compute difference from mean
            delta = pixels - mean_2d

            # Compute inverse covariance
            cov_inv = torch.linalg.inv(cov_2d)

            # Mahalanobis distance: (x-μ)^T Σ^-1 (x-μ)
            mahalanobis = torch.sum(delta @ cov_inv * delta, dim=-1)

            # Gaussian weight
            weight = torch.exp(-0.5 * mahalanobis)

            # Apply opacity and clamp to valid range
            alpha = torch.clamp(weight * gaussian.opacity, 0.0, 1.0)

            # Alpha compositing (front to back)
            current_transmittance = transmittance[min_y:max_y, min_x:max_x]
            contribution = alpha * current_transmittance

            # Update image
            image[min_y:max_y, min_x:max_x] += contribution.unsqueeze(-1) * gaussian.color

            # Update transmittance
            transmittance[min_y:max_y, min_x:max_x] *= 1.0 - alpha
