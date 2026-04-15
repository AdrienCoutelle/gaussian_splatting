from dataclasses import dataclass

import numpy as np
import torch

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import Gaussian
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
    near_plane: float = 1e-4
    covariance_regularization: float = 0.3
    gaussian_extent: float = 3.0


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
        camera_space_gaussians = self._transform_to_camera_space(
            camera=camera,
            gaussians=gaussians,
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
        gaussians: list[Gaussian],
    ) -> list[Gaussian]:
        output_gaussians = []

        for gaussian in gaussians:
            # p_camera = R^T @ (p_world - t)
            world_to_camera = camera.pose[:3, :3].to(device=self.device, dtype=torch.float32).transpose(0, 1)
            camera_position = camera.pose[:3, 3].to(device=self.device, dtype=torch.float32)

            mean = gaussian.mean.to(device=self.device, dtype=torch.float32)
            covariance = gaussian.covariance.to(device=self.device, dtype=torch.float32)
            color = gaussian.color.to(device=self.device, dtype=torch.float32)
            opacity = gaussian.opacity.to(device=self.device, dtype=torch.float32)

            camera_mean = world_to_camera @ (mean - camera_position)
            camera_covariance = world_to_camera @ covariance @ world_to_camera.transpose(0, 1)

            output_gaussians.append(
                Gaussian(
                    mean=camera_mean,
                    covariance=camera_covariance,
                    color=color,
                    opacity=opacity,
                )
            )

        return output_gaussians

    def _project_to_screen_space(
        self,
        camera: Camera,
        gaussians: list[Gaussian],
    ) -> list[ScreenSpaceGaussian]:
        principal_point_x = camera.w / 2.0
        principal_point_y = camera.h / 2.0

        output_gaussians = []

        for gaussian in gaussians:
            camera_mean = gaussian.mean
            depth = -camera_mean[2].item()

            if depth <= self.config.near_plane:
                continue

            mean_2d = torch.tensor(
                [
                    camera.f * (camera_mean[0] / depth) + principal_point_x,
                    principal_point_y - camera.f * (camera_mean[1] / depth),
                ],
                device=self.device,
                dtype=torch.float32,
            )

            jacobian = torch.tensor(
                [
                    [camera.f / depth, 0.0, camera.f * camera_mean[0] / (depth**2)],
                    [0.0, -camera.f / depth, -camera.f * camera_mean[1] / (depth**2)],
                ],
                device=self.device,
                dtype=torch.float32,
            )

            covariance_2d = jacobian @ gaussian.covariance @ jacobian.transpose(0, 1)
            covariance_2d += torch.eye(2, device=self.device) * self.config.covariance_regularization

            output_gaussians.append(
                ScreenSpaceGaussian(
                    mean_2d=mean_2d,
                    covariance_2d=covariance_2d,
                    depth=depth,
                    color=gaussian.color,
                    opacity=gaussian.opacity,
                )
            )

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

        for idx in sorted_indices:
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
