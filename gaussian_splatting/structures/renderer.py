from dataclasses import dataclass

import numpy as np
import torch

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import GaussianCollection


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


@dataclass
class ScreenSpaceGaussianCollection:
    means_2d: torch.Tensor
    covariances_2d: torch.Tensor
    depths: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor

    def __len__(self) -> int:
        return int(self.depths.shape[0])


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
        gaussian_collection: GaussianCollection,
    ) -> Image:
        camera_space_gaussians = self._transform_to_camera_space(
            camera=camera,
            gaussian_collection=gaussian_collection,
        )

        screen_space_gaussians = self._project_to_screen_space(
            camera=camera,
            gaussian_collection=camera_space_gaussians,
        )

        output_image = torch.zeros(
            (camera.h, camera.w, 3),
            device=self.device,
            dtype=torch.float32,
        )

        for gaussian in self._sort_back_to_front(screen_space_gaussians):
            self._splat_gaussian(
                image=output_image,
                gaussian=gaussian,
            )

        return Image(
            array=output_image.clamp(0.0, 1.0).detach().cpu().numpy(),
        )

    def _transform_to_camera_space(
        self,
        camera: Camera,
        gaussian_collection: GaussianCollection,
    ) -> GaussianCollection:
        """Transform world-space gaussians into camera space."""
        world_to_camera = camera.pose[:3, :3].to(device=self.device, dtype=torch.float32).transpose(0, 1)
        camera_position = camera.pose[:3, 3].to(device=self.device, dtype=torch.float32)

        means = gaussian_collection.means.to(device=self.device, dtype=torch.float32)
        covariances = gaussian_collection.covariances.to(device=self.device, dtype=torch.float32)
        colors = gaussian_collection.colors.to(device=self.device, dtype=torch.float32)
        opacities = gaussian_collection.opacities.to(device=self.device, dtype=torch.float32)

        centered_means = means - camera_position.unsqueeze(0)
        camera_means = centered_means @ world_to_camera.transpose(0, 1)
        camera_covariances = torch.einsum(
            "ij,njk,kl->nil",
            world_to_camera,
            covariances,
            world_to_camera.transpose(0, 1),
        )

        return GaussianCollection.from_tensors(
            means=camera_means,
            covariances=camera_covariances,
            colors=colors,
            opacities=opacities,
        )

    def _project_to_screen_space(
        self,
        camera: Camera,
        gaussian_collection: GaussianCollection,
    ) -> ScreenSpaceGaussianCollection:
        """Project camera-space gaussians into 2D image space.

        This follows the standard Gaussian splatting steps:
        - perspective projection of the 3D mean,
        - Jacobian-based projection of the 3D covariance,
        - small diagonal regularization for numerical stability.
        """
        if len(gaussian_collection) == 0:
            return self._empty_screen_space_gaussians()

        principal_point_x = camera.w / 2.0
        principal_point_y = camera.h / 2.0

        camera_means = gaussian_collection.means
        depths = -camera_means[:, 2]
        valid_mask = depths > self.config.near_plane

        if not torch.any(valid_mask):
            return self._empty_screen_space_gaussians()

        valid_means = camera_means[valid_mask]
        valid_depths = depths[valid_mask]
        valid_covariances = gaussian_collection.covariances[valid_mask]
        valid_colors = gaussian_collection.colors[valid_mask]
        valid_opacities = gaussian_collection.opacities[valid_mask]

        mean_x = camera.f * (valid_means[:, 0] / valid_depths) + principal_point_x
        mean_y = principal_point_y - camera.f * (valid_means[:, 1] / valid_depths)
        means_2d = torch.stack([mean_x, mean_y], dim=-1)

        jacobians = torch.zeros(
            (valid_means.shape[0], 2, 3),
            device=self.device,
            dtype=torch.float32,
        )
        jacobians[:, 0, 0] = camera.f / valid_depths
        jacobians[:, 0, 2] = camera.f * valid_means[:, 0] / (valid_depths**2)
        jacobians[:, 1, 1] = -camera.f / valid_depths
        jacobians[:, 1, 2] = -camera.f * valid_means[:, 1] / (valid_depths**2)

        covariance_2d = torch.einsum(
            "nij,njk,nlk->nil",
            jacobians,
            valid_covariances,
            jacobians,
        )
        covariance_2d[:, 0, 0] += self.config.covariance_regularization
        covariance_2d[:, 1, 1] += self.config.covariance_regularization

        return ScreenSpaceGaussianCollection(
            means_2d=means_2d,
            covariances_2d=covariance_2d,
            depths=valid_depths,
            colors=valid_colors,
            opacities=valid_opacities,
        )

    def _sort_back_to_front(
        self,
        gaussian_collection: ScreenSpaceGaussianCollection,
    ) -> list[ScreenSpaceGaussian]:
        if len(gaussian_collection) == 0:
            return []

        sorted_indices = torch.argsort(gaussian_collection.depths, descending=True)

        return [
            ScreenSpaceGaussian(
                mean_2d=gaussian_collection.means_2d[index],
                covariance_2d=gaussian_collection.covariances_2d[index],
                depth=float(gaussian_collection.depths[index].item()),
                color=gaussian_collection.colors[index],
                opacity=gaussian_collection.opacities[index],
            )
            for index in sorted_indices.tolist()
        ]

    def _splat_gaussian(
        self,
        image: torch.Tensor,
        gaussian: ScreenSpaceGaussian,
    ) -> None:
        determinant = torch.linalg.det(gaussian.covariance_2d)
        if determinant <= 0:
            return

        inverse_covariance = torch.linalg.inv(gaussian.covariance_2d)

        std_x = float(torch.sqrt(gaussian.covariance_2d[0, 0]).item())
        std_y = float(torch.sqrt(gaussian.covariance_2d[1, 1]).item())

        x_min = max(
            0,
            int(np.floor(float(gaussian.mean_2d[0].item() - self.config.gaussian_extent * std_x))),
        )
        x_max = min(
            image.shape[1],
            int(np.ceil(float(gaussian.mean_2d[0].item() + self.config.gaussian_extent * std_x))),
        )
        y_min = max(
            0,
            int(np.floor(float(gaussian.mean_2d[1].item() - self.config.gaussian_extent * std_y))),
        )
        y_max = min(
            image.shape[0],
            int(np.ceil(float(gaussian.mean_2d[1].item() + self.config.gaussian_extent * std_y))),
        )

        if x_max <= x_min or y_max <= y_min:
            return

        xs = torch.arange(x_min, x_max, device=self.device)
        ys = torch.arange(y_min, y_max, device=self.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        sample_positions = torch.stack([grid_x, grid_y], dim=-1).to(dtype=torch.float32) + 0.5

        deltas = sample_positions - gaussian.mean_2d
        exponent = -0.5 * torch.einsum(
            "...i,ij,...j->...",
            deltas,
            inverse_covariance,
            deltas,
        )

        opacity = gaussian.opacity.reshape(1).to(dtype=torch.float32)
        alpha = (opacity * torch.exp(exponent)).unsqueeze(-1).clamp(0.0, 1.0)

        image_patch = image[y_min:y_max, x_min:x_max]
        image[y_min:y_max, x_min:x_max] = gaussian.color * alpha + image_patch * (1.0 - alpha)

    def _empty_screen_space_gaussians(self) -> ScreenSpaceGaussianCollection:
        return ScreenSpaceGaussianCollection(
            means_2d=torch.empty((0, 2), device=self.device, dtype=torch.float32),
            covariances_2d=torch.empty((0, 2, 2), device=self.device, dtype=torch.float32),
            depths=torch.empty((0,), device=self.device, dtype=torch.float32),
            colors=torch.empty((0, 3), device=self.device, dtype=torch.float32),
            opacities=torch.empty((0,), device=self.device, dtype=torch.float32),
        )
