from dataclasses import dataclass

import numpy as np
import torch

from gaussian_splatting.structures.gaussian import Gaussian


@dataclass
class Camera:
    pose: torch.Tensor
    focal_length: float
    width: int
    height: int

    @property
    def f(self) -> float:
        return self.focal_length

    @property
    def h(self) -> int:
        return self.height

    @property
    def w(self) -> int:
        return self.width


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
class ProjectedGaussian:
    mean: torch.Tensor
    covariance: torch.Tensor
    depth: float
    color: torch.Tensor
    opacity: torch.Tensor


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
        camera_space_gaussians = self._gaussians_to_camera_space(
            camera=camera,
            gaussians=gaussians,
        )

        ray_space_gaussians = self._gaussians_to_ray_space(
            camera=camera,
            gaussians=camera_space_gaussians,
        )

        output_image = torch.zeros(
            (camera.h, camera.w, 3),
            device=self.device,
            dtype=torch.float32,
        )

        sorted_gaussians = sorted(
            ray_space_gaussians,
            key=lambda gaussian: gaussian.depth,
            reverse=True,
        )

        for gaussian in sorted_gaussians:
            self._render_projected_gaussian(
                image=output_image,
                gaussian=gaussian,
            )

        return Image(
            array=output_image.clamp(0.0, 1.0).detach().cpu().numpy(),
        )

    def _gaussians_to_camera_space(
        self,
        camera: Camera,
        gaussians: list[Gaussian],
    ) -> list[Gaussian]:  # TODO: Inplace conversion ?
        world_to_camera = camera.pose[:3, :3].to(device=self.device, dtype=torch.float32).transpose(0, 1)
        camera_position = camera.pose[:3, 3].to(device=self.device, dtype=torch.float32)

        output_gaussians: list[Gaussian] = []
        for gaussian in gaussians:
            mean = gaussian.mean.to(device=self.device, dtype=torch.float32)
            covariance = gaussian.covariance.to(device=self.device, dtype=torch.float32)
            color = gaussian.color.to(device=self.device, dtype=torch.float32)
            opacity = gaussian.opacity.to(device=self.device, dtype=torch.float32)

            centered_mean = mean - camera_position
            camera_mean = world_to_camera @ centered_mean
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

    def _gaussians_to_ray_space(
        self,
        camera: Camera,
        gaussians: list[Gaussian],
    ) -> list[ProjectedGaussian]:
        projected_gaussians: list[ProjectedGaussian] = []
        principal_point_x = camera.w / 2.0
        principal_point_y = camera.h / 2.0

        for gaussian in gaussians:
            x_cam, y_cam, z_cam = gaussian.mean
            depth = float(-z_cam.item())

            if depth <= self.config.near_plane:
                continue

            mean_2d = torch.tensor(
                [
                    camera.f * (x_cam / depth) + principal_point_x,
                    principal_point_y - camera.f * (y_cam / depth),
                ],
                device=self.device,
                dtype=torch.float32,
            )

            jacobian = torch.tensor(
                [
                    [camera.f / depth, 0.0, camera.f * x_cam / (depth**2)],
                    [0.0, -camera.f / depth, -camera.f * y_cam / (depth**2)],
                ],
                device=self.device,
                dtype=torch.float32,
            )

            covariance_2d = jacobian @ gaussian.covariance @ jacobian.transpose(0, 1)
            covariance_2d[0, 0] += self.config.covariance_regularization
            covariance_2d[1, 1] += self.config.covariance_regularization

            projected_gaussians.append(
                ProjectedGaussian(
                    mean=mean_2d,
                    covariance=covariance_2d,
                    depth=depth,
                    color=gaussian.color,
                    opacity=gaussian.opacity,
                )
            )

        return projected_gaussians

    def _render_projected_gaussian(
        self,
        image: torch.Tensor,
        gaussian: ProjectedGaussian,
    ) -> None:
        determinant = torch.linalg.det(gaussian.covariance)
        if determinant <= 0:
            return

        inverse_covariance = torch.linalg.inv(gaussian.covariance)

        std_x = float(torch.sqrt(gaussian.covariance[0, 0]).item())
        std_y = float(torch.sqrt(gaussian.covariance[1, 1]).item())

        x_min = max(0, int(np.floor(float(gaussian.mean[0].item() - self.config.gaussian_extent * std_x))))
        x_max = min(image.shape[1], int(np.ceil(float(gaussian.mean[0].item() + self.config.gaussian_extent * std_x))))
        y_min = max(0, int(np.floor(float(gaussian.mean[1].item() - self.config.gaussian_extent * std_y))))
        y_max = min(image.shape[0], int(np.ceil(float(gaussian.mean[1].item() + self.config.gaussian_extent * std_y))))

        if x_max <= x_min or y_max <= y_min:
            return

        xs = torch.arange(x_min, x_max, device=self.device)
        ys = torch.arange(y_min, y_max, device=self.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        sample_positions = torch.stack([grid_x, grid_y], dim=-1).to(dtype=torch.float32)

        deltas = sample_positions - gaussian.mean
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
