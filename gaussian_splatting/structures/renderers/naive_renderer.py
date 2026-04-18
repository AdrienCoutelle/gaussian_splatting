from dataclasses import dataclass

import torch
from tqdm import tqdm

from gaussian_splatting.structures.renderers.base_renderer import BaseRenderer, ScreenSpaceGaussian


@dataclass
class NaiveRendererParams:
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
    ) -> "NaiveRendererParams":
        if not isinstance(config_dict, dict):
            raise ValueError(f"NaiveRendererParams must be a dictionary, got '{type(config_dict).__name__}'.")

        mandatory_fields = {
            "width",
            "height",
            "focal_length",
        }
        if not set(config_dict.keys()).issuperset(mandatory_fields):
            missing_fields = mandatory_fields - set(config_dict.keys())
            raise ValueError(
                f"NaiveRendererParams is missing the following mandatory fields: {', '.join(missing_fields)}, "
                f"got {', '.join(config_dict.keys())}."
            )

        width = config_dict["width"]
        height = config_dict["height"]
        focal_length = config_dict["focal_length"]
        near_plane = config_dict.get("near_plane", 1e-4)
        covariance_regularization = config_dict.get("covariance_regularization", 0.3)
        gaussian_extent = config_dict.get("gaussian_extent", 3.0)

        return NaiveRendererParams(
            width=width,
            height=height,
            focal_length=focal_length,
            near_plane=near_plane,
            covariance_regularization=covariance_regularization,
            gaussian_extent=gaussian_extent,
        )


class NaiveRenderer(BaseRenderer):
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

            max_variance = torch.max(cov_2d[0, 0], cov_2d[1, 1]) + torch.abs(cov_2d[0, 1])
            max_extent = self.config.gaussian_extent * torch.sqrt(max_variance)

            min_x = int(torch.floor(mean_2d[0] - max_extent).item())
            max_x = int(torch.ceil(mean_2d[0] + max_extent).item())
            min_y = int(torch.floor(mean_2d[1] - max_extent).item())
            max_y = int(torch.ceil(mean_2d[1] + max_extent).item())

            min_x = max(0, min_x)
            max_x = min(image_width, max_x)
            min_y = max(0, min_y)
            max_y = min(image_height, max_y)

            if min_x >= max_x or min_y >= max_y:
                continue

            pixels = pixel_positions[min_y:max_y, min_x:max_x]

            delta = pixels - mean_2d

            cov_inv = torch.linalg.inv(cov_2d)

            mahalanobis = torch.sum(delta @ cov_inv * delta, dim=-1)

            weight = torch.exp(-0.5 * mahalanobis)

            alpha = torch.clamp(weight * gaussian.opacity, 0.0, 1.0)

            current_transmittance = transmittance[min_y:max_y, min_x:max_x]
            contribution = alpha * current_transmittance

            image[min_y:max_y, min_x:max_x] += contribution.unsqueeze(-1) * gaussian.color

            transmittance[min_y:max_y, min_x:max_x] *= 1.0 - alpha
