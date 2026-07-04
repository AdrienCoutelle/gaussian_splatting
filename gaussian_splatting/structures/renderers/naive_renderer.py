from typing import Literal

import torch
import torch.utils.checkpoint
from pydantic import ConfigDict
from tqdm import tqdm

from gaussian_splatting.structures.renderers.base_renderer import BaseRenderer, RendererParams, ScreenSpaceGaussians
from gaussian_splatting.utils.profiler import profile


def _compute_gaussian_batch_contribution(
    batch_means: torch.Tensor,
    batch_conics: torch.Tensor,
    batch_colors: torch.Tensor,
    batch_opacities: torch.Tensor,
    batch_extents: torch.Tensor,
    tile_pixels: torch.Tensor,
    tile_transmittance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the color contribution and updated transmittance for one batch of Gaussians over a tile.

    :param batch_means: (G, 2) screen-space means for this batch.
    :param batch_conics: (G, 3) [a, b, c] inverse covariance entries.
    :param batch_colors: (G, 3) RGB colors.
    :param batch_opacities: (G,) per-Gaussian opacities.
    :param batch_extents: (G,) pixel-space radii for early rejection.
    :param tile_pixels: (P, 2) pixel center coordinates for the tile.
    :param tile_transmittance: (P,) accumulated transmittance before this batch.
    :return: (tile_image_delta (P, 3), new_tile_transmittance (P,))
    """
    delta = tile_pixels.unsqueeze(0) - batch_means.unsqueeze(1)  # (G, P, 2)
    delta_x = delta[..., 0]
    delta_y = delta[..., 1]

    mahalanobis = (
        batch_conics[:, 0].unsqueeze(1) * delta_x.square()
        + 2.0 * batch_conics[:, 1].unsqueeze(1) * delta_x * delta_y
        + batch_conics[:, 2].unsqueeze(1) * delta_y.square()
    )  # (G, P)

    within_extent = (torch.abs(delta_x) <= batch_extents.unsqueeze(1)) & (
        torch.abs(delta_y) <= batch_extents.unsqueeze(1)
    )  # (G, P)

    alpha = torch.clamp(torch.exp(-0.5 * mahalanobis) * batch_opacities.unsqueeze(1), 0.0, 1.0)
    alpha = torch.where(within_extent, alpha, torch.zeros_like(alpha))

    one_minus_alpha = 1.0 - alpha  # (G, P)
    transmittance_before = torch.cat(
        [
            torch.ones((1, alpha.shape[1]), device=alpha.device, dtype=alpha.dtype),
            torch.cumprod(one_minus_alpha[:-1], dim=0),
        ],
        dim=0,
    )  # (G, P)

    contribution = alpha * transmittance_before * tile_transmittance.unsqueeze(0)  # (G, P)
    tile_image_delta = contribution.transpose(0, 1) @ batch_colors  # (P, 3)
    new_tile_transmittance = tile_transmittance * torch.prod(one_minus_alpha, dim=0)  # (P,)

    return tile_image_delta, new_tile_transmittance


class NaiveRendererParams(RendererParams):
    name: Literal["naive"]

    model_config = ConfigDict(extra="forbid")

    width: int
    height: int
    focal_length: float

    near_plane: float = 1e-4
    covariance_regularization: float = 0.3
    gaussian_extent: float = 3.0
    tile_size: int = 16
    max_gaussians_per_batch: int = 1024
    use_checkpointing: bool = False


@profile
class NaiveRenderer(BaseRenderer):
    def _splat_gaussians_vectorized(
        self,
        image: torch.Tensor,
        gaussians: ScreenSpaceGaussians,
        sorted_indices: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> None:
        if len(sorted_indices) == 0:
            return

        means_2d = gaussians.means_2d[sorted_indices]
        covariances_2d = gaussians.covariances_2d[sorted_indices]
        colors = gaussians.colors[sorted_indices]
        opacities = gaussians.opacities[sorted_indices].reshape(-1)

        a = covariances_2d[:, 0, 0]
        b = covariances_2d[:, 0, 1]
        c = covariances_2d[:, 1, 1]
        determinant = torch.clamp(a * c - b * b, min=1e-10)
        inverse_determinant = 1.0 / determinant
        conics = torch.stack(
            [
                c * inverse_determinant,
                -b * inverse_determinant,
                a * inverse_determinant,
            ],
            dim=1,
        )

        max_variance = torch.maximum(a, c) + torch.abs(b)
        max_extent = self.config.gaussian_extent * torch.sqrt(torch.clamp(max_variance, min=1e-10))

        min_x = torch.floor(means_2d[:, 0] - max_extent).to(dtype=torch.int64)
        max_x = torch.ceil(means_2d[:, 0] + max_extent).to(dtype=torch.int64)
        min_y = torch.floor(means_2d[:, 1] - max_extent).to(dtype=torch.int64)
        max_y = torch.ceil(means_2d[:, 1] + max_extent).to(dtype=torch.int64)

        min_x = torch.clamp(min_x, min=0, max=image_width)
        max_x = torch.clamp(max_x, min=0, max=image_width)
        min_y = torch.clamp(min_y, min=0, max=image_height)
        max_y = torch.clamp(max_y, min=0, max=image_height)

        valid_mask = (min_x < max_x) & (min_y < max_y)
        if not torch.any(valid_mask):
            return

        means_2d = means_2d[valid_mask]
        conics = conics[valid_mask]
        colors = colors[valid_mask]
        opacities = opacities[valid_mask]
        max_extent = max_extent[valid_mask]
        min_x = min_x[valid_mask]
        max_x = max_x[valid_mask]
        min_y = min_y[valid_mask]
        max_y = max_y[valid_mask]

        tile_size = self.config.tile_size
        max_gaussians_per_batch = self.config.max_gaussians_per_batch

        for tile_min_y in tqdm(range(0, image_height, tile_size), desc="Splatting tiles", leave=False):
            tile_max_y = min(tile_min_y + tile_size, image_height)

            y_indices = torch.arange(tile_min_y, tile_max_y, device=self.device, dtype=torch.int64)
            y_centers = y_indices.to(dtype=torch.float32) + 0.5

            for tile_min_x in range(0, image_width, tile_size):
                tile_max_x = min(tile_min_x + tile_size, image_width)

                overlaps_tile = (
                    (min_x < tile_max_x) & (max_x > tile_min_x) & (min_y < tile_max_y) & (max_y > tile_min_y)
                )
                if not torch.any(overlaps_tile):
                    continue

                tile_means = means_2d[overlaps_tile]
                tile_conics = conics[overlaps_tile]
                tile_colors = colors[overlaps_tile]
                tile_opacities = opacities[overlaps_tile]
                tile_extents = max_extent[overlaps_tile]

                x_indices = torch.arange(tile_min_x, tile_max_x, device=self.device, dtype=torch.int64)
                x_centers = x_indices.to(dtype=torch.float32) + 0.5

                tile_y_coords, tile_x_coords = torch.meshgrid(y_centers, x_centers, indexing="ij")
                tile_pixels = torch.stack([tile_x_coords, tile_y_coords], dim=-1).reshape(-1, 2)

                tile_image = torch.zeros((tile_pixels.shape[0], 3), device=self.device, dtype=torch.float32)
                tile_transmittance = torch.ones(tile_pixels.shape[0], device=self.device, dtype=torch.float32)

                for start_idx in range(0, tile_means.shape[0], max_gaussians_per_batch):
                    end_idx = start_idx + max_gaussians_per_batch

                    batch_means = tile_means[start_idx:end_idx]
                    batch_conics = tile_conics[start_idx:end_idx]
                    batch_colors = tile_colors[start_idx:end_idx]
                    batch_opacities = tile_opacities[start_idx:end_idx]
                    batch_extents = tile_extents[start_idx:end_idx]

                    if self.config.use_checkpointing and torch.is_grad_enabled():
                        tile_image_delta, tile_transmittance = torch.utils.checkpoint.checkpoint(
                            _compute_gaussian_batch_contribution,
                            batch_means,
                            batch_conics,
                            batch_colors,
                            batch_opacities,
                            batch_extents,
                            tile_pixels,
                            tile_transmittance,
                            use_reentrant=False,
                        )
                    else:
                        tile_image_delta, tile_transmittance = _compute_gaussian_batch_contribution(
                            batch_means,
                            batch_conics,
                            batch_colors,
                            batch_opacities,
                            batch_extents,
                            tile_pixels,
                            tile_transmittance,
                        )

                    tile_image = tile_image + tile_image_delta

                tile_height = tile_max_y - tile_min_y
                tile_width = tile_max_x - tile_min_x
                image[tile_min_y:tile_max_y, tile_min_x:tile_max_x] += tile_image.reshape(tile_height, tile_width, 3)
