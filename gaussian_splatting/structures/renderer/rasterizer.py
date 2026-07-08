import mlx.core as mx
import numpy as np
from tqdm import tqdm

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.renderer.screen_gaussian import ScreenSpaceGaussians


def _compute_gaussian_batch_contribution(
    batch_means: mx.array,
    batch_conics: mx.array,
    batch_colors: mx.array,
    batch_opacities: mx.array,
    batch_extents: mx.array,
    tile_pixels: mx.array,
    tile_transmittance: mx.array,
) -> tuple[mx.array, mx.array]:
    """
    Compute the color contribution and updated transmittance for one batch of Gaussians over a tile.
    """
    # (G, P, 2) coordinate difference
    delta = tile_pixels[None, :, :] - batch_means[:, None, :]
    delta_x = delta[..., 0]
    delta_y = delta[..., 1]

    # Evaluate distance metric with inverse covariance matrix
    mahalanobis = (
        batch_conics[:, 0, None] * mx.square(delta_x)
        + 2.0 * batch_conics[:, 1, None] * delta_x * delta_y
        + batch_conics[:, 2, None] * mx.square(delta_y)
    )  # (G, P)

    # Perform early bounding-box rejection check
    within_extent = (mx.abs(delta_x) <= batch_extents[:, None]) & (mx.abs(delta_y) <= batch_extents[:, None])  # (G, P)

    # Scale opacities with Gaussian weights and clamp values
    alpha = mx.clip(mx.exp(-0.5 * mahalanobis) * batch_opacities[:, None], 0.0, 1.0)
    alpha = mx.where(within_extent, alpha, 0.0)

    one_minus_alpha = 1.0 - alpha  # (G, P)

    # Compute transmittance iteratively across depth layers
    if alpha.shape[0] > 1:
        cumprod_alphas = mx.cumprod(one_minus_alpha[:-1], axis=0)
        transmittance_before = mx.concatenate(
            [
                mx.ones((1, alpha.shape[1]), dtype=alpha.dtype),
                cumprod_alphas,
            ],
            axis=0,
        )  # (G, P)
    else:
        transmittance_before = mx.ones((1, alpha.shape[1]), dtype=alpha.dtype)  # (G, P)

    # Accumulate light contribution
    contribution = alpha * transmittance_before * tile_transmittance[None, :]  # (G, P)
    tile_image_delta = mx.transpose(contribution, (1, 0)) @ batch_colors  # (P, 3)
    new_tile_transmittance = tile_transmittance * mx.prod(one_minus_alpha, axis=0)  # (P,)

    return tile_image_delta, new_tile_transmittance


class Rasterizer:
    def __init__(
        self,
        gaussian_extent: float,
        tile_size: int | tuple[int, int],
        max_gaussians_per_batch: int,
    ) -> None:
        self.gaussian_extent = gaussian_extent
        self.tile_size = tile_size
        self.max_gaussians_per_batch = max_gaussians_per_batch

    def run(
        self,
        gaussians: ScreenSpaceGaussians,
        sorted_indices: mx.array,
        camera: Camera,
    ) -> mx.array:
        image_height = camera.h
        image_width = camera.w

        if len(sorted_indices) == 0:
            return mx.zeros((image_height, image_width, 3), dtype=mx.float32)

        # Initialize output canvas
        image = mx.zeros((image_height, image_width, 3), dtype=mx.float32)

        gaussians = gaussians[sorted_indices]

        if gaussians.opacities.ndim > 1:
            opacities = gaussians.opacities.squeeze(-1)

        a = gaussians.covariances_2d[:, 0, 0]
        b = gaussians.covariances_2d[:, 0, 1]
        c = gaussians.covariances_2d[:, 1, 1]
        determinant = mx.maximum(a * c - b * b, 1e-10)
        inverse_determinant = 1.0 / determinant
        conics = mx.stack(
            [
                c * inverse_determinant,
                -b * inverse_determinant,
                a * inverse_determinant,
            ],
            axis=1,
        )

        # Determine bounding extents
        gaussian_extent = self.gaussian_extent
        max_variance = mx.maximum(a, c) + mx.abs(b)
        max_extent = gaussian_extent * mx.sqrt(mx.maximum(max_variance, 1e-10))

        # Pixel bounds of the bounding box
        min_x = mx.floor(gaussians.means_2d[:, 0] - max_extent).astype(mx.int32)
        max_x = mx.ceil(gaussians.means_2d[:, 0] + max_extent).astype(mx.int32)
        min_y = mx.floor(gaussians.means_2d[:, 1] - max_extent).astype(mx.int32)
        max_y = mx.ceil(gaussians.means_2d[:, 1] + max_extent).astype(mx.int32)

        # Clamp calculations within screen dimensions
        min_x = mx.clip(min_x, 0, image_width)
        max_x = mx.clip(max_x, 0, image_width)
        min_y = mx.clip(min_y, 0, image_height)
        max_y = mx.clip(max_y, 0, image_height)

        # Discard Gaussians outside image boundaries
        valid_mask = (min_x < max_x) & (min_y < max_y)

        # Bridge via NumPy to resolve dynamic shape indices
        valid_indices = mx.array(np.where(np.array(valid_mask))[0])

        if valid_indices.shape[0] == 0:
            return image

        means_2d = gaussians.means_2d[valid_indices]
        conics = conics[valid_indices]
        colors = gaussians.colors[valid_indices]
        opacities = opacities[valid_indices]
        max_extent = max_extent[valid_indices]
        min_x = min_x[valid_indices]
        max_x = max_x[valid_indices]
        min_y = min_y[valid_indices]
        max_y = max_y[valid_indices]

        # Handle tile_size configuration parsing
        tile_size_cfg = self.tile_size
        if isinstance(tile_size_cfg, int):
            tx = ty = tile_size_cfg
        else:
            tx, ty = tile_size_cfg

        max_gaussians_per_batch = self.max_gaussians_per_batch

        # Flatten the 2D tile coordinates into a list for tqdm tracking
        tile_coords = [
            (tile_min_y, tile_min_x)
            for tile_min_y in range(0, image_height, ty)
            for tile_min_x in range(0, image_width, tx)
        ]

        active_tiles_count = 0
        total_overlapping_gaussians = 0

        # Loop over tiles using tqdm for real-time progress tracking
        for tile_min_y, tile_min_x in tqdm(tile_coords, desc="Splatting Tiles", leave=False):
            tile_max_y = min(tile_min_y + ty, image_height)
            tile_max_x = min(tile_min_x + tx, image_width)

            # Check overlaps with current tile boundary
            overlaps_tile = (min_x < tile_max_x) & (max_x > tile_min_x) & (min_y < tile_max_y) & (max_y > tile_min_y)

            overlap_indices = mx.array(np.where(np.array(overlaps_tile))[0])
            if overlap_indices.shape[0] == 0:
                continue

            active_tiles_count += 1
            total_overlapping_gaussians += overlap_indices.shape[0]

            tile_means = means_2d[overlap_indices]
            tile_conics = conics[overlap_indices]
            tile_colors = colors[overlap_indices]
            tile_opacities = opacities[overlap_indices]
            tile_extents = max_extent[overlap_indices]

            y_centers = mx.arange(tile_min_y, tile_max_y, dtype=mx.float32) + 0.5
            x_centers = mx.arange(tile_min_x, tile_max_x, dtype=mx.float32) + 0.5

            # Build spatial coordinate grid for evaluation
            tile_y_coords, tile_x_coords = mx.meshgrid(y_centers, x_centers, indexing="ij")
            tile_pixels = mx.stack([tile_x_coords, tile_y_coords], axis=-1).reshape(-1, 2)

            tile_image = mx.zeros((tile_pixels.shape[0], 3), dtype=mx.float32)
            tile_transmittance = mx.ones(tile_pixels.shape[0], dtype=mx.float32)

            num_tile_gaussians = tile_means.shape[0]
            for start_idx in range(0, num_tile_gaussians, max_gaussians_per_batch):
                end_idx = min(start_idx + max_gaussians_per_batch, num_tile_gaussians)

                batch_means = tile_means[start_idx:end_idx]
                batch_conics = tile_conics[start_idx:end_idx]
                batch_colors = tile_colors[start_idx:end_idx]
                batch_opacities = tile_opacities[start_idx:end_idx]
                batch_extents = tile_extents[start_idx:end_idx]

                tile_image_delta, tile_transmittance = _compute_gaussian_batch_contribution(
                    batch_means=batch_means,
                    batch_conics=batch_conics,
                    batch_colors=batch_colors,
                    batch_opacities=batch_opacities,
                    batch_extents=batch_extents,
                    tile_pixels=tile_pixels,
                    tile_transmittance=tile_transmittance,
                )

                tile_image = tile_image + tile_image_delta

            tile_h = tile_max_y - tile_min_y
            tile_w = tile_max_x - tile_min_x

            tile_patch = tile_image.reshape(tile_h, tile_w, 3)
            image[tile_min_y:tile_max_y, tile_min_x:tile_max_x] = tile_patch

        return image
