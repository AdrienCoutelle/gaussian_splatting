import mlx.core as mx
import numpy as np

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.renderer.screen_gaussian import ScreenSpaceGaussians
from gaussian_splatting.utils.profiler import profile


class Tile:
    def __init__(
        self,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
    ) -> None:
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

        self.width = max_x - min_x
        self.height = max_y - min_y

        self.gaussian_indices: mx.array | None = None

    def generate_pixel_grid(self) -> mx.array:
        y_centers = mx.arange(self.min_y, self.max_y, dtype=mx.float32) + 0.5
        x_centers = mx.arange(self.min_x, self.max_x, dtype=mx.float32) + 0.5

        tile_y_coords, tile_x_coords = mx.meshgrid(y_centers, x_centers, indexing="ij")
        return mx.stack([tile_x_coords, tile_y_coords], axis=-1).reshape(-1, 2)


@profile
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
        camera: Camera,
    ) -> mx.array:

        sorted_gaussians = self._get_sorted_gaussians(gaussians)

        tiles = self._create_tiles(camera)

        self._compute_tile_participation(
            sorted_gaussians,
            tiles=tiles,
        )

        num_tiles_x = (camera.w + self.tile_size - 1) // self.tile_size
        num_tiles_y = (camera.h + self.tile_size - 1) // self.tile_size

        patches_grid = [[None for _ in range(num_tiles_x)] for _ in range(num_tiles_y)]

        # Collect rendered patches into a grid
        for tile_y_idx in range(num_tiles_y):
            for tile_x_idx in range(num_tiles_x):
                tile_idx = tile_y_idx * num_tiles_x + tile_x_idx
                tile = tiles[tile_idx]

                tile_patch = self._render_single_tile(
                    tile=tile,
                    gaussians=sorted_gaussians,
                )
                patches_grid[tile_y_idx][tile_x_idx] = tile_patch

        # Concatenate horizontally across columns, then vertically across rows
        rows = [mx.concatenate(row_patches, axis=1) for row_patches in patches_grid]
        image = mx.concatenate(rows, axis=0)

        return image

    def _get_sorted_gaussians(
        self,
        gaussians: ScreenSpaceGaussians,
    ) -> ScreenSpaceGaussians:
        sorted_indices = mx.argsort(gaussians.depths)
        return gaussians[sorted_indices]

    def _create_tiles(
        self,
        camera: Camera,
    ) -> list[Tile]:
        image_height = camera.h
        image_width = camera.w

        num_tiles_x = (image_width + self.tile_size - 1) // self.tile_size
        num_tiles_y = (image_height + self.tile_size - 1) // self.tile_size

        tiles: list[Tile] = []
        for tile_y_idx in range(num_tiles_y):
            for tile_x_idx in range(num_tiles_x):
                tile_min_y = tile_y_idx * self.tile_size
                tile_max_y = min(tile_min_y + self.tile_size, image_height)
                tile_min_x = tile_x_idx * self.tile_size
                tile_max_x = min(tile_min_x + self.tile_size, image_width)

                tiles.append(
                    Tile(
                        min_x=tile_min_x,
                        min_y=tile_min_y,
                        max_x=tile_max_x,
                        max_y=tile_max_y,
                    )
                )

        return tiles

    def _compute_conics_and_extents(
        self,
        covariances_2d: mx.array,
        extent_scale: float,
    ) -> tuple[mx.array, mx.array]:
        """
        Differentiable calculation of 2D Gaussian conics and extents.
        Supports both [N, 2, 2] and [N, 3] shapes of covariances.
        Includes small epsilons to prevent square root NaN gradients.
        """
        if covariances_2d.ndim == 3:
            a = covariances_2d[..., 0, 0]
            b = covariances_2d[..., 0, 1]
            c = covariances_2d[..., 1, 1]
        else:
            a = covariances_2d[..., 0]
            b = covariances_2d[..., 1]
            c = covariances_2d[..., 2]

        # 1. Compute inverse covariance elements (conics)
        det = a * c - b * b
        eps = 1e-6
        det = mx.where(det < eps, eps, det)
        inv_det = 1.0 / det
        conics = mx.stack([c * inv_det, -b * inv_det, a * inv_det], axis=-1)

        # 2. Compute stable extents (eigenvalues of 2D covariance)
        # Adding 1e-10 inside the square roots avoids NaN gradients during backpropagation.
        D = (a - c) ** 2 + 4.0 * b * b
        lambda_max = 0.5 * ((a + c) + mx.sqrt(D + 1e-10))
        extents = extent_scale * mx.sqrt(lambda_max + 1e-10)

        return conics, extents

    def _compute_tile_participation(
        self,
        sorted_gaussians: ScreenSpaceGaussians,
        tiles: list[Tile],
    ) -> None:
        means_2d = sorted_gaussians.means_2d

        _, radii = self._compute_conics_and_extents(
            sorted_gaussians.covariances_2d,
            self.gaussian_extent,
        )

        g_min_x = means_2d[:, 0] - radii
        g_max_x = means_2d[:, 0] + radii
        g_min_y = means_2d[:, 1] - radii
        g_max_y = means_2d[:, 1] + radii

        tile_min_x = mx.array([t.min_x for t in tiles], dtype=mx.float32)[:, None]
        tile_max_x = mx.array([t.max_x for t in tiles], dtype=mx.float32)[:, None]
        tile_min_y = mx.array([t.min_y for t in tiles], dtype=mx.float32)[:, None]
        tile_max_y = mx.array([t.max_y for t in tiles], dtype=mx.float32)[:, None]

        overlap = (
            (g_min_x[None, :] < tile_max_x)
            & (g_max_x[None, :] > tile_min_x)
            & (g_min_y[None, :] < tile_max_y)
            & (g_max_y[None, :] > tile_min_y)
        )

        np_overlap = np.array(overlap)

        for i, tile in enumerate(tiles):
            tile.gaussian_indices = mx.array(np.flatnonzero(np_overlap[i]), dtype=mx.int32)

    def _render_single_tile(
        self,
        tile: Tile,
        gaussians: ScreenSpaceGaussians,
    ) -> mx.array:
        # Checking shape is free because tile_indices are concrete arrays with known shapes on the host
        if tile.gaussian_indices is None or tile.gaussian_indices.shape[0] == 0:
            return mx.zeros((tile.height, tile.width, 3), dtype=mx.float32)

        tile_gaussians = gaussians[tile.gaussian_indices]

        tile_means = tile_gaussians.means_2d
        tile_colors = tile_gaussians.colors
        tile_opacities = tile_gaussians.opacities

        # Derive differentiable conics and extents stably inside the rasterizer
        tile_conics, tile_extents = self._compute_conics_and_extents(
            tile_gaussians.covariances_2d,
            self.gaussian_extent,
        )

        tile_pixels = tile.generate_pixel_grid()
        P = tile_pixels.shape[0]

        tile_image = mx.zeros((P, 3), dtype=mx.float32)
        tile_transmittance = mx.ones(P, dtype=mx.float32)

        num_tile_gaussians = tile_means.shape[0]
        for start_idx in range(0, num_tile_gaussians, self.max_gaussians_per_batch):
            end_idx = min(start_idx + self.max_gaussians_per_batch, num_tile_gaussians)

            batch_means = tile_means[start_idx:end_idx].reshape(-1, 2)
            batch_conics = tile_conics[start_idx:end_idx].reshape(-1, 3)
            batch_colors = tile_colors[start_idx:end_idx].reshape(-1, 3)
            batch_opacities = tile_opacities[start_idx:end_idx].reshape(-1)
            batch_extents = tile_extents[start_idx:end_idx].reshape(-1)

            tile_image_delta, tile_transmittance = self._render_batch_step(
                tile_pixels=tile_pixels,
                batch_means=batch_means,
                batch_conics=batch_conics,
                batch_colors=batch_colors,
                batch_opacities=batch_opacities,
                batch_extents=batch_extents,
                tile_transmittance=tile_transmittance,
            )
            tile_image = tile_image + tile_image_delta

        return tile_image.reshape(tile.height, tile.width, 3)

    def _render_batch_step(
        self,
        tile_pixels: mx.array,
        batch_means: mx.array,
        batch_conics: mx.array,
        batch_colors: mx.array,
        batch_opacities: mx.array,
        batch_extents: mx.array,
        tile_transmittance: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Compiled mathematical core step. Fuses offset computations, power exponent calculations,
        masking, and alpha-blending logic into a single compiled GPU operation.
        """
        # 1. Compute pixel-to-mean offsets
        d = tile_pixels[None, :, :] - batch_means[:, None, :]  # [G, P, 2]
        dx = d[:, :, 0]  # [G, P]
        dy = d[:, :, 1]  # [G, P]

        # 2. Distance squared
        dist_sq = dx**2 + dy**2  # [G, P]

        # 3. Compute power exponent using conics
        k_a = batch_conics[:, 0, None]  # [G, 1]
        k_b = batch_conics[:, 1, None]  # [G, 1]
        k_c = batch_conics[:, 2, None]  # [G, 1]

        power = -0.5 * (k_a * dx**2 + 2.0 * k_b * dx * dy + k_c * dy**2)  # [G, P]

        # 4. Filter pixels outside the extent / radius
        extent_sq = (batch_extents[:, None]) ** 2  # [G, 1]
        inside_mask = (dist_sq <= extent_sq) & (power <= 0.0)

        # 5. Compute alpha (pixel opacities)
        alpha = batch_opacities[:, None] * mx.exp(power)  # [G, P]
        alpha = mx.where(inside_mask, alpha, 0.0)

        # 6. Front-to-back alpha blending using cumulative products
        P = tile_pixels.shape[0]
        ones = mx.ones((1, P), dtype=mx.float32)

        # Concatenate ones to handle base transmittance; alpha[:-1] handles G=1 safely
        padded = mx.concatenate([ones, 1.0 - alpha[:-1]], axis=0)
        T_local = mx.cumprod(padded, axis=0)  # [G, P]

        # Global absolute transmittance at each step
        T_global = tile_transmittance[None, :] * T_local  # [G, P]

        # 7. Compute color contributions
        weights = T_global * alpha  # [G, P]
        colors = batch_colors[:, None, :]  # [G, 1, 3]

        # Sum up RGB values weighted by global alpha and accumulate
        tile_image_delta = mx.sum(weights[:, :, None] * colors, axis=0)  # [P, 3]

        # Update global transmittance for the next batch loop
        next_transmittance = tile_transmittance * mx.prod(1.0 - alpha, axis=0)

        return tile_image_delta, next_transmittance
