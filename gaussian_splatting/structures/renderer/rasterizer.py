import mlx.core as mx
import numpy as np

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.renderer.metal_rasterizer import rasterize_image
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
        max_gaussians_per_tile: int = 1024,
    ) -> None:
        self.gaussian_extent = gaussian_extent
        self.tile_size = tile_size
        self.max_gaussians_per_batch = max_gaussians_per_batch
        self.max_gaussians_per_tile = max_gaussians_per_tile

    def run(
        self,
        gaussians: ScreenSpaceGaussians,
        camera: Camera,
    ) -> mx.array:

        sorted_gaussians = self._get_sorted_gaussians(gaussians)

        # Retrieve arrays
        means = sorted_gaussians.means_2d
        colors = sorted_gaussians.colors
        opacities = sorted_gaussians.opacities
        covs = sorted_gaussians.covariances_2d

        # Align dummy dimensions dynamically with source dimensions to prevent ValueError
        dummy_mean = mx.zeros((1, 2), dtype=mx.float32) - 1e5
        dummy_color = mx.zeros((1, 3), dtype=mx.float32)

        if opacities.ndim == 2:
            dummy_opacity = mx.zeros((1, 1), dtype=mx.float32)
        else:
            dummy_opacity = mx.zeros((1,), dtype=mx.float32)

        if covs.ndim == 3:
            dummy_cov = mx.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=mx.float32)
        else:
            dummy_cov = mx.array([[1.0, 0.0, 1.0]], dtype=mx.float32)

        # Safe concatenation along the Gaussian axis (axis=0)
        padded_means = mx.concatenate([means, dummy_mean], axis=0)
        padded_colors = mx.concatenate([colors, dummy_color], axis=0)
        padded_opacities = mx.concatenate([opacities, dummy_opacity], axis=0)
        padded_covs = mx.concatenate([covs, dummy_cov], axis=0)

        # Compute conics and extents globally to minimize redundant calculations per-tile
        padded_conics, padded_extents = self._compute_conics_and_extents(
            padded_covs,
            self.gaussian_extent,
        )

        tiles = self._create_tiles(camera)

        # Assign indices padded to self.max_gaussians_per_tile
        self._compute_tile_participation(
            sorted_gaussians,
            tiles=tiles,
        )

        pixels = Tile(min_x=0, min_y=0, max_x=camera.w, max_y=camera.h).generate_pixel_grid()
        num_tiles_x = (camera.w + self.tile_size - 1) // self.tile_size
        pixel_tile_indices = (
            (pixels[:, 1].astype(mx.int32) // self.tile_size) * num_tiles_x
            + pixels[:, 0].astype(mx.int32) // self.tile_size
        ).astype(mx.float32)
        tile_indices = mx.stack([tile.gaussian_indices for tile in tiles]).astype(mx.float32)

        image = rasterize_image(
            pixels,
            padded_means,
            padded_conics,
            padded_colors,
            padded_opacities.reshape(-1),
            padded_extents,
            tile_indices,
            pixel_tile_indices,
        )
        return image.reshape(camera.h, camera.w, 3)

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
        if covariances_2d.ndim == 3:
            a = covariances_2d[..., 0, 0]
            b = covariances_2d[..., 0, 1]
            c = covariances_2d[..., 1, 1]
        else:
            a = covariances_2d[..., 0]
            b = covariances_2d[..., 1]
            c = covariances_2d[..., 2]

        # Inverse covariance calculations
        det = a * c - b * b
        eps = 1e-6
        det = mx.where(det < eps, eps, det)
        inv_det = 1.0 / det
        conics = mx.stack([c * inv_det, -b * inv_det, a * inv_det], axis=-1)

        # Safe eigenvalues derivation
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
        dummy_idx = len(sorted_gaussians)  # Index pointing to our appended transparent dummy Gaussian

        for i, tile in enumerate(tiles):
            indices = np.flatnonzero(np_overlap[i])
            if len(indices) > self.max_gaussians_per_tile:
                indices = indices[: self.max_gaussians_per_tile]

            padding_needed = self.max_gaussians_per_tile - len(indices)
            if padding_needed > 0:
                padded_indices = np.concatenate([indices, np.full(padding_needed, dummy_idx, dtype=np.int32)])
            else:
                padded_indices = indices

            tile.gaussian_indices = mx.array(padded_indices, dtype=mx.int32)
