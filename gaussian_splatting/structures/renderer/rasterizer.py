import mlx.core as mx
import numpy as np
from tqdm import tqdm

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

        image = mx.zeros((camera.h, camera.w, 3), dtype=mx.float32)

        for tile_id in tqdm(range(len(tiles)), desc="Rendering tiles", leave=False):
            tile = tiles[tile_id]

            if (
                tile.gaussian_indices is None
                or tile.gaussian_indices.shape[0] == 0
            ):  # fmt:skip
                continue

            tile_patch = self._render_single_tile(
                tile=tile,
                gaussians=sorted_gaussians,
            )

            image[tile.min_y : tile.max_y, tile.min_x : tile.max_x] = tile_patch

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

    def _compute_tile_participation(
        self,
        sorted_gaussians: ScreenSpaceGaussians,
        tiles: list[Tile],
    ) -> None:
        means_2d = sorted_gaussians.means_2d

        radii = sorted_gaussians.compute_max_extent(self.gaussian_extent)

        g_min_x = means_2d[:, 0] - radii
        g_max_x = means_2d[:, 0] + radii
        g_min_y = means_2d[:, 1] - radii
        g_max_y = means_2d[:, 1] + radii

        for tile in tiles:
            overlap = (g_min_x < tile.max_x) & (g_max_x > tile.min_x) & (g_min_y < tile.max_y) & (g_max_y > tile.min_y)

            indices = np.nonzero(overlap)[0]
            tile.gaussian_indices = mx.array(indices, dtype=mx.int32)

    def _render_single_tile(
        self,
        tile: Tile,
        gaussians: ScreenSpaceGaussians,
    ) -> mx.array:
        if tile.gaussian_indices is None or tile.gaussian_indices.shape[0] == 0:
            return mx.zeros((tile.height, tile.width, 3), dtype=mx.float32)

        # Slice the dataclass structure to retain only the Gaussians assigned to this tile
        tile_gaussians = gaussians[tile.gaussian_indices]

        # Extract local properties (evaluation of computed properties happens on the subset)
        tile_means = tile_gaussians.means_2d
        tile_conics = tile_gaussians.conics
        tile_colors = tile_gaussians.colors
        tile_opacities = tile_gaussians.opacities
        tile_extents = tile_gaussians.compute_max_extent(self.gaussian_extent)

        # Generate local pixel coordinates inside the tile
        tile_pixels = tile.generate_pixel_grid()
        P = tile_pixels.shape[0]

        # Initialize local accumulation canvas and light transmittance
        tile_image = mx.zeros((P, 3), dtype=mx.float32)
        tile_transmittance = mx.ones(P, dtype=mx.float32)

        # Draw Gaussians in batched chunks
        num_tile_gaussians = tile_means.shape[0]
        for start_idx in range(0, num_tile_gaussians, self.max_gaussians_per_batch):
            end_idx = min(start_idx + self.max_gaussians_per_batch, num_tile_gaussians)

            # Defensive reshaping to guarantee 1D and 2D arrays
            batch_means = tile_means[start_idx:end_idx].reshape(-1, 2)
            batch_conics = tile_conics[start_idx:end_idx].reshape(-1, 3)
            batch_colors = tile_colors[start_idx:end_idx].reshape(-1, 3)
            batch_opacities = tile_opacities[start_idx:end_idx].reshape(-1)
            batch_extents = tile_extents[start_idx:end_idx].reshape(-1)

            G = batch_means.shape[0]

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
            ones = mx.ones((1, P), dtype=mx.float32)
            if G > 1:
                padded = mx.concatenate([ones, 1.0 - alpha[:-1]], axis=0)
            else:
                padded = ones

            T_local = mx.cumprod(padded, axis=0)  # [G, P]

            # Global absolute transmittance at each step
            T_global = tile_transmittance[None, :] * T_local  # [G, P]

            # 7. Compute color contributions
            weights = T_global * alpha  # [G, P]
            colors = batch_colors[:, None, :]  # [G, 1, 3]

            # Sum up RGB values weighted by global alpha and accumulate
            tile_image_delta = mx.sum(weights[:, :, None] * colors, axis=0)  # [P, 3]
            tile_image = tile_image + tile_image_delta

            # 8. Update global transmittance for the next batch loop
            tile_transmittance = tile_transmittance * mx.prod(1.0 - alpha, axis=0)

            # Early saturation stopping
            if mx.all(tile_transmittance < 1e-4):
                break

        return tile_image.reshape(tile.height, tile.width, 3)
