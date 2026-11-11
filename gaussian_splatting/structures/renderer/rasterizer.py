import mlx.core as mx
from pydantic import BaseModel, ConfigDict

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


class RasterizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gaussian_extent: float = 3.0
    tile_size: int = 16


@profile
class Rasterizer:
    def __init__(
        self,
        config: RasterizerConfig,
    ) -> None:
        self.gaussian_extent = config.gaussian_extent
        self.tile_size = config.tile_size

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

        # Assign each tile all of its overlapping Gaussians, in depth order
        tile_indices, tile_offsets = self._bin_gaussians_to_tiles(
            means=mx.stop_gradient(means),
            radii=mx.stop_gradient(padded_extents[:-1]),
            camera=camera,
        )

        pixels = Tile(min_x=0, min_y=0, max_x=camera.w, max_y=camera.h).generate_pixel_grid()
        num_tiles_x = (camera.w + self.tile_size - 1) // self.tile_size
        pixel_tile_indices = (
            (pixels[:, 1].astype(mx.int32) // self.tile_size) * num_tiles_x
            + pixels[:, 0].astype(mx.int32) // self.tile_size
        ).astype(mx.float32)

        image = rasterize_image(
            pixels,
            padded_means,
            padded_conics,
            padded_colors,
            padded_opacities.reshape(-1),
            padded_extents,
            tile_indices,
            tile_offsets,
            pixel_tile_indices,
        )
        return image.reshape(camera.h, camera.w, 3)

    def _get_sorted_gaussians(
        self,
        gaussians: ScreenSpaceGaussians,
    ) -> ScreenSpaceGaussians:
        sorted_indices = mx.argsort(gaussians.depths)
        return gaussians[sorted_indices]

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

    def _bin_gaussians_to_tiles(
        self,
        means: mx.array,
        radii: mx.array,
        camera: Camera,
    ) -> tuple[mx.array, mx.array]:
        tile_size = self.tile_size
        num_tiles_x = (camera.w + tile_size - 1) // tile_size
        num_tiles_y = (camera.h + tile_size - 1) // tile_size
        num_tiles = num_tiles_x * num_tiles_y
        num_gaussians = means.shape[0]
        dummy_entry = mx.array([num_gaussians], dtype=mx.float32)

        # Tile range [x0, x1) x [y0, y1) overlapped by each bounding square
        x0 = mx.clip(mx.floor((means[:, 0] - radii) / tile_size), 0, num_tiles_x).astype(mx.int32)
        x1 = mx.clip(mx.ceil((means[:, 0] + radii) / tile_size), 0, num_tiles_x).astype(mx.int32)
        y0 = mx.clip(mx.floor((means[:, 1] - radii) / tile_size), 0, num_tiles_y).astype(mx.int32)
        y1 = mx.clip(mx.ceil((means[:, 1] + radii) / tile_size), 0, num_tiles_y).astype(mx.int32)
        widths = mx.maximum(x1 - x0, 0)
        counts = widths * mx.maximum(y1 - y0, 0)

        ends = mx.cumsum(counts)
        starts = ends - counts
        num_entries = int(ends[-1].item()) if num_gaussians > 0 else 0
        if num_entries == 0:
            return dummy_entry, mx.zeros((num_tiles + 1,), dtype=mx.float32)
        if num_entries >= 2**24:
            # The kernels read indices / offsets as float32, which is exact only up to 2^24
            raise ValueError(f"Too many Gaussian-tile pairs for float32 indexing: {num_entries}")

        # Entry -> Gaussian: mark each Gaussian's first entry with its index, then propagate with a running max
        gaussian_ids = mx.arange(num_gaussians, dtype=mx.int32)
        has_entries = counts > 0
        first_entry_marks = mx.full((num_entries + 1,), -1, dtype=mx.int32)
        first_entry_marks = first_entry_marks.at[mx.where(has_entries, starts, num_entries)].maximum(
            mx.where(has_entries, gaussian_ids, -1)
        )
        entry_gaussian = mx.cummax(first_entry_marks[:num_entries])

        local_index = mx.arange(num_entries, dtype=mx.int32) - starts[entry_gaussian]
        entry_width = widths[entry_gaussian]
        tile_x = x0[entry_gaussian] + local_index % entry_width
        tile_y = y0[entry_gaussian] + local_index // entry_width
        entry_tile = tile_y * num_tiles_x + tile_x

        sort_keys = entry_tile.astype(mx.int64) * num_gaussians + entry_gaussian.astype(mx.int64)
        tile_indices = entry_gaussian[mx.argsort(sort_keys)]

        entries_per_tile = mx.zeros((num_tiles,), dtype=mx.int32).at[entry_tile].add(1)
        tile_offsets = mx.concatenate([mx.zeros((1,), dtype=mx.int32), mx.cumsum(entries_per_tile)])

        return (
            mx.concatenate([tile_indices.astype(mx.float32), dummy_entry]),
            tile_offsets.astype(mx.float32),
        )
