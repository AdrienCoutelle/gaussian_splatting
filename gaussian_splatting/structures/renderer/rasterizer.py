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

    def _compute_gaussian_tile_participation(
        self,
        gaussians: ScreenSpaceGaussians,
        image_width: int,
        image_height: int,
        tx: int,
        ty: int,
        num_tiles_x: int,
        num_tiles_y: int,
    ) -> tuple[mx.array, list[int], list[int], mx.array, mx.array]:
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

        max_variance = mx.maximum(a, c) + mx.abs(b)
        max_extent = self.gaussian_extent * mx.sqrt(mx.maximum(max_variance, 1e-10))

        min_x = mx.floor(gaussians.means_2d[:, 0] - max_extent).astype(mx.int32)
        max_x = mx.ceil(gaussians.means_2d[:, 0] + max_extent).astype(mx.int32)
        min_y = mx.floor(gaussians.means_2d[:, 1] - max_extent).astype(mx.int32)
        max_y = mx.ceil(gaussians.means_2d[:, 1] + max_extent).astype(mx.int32)

        min_x = mx.clip(min_x, 0, image_width)
        max_x = mx.clip(max_x, 0, image_width)
        min_y = mx.clip(min_y, 0, image_height)
        max_y = mx.clip(max_y, 0, image_height)

        # Guard band step to trivially reject Gaussians outside the frustum
        margin = 100.0
        guard_band_mask = (
            (gaussians.means_2d[:, 0] >= -margin)
            & (gaussians.means_2d[:, 0] <= image_width + margin)
            & (gaussians.means_2d[:, 1] >= -margin)
            & (gaussians.means_2d[:, 1] <= image_height + margin)
            & (gaussians.depths > 0.05)
        )
        valid_bbox = (min_x < max_x) & (min_y < max_y)
        valid_mask = guard_band_mask & valid_bbox

        valid_indices = mx.array(np.where(np.array(valid_mask))[0])
        if valid_indices.shape[0] == 0:
            empty = mx.array([], dtype=mx.int32)
            return empty, [], [], conics, max_extent

        min_x_tile = mx.clip(min_x[valid_indices] // tx, 0, num_tiles_x)
        max_x_tile = mx.clip((max_x[valid_indices] + tx - 1) // tx, 0, num_tiles_x)
        min_y_tile = mx.clip(min_y[valid_indices] // ty, 0, num_tiles_y)
        max_y_tile = mx.clip((max_y[valid_indices] + ty - 1) // ty, 0, num_tiles_y)

        num_instances_per_gaussian = (max_x_tile - min_x_tile) * (max_y_tile - min_y_tile)

        valid_indices_np = np.array(valid_indices)
        num_instances_np = np.array(num_instances_per_gaussian)
        min_x_tile_np = np.array(min_x_tile)
        min_y_tile_np = np.array(min_y_tile)
        max_x_tile_np = np.array(max_x_tile)

        instance_gaussian_indices_np = np.repeat(valid_indices_np, num_instances_np)
        total_instances = len(instance_gaussian_indices_np)

        if total_instances == 0:
            empty = mx.array([], dtype=mx.int32)
            return empty, [], [], conics, max_extent

        # Calculate coordinate offsets of overlapping tiles for each instance
        offsets_np = np.cumsum(num_instances_np)
        starts_np = offsets_np - num_instances_np
        repeated_starts_np = np.repeat(starts_np, num_instances_np)
        global_indices_np = np.arange(total_instances)
        local_idx_np = global_indices_np - repeated_starts_np

        repeated_min_x_tile_np = np.repeat(min_x_tile_np, num_instances_np)
        repeated_max_x_tile_np = np.repeat(max_x_tile_np, num_instances_np)
        repeated_min_y_tile_np = np.repeat(min_y_tile_np, num_instances_np)

        bbox_width_np = repeated_max_x_tile_np - repeated_min_x_tile_np
        local_y_np = local_idx_np // bbox_width_np
        local_x_np = local_idx_np % bbox_width_np

        tile_x_np = repeated_min_x_tile_np + local_x_np
        tile_y_np = repeated_min_y_tile_np + local_y_np
        tile_ids_np = tile_y_np * num_tiles_x + tile_x_np

        # Convert mappings back to MLX arrays
        instance_gaussian_indices = mx.array(instance_gaussian_indices_np)
        tile_ids = mx.array(tile_ids_np)

        # Sort indices combining tile ID (high bits) and view-space depth (low bits)
        instance_depths = gaussians.depths[instance_gaussian_indices]
        max_depth = mx.max(instance_depths) + 1.0
        sort_keys = tile_ids.astype(mx.float32) * max_depth + instance_depths

        sorted_instance_indices = mx.argsort(sort_keys)
        sorted_gaussian_indices = instance_gaussian_indices[sorted_instance_indices]
        sorted_tile_ids = tile_ids[sorted_instance_indices]

        # Retrieve boundary ranges per tile
        sorted_tile_ids_np = np.array(sorted_tile_ids)
        all_tile_ids_np = np.arange(num_tiles_x * num_tiles_y)
        tile_starts_np = np.searchsorted(sorted_tile_ids_np, all_tile_ids_np, side="left")
        tile_ends_np = np.searchsorted(sorted_tile_ids_np, all_tile_ids_np, side="right")

        # Convert to native Python lists to prevent CPU-GPU stalls in the rendering loop
        tile_starts_list = tile_starts_np.tolist()
        tile_ends_list = tile_ends_np.tolist()

        return sorted_gaussian_indices, tile_starts_list, tile_ends_list, conics, max_extent

    def _compute_gaussian_batch_contribution(
        self,
        batch_means: mx.array,
        batch_conics: mx.array,
        batch_colors: mx.array,
        batch_opacities: mx.array,
        batch_extents: mx.array,
        tile_pixels: mx.array,
        tile_transmittance: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """
        Computes the color contribution and updated transmittance for one batch of Gaussians.
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
        within_extent = (mx.abs(delta_x) <= batch_extents[:, None]) & (
            mx.abs(delta_y) <= batch_extents[:, None]
        )  # (G, P)

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

    def _render_single_tile(
        self,
        tile: Tile,
        means_2d: mx.array,
        conics: mx.array,
        colors: mx.array,
        opacities: mx.array,
        max_extent: mx.array,
    ) -> mx.array:
        """
        Renders the sorted, overlapping Gaussians stored inside the Tile object.
        """
        if tile.gaussian_indices is None or tile.gaussian_indices.shape[0] == 0:
            return mx.zeros((tile.height, tile.width, 3), dtype=mx.float32)

        tile_means = means_2d[tile.gaussian_indices]
        tile_conics = conics[tile.gaussian_indices]
        tile_colors = colors[tile.gaussian_indices]
        tile_opacities = opacities[tile.gaussian_indices]
        tile_extents = max_extent[tile.gaussian_indices]

        # Generate local pixel coordinates inside the tile
        tile_pixels = tile.generate_pixel_grid()

        # Initialize local accumulation canvas and light transmittance
        tile_image = mx.zeros((tile_pixels.shape[0], 3), dtype=mx.float32)
        tile_transmittance = mx.ones(tile_pixels.shape[0], dtype=mx.float32)

        # Draw Gaussians in batched chunks
        num_tile_gaussians = tile_means.shape[0]
        for start_idx in range(0, num_tile_gaussians, self.max_gaussians_per_batch):
            end_idx = min(start_idx + self.max_gaussians_per_batch, num_tile_gaussians)

            batch_means = tile_means[start_idx:end_idx]
            batch_conics = tile_conics[start_idx:end_idx]
            batch_colors = tile_colors[start_idx:end_idx]
            batch_opacities = tile_opacities[start_idx:end_idx]
            batch_extents = tile_extents[start_idx:end_idx]

            tile_image_delta, tile_transmittance = self._compute_gaussian_batch_contribution(
                batch_means=batch_means,
                batch_conics=batch_conics,
                batch_colors=batch_colors,
                batch_opacities=batch_opacities,
                batch_extents=batch_extents,
                tile_pixels=tile_pixels,
                tile_transmittance=tile_transmittance,
            )

            tile_image = tile_image + tile_image_delta

            # Early saturation stopping
            if mx.all(tile_transmittance < 1e-4):
                break

        return tile_image.reshape(tile.height, tile.width, 3)

    def run(
        self,
        gaussians: ScreenSpaceGaussians,
        camera: Camera,
    ) -> mx.array:
        image_height = camera.h
        image_width = camera.w

        sorted_indices = mx.argsort(gaussians.depths)

        if len(sorted_indices) == 0:
            return mx.zeros((image_height, image_width, 3), dtype=mx.float32)

        # Align with pre-sorted representation
        sorted_gaussians = gaussians[sorted_indices]

        # Resolve tile configuration parameters
        tile_size_cfg = self.tile_size
        tx, ty = (tile_size_cfg, tile_size_cfg) if isinstance(tile_size_cfg, int) else tile_size_cfg
        num_tiles_x = (image_width + tx - 1) // tx
        num_tiles_y = (image_height + ty - 1) // ty

        # Determine tile overlap participation, culling, and keys sorting
        (
            sorted_gaussian_indices,
            tile_starts_list,
            tile_ends_list,
            conics,
            max_extent,
        ) = self._compute_gaussian_tile_participation(
            sorted_gaussians, image_width, image_height, tx, ty, num_tiles_x, num_tiles_y
        )

        if sorted_gaussian_indices.shape[0] == 0:
            return mx.zeros((image_height, image_width, 3), dtype=mx.float32)

        # Allocate image canvas
        image = mx.zeros((image_height, image_width, 3), dtype=mx.float32)

        # Iterate and render active tiles
        for tile_id in tqdm(range(num_tiles_x * num_tiles_y), desc="Splatting Tiles", leave=False):
            # Access native Python lists directly to avoid lazy evaluation stalls
            start_idx = tile_starts_list[tile_id]
            end_idx = tile_ends_list[tile_id]

            if start_idx >= end_idx:
                continue

            tile_y_idx = tile_id // num_tiles_x
            tile_x_idx = tile_id % num_tiles_x

            tile_min_y = tile_y_idx * ty
            tile_max_y = min(tile_min_y + ty, image_height)
            tile_min_x = tile_x_idx * tx
            tile_max_x = min(tile_min_x + tx, image_width)

            tile = Tile(
                min_x=tile_min_x,
                min_y=tile_min_y,
                max_x=tile_max_x,
                max_y=tile_max_y,
            )
            # Store subset of Gaussian indices overlapping this tile
            tile.gaussian_indices = sorted_gaussian_indices[start_idx:end_idx]

            tile_patch = self._render_single_tile(
                tile=tile,
                means_2d=sorted_gaussians.means_2d,
                conics=conics,
                colors=sorted_gaussians.colors,
                opacities=sorted_gaussians.opacities.squeeze(-1),
                max_extent=max_extent,
            )

            image[tile_min_y:tile_max_y, tile_min_x:tile_max_x] = tile_patch

        return image
