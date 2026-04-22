import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from gaussian_splatting.structures.renderers.base_renderer import BaseRenderer, ScreenSpaceGaussians
from gaussian_splatting.utils.profiler import profile


@dataclass
class AppleSiliconRendererParams:
    width: int
    height: int
    focal_length: float
    near_plane: float = 1e-4
    covariance_regularization: float = 0.3
    tile_size: tuple[int, int] = (16, 16)
    max_gaussians_per_tile: int = 4000
    sigma_cut: float = 12.0
    eps: float = 1e-3
    verbose: bool = False

    @classmethod
    def from_dict(
        cls,
        config_dict: dict,
    ) -> "AppleSiliconRendererParams":
        if not isinstance(config_dict, dict):
            raise ValueError(f"AppleSiliconRendererParams must be a dictionary, got '{type(config_dict).__name__}'.")

        mandatory_fields = {
            "width",
            "height",
            "focal_length",
        }
        if not set(config_dict.keys()).issuperset(mandatory_fields):
            missing_fields = mandatory_fields - set(config_dict.keys())
            raise ValueError(
                f"AppleSiliconRendererParams is missing the following mandatory fields: {', '.join(missing_fields)}, "
                f"got {', '.join(config_dict.keys())}."
            )

        return AppleSiliconRendererParams(
            width=config_dict["width"],
            height=config_dict["height"],
            focal_length=config_dict["focal_length"],
            near_plane=config_dict.get("near_plane", 1e-4),
            covariance_regularization=config_dict.get("covariance_regularization", 0.3),
            tile_size=tuple(config_dict.get("tile_size", (16, 16))),
            max_gaussians_per_tile=config_dict.get("max_gaussians_per_tile", 4000),
            sigma_cut=config_dict.get("sigma_cut", 12.0),
            eps=config_dict.get("eps", 1e-3),
            verbose=config_dict.get("verbose", False),
        )


@profile
class AppleSiliconRenderer(BaseRenderer):
    def _splat_gaussians_vectorized(
        self,
        image: torch.Tensor,
        gaussians: ScreenSpaceGaussians,
        sorted_indices: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> None:
        if len(gaussians.means_2d) == 0:
            return

        # Sort all gaussian data by depth
        means_2d = gaussians.means_2d[sorted_indices]
        covariances_2d = gaussians.covariances_2d[sorted_indices]
        colors = gaussians.colors[sorted_indices]
        opacities = gaussians.opacities[sorted_indices]
        depths = gaussians.depths[sorted_indices]

        # Compute conic (inverse of 2D covariance) stored as (q11, q12, q22)
        # For [[a, b], [b, c]]: inv = 1/det * [[c, -b], [-b, a]]
        a = covariances_2d[:, 0, 0]
        b = covariances_2d[:, 0, 1]
        c = covariances_2d[:, 1, 1]
        det = torch.clamp(a * c - b * b, min=1e-10)
        inv_det = 1.0 / det
        conic = torch.stack(
            [
                c * inv_det,
                -b * inv_det,
                a * inv_det,
            ],
            dim=1,
        )

        if opacities.dim() > 1:
            opacities = opacities.squeeze(-1)

        # Convert to MLX arrays
        projected = _ProjectedGaussians(
            xys=mx.array(means_2d.cpu().numpy().astype(np.float32)),
            conic=mx.array(conic.cpu().numpy().astype(np.float32)),
            opacity=mx.array(opacities.cpu().numpy().astype(np.float32)),
            color=mx.array(colors.cpu().numpy().astype(np.float32)),
            depths=mx.array(depths.cpu().numpy().astype(np.float32)),
        )

        camera = _Camera(
            width=image_width,
            height=image_height,
        )

        tile_bins = _TileBins.create(
            projected=projected,
            camera=camera,
            tile_size=self.config.tile_size,
        )

        rendered_image = _rasterize_gaussians_metal(
            projected=projected,
            tile_bins=tile_bins,
            camera=camera,
            tile_size=self.config.tile_size,
            max_gaussians_per_tile=self.config.max_gaussians_per_tile,
            verbose=self.config.verbose,
            sigma_cut=self.config.sigma_cut,
            eps=self.config.eps,
        )

        image_np = np.array(rendered_image)
        image.copy_(torch.from_numpy(image_np).to(self.device))


@dataclass
class _Camera:
    width: int
    height: int


@dataclass
class _ProjectedGaussians:
    xys: mx.array
    conic: mx.array
    opacity: mx.array
    color: mx.array
    depths: mx.array


@dataclass
class _TileBins:
    starts: list[int]
    ends: list[int]
    gauss_ids_sorted: list[int]

    @staticmethod
    @profile
    def create(
        projected: "_ProjectedGaussians",
        camera: _Camera,
        tile_size: tuple[int, int] = (16, 16),
    ) -> "_TileBins":
        tx, ty = tile_size
        tile_width = (camera.width + tx - 1) // tx
        tile_height = (camera.height + ty - 1) // ty
        num_tiles = tile_width * tile_height

        xys_np = np.array(projected.xys)
        depths_np = np.array(projected.depths)
        n = len(xys_np)

        starts = [-1] * num_tiles
        ends = [-1] * num_tiles
        gauss_ids_sorted = []

        if n > 0:
            tile_ids = []
            for i in range(n):
                x, y = xys_np[i]
                tx_idx = int(x // tx)
                ty_idx = int(y // ty)
                if 0 <= tx_idx < tile_width and 0 <= ty_idx < tile_height:
                    tile_id = ty_idx * tile_width + tx_idx
                    tile_ids.append((tile_id, depths_np[i], i))

            if tile_ids:
                tile_ids.sort(key=lambda x: (x[0], x[1]))
                gauss_ids_sorted = [x[2] for x in tile_ids]

                current_tile = tile_ids[0][0]
                starts[current_tile] = 0

                for idx, (tile_id, _, _) in enumerate(tile_ids):
                    if tile_id != current_tile:
                        ends[current_tile] = idx
                        current_tile = tile_id
                        starts[current_tile] = idx

                ends[current_tile] = len(tile_ids)

        return _TileBins(
            starts=starts,
            ends=ends,
            gauss_ids_sorted=gauss_ids_sorted,
        )


SPLAT_TILE_FWD_OPTIMIZED_SRC = r"""
#define CHUNK_SIZE 128

const uint P = tx * ty;
const uint gtid = thread_position_in_grid.x;
const uint tile_id = gtid / P;
const uint pid     = gtid % P;
const uint tid_flat = thread_index_in_threadgroup;
const uint tg_size = tx * ty;

if (pid >= P) return;

threadgroup float2 shared_xy[CHUNK_SIZE];
threadgroup float  shared_q11[CHUNK_SIZE];
threadgroup float  shared_q12[CHUNK_SIZE];
threadgroup float  shared_q22[CHUNK_SIZE];
threadgroup float  shared_opacity[CHUNK_SIZE];
threadgroup float  shared_col_r[CHUNK_SIZE];
threadgroup float  shared_col_g[CHUNK_SIZE];
threadgroup float  shared_col_b[CHUNK_SIZE];

const uint x0 = tile_origin[tile_id * 2];
const uint y0 = tile_origin[tile_id * 2 + 1];
const uint px = pid % tx;
const uint py = pid / tx;

const uint img_x = x0 + px;
const uint img_y = y0 + py;

if (img_x >= width || img_y >= height) {
    const uint out_idx = (tile_id * P + pid) * 3;
    out_rgb[out_idx + 0] = bg_color[0];
    out_rgb[out_idx + 1] = bg_color[1];
    out_rgb[out_idx + 2] = bg_color[2];
    return;
}

const float x = float(img_x) + 0.5f;
const float y = float(img_y) + 0.5f;

const uint gstart = tile_gstart[tile_id];
const uint gcount = tile_gcount[tile_id];

float T = 1.0f;
float3 C = float3(0.0f);
bool early_exit = false;

for (uint chunk_start = 0; chunk_start < gcount; chunk_start += CHUNK_SIZE) {
    const uint chunk_end = min(chunk_start + CHUNK_SIZE, gcount);
    const uint chunk_size = chunk_end - chunk_start;

    for (uint i = tid_flat; i < chunk_size; i += tg_size) {
        const uint idx = gstart + chunk_start + i;
        shared_xy[i] = float2(gauss_xy[idx * 2], gauss_xy[idx * 2 + 1]);
        shared_q11[i] = gauss_conic[idx * 3 + 0];
        shared_q12[i] = gauss_conic[idx * 3 + 1];
        shared_q22[i] = gauss_conic[idx * 3 + 2];
        shared_opacity[i] = gauss_opacity[idx];
        shared_col_r[i] = gauss_color[idx * 3 + 0];
        shared_col_g[i] = gauss_color[idx * 3 + 1];
        shared_col_b[i] = gauss_color[idx * 3 + 2];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!early_exit) {
        for (uint k = 0; k < chunk_size; ++k) {
            const float2 mu = shared_xy[k];
            const float q11 = shared_q11[k];
            const float q12 = shared_q12[k];
            const float q22 = shared_q22[k];
            const float op  = shared_opacity[k];
            const float3 col = float3(shared_col_r[k], shared_col_g[k], shared_col_b[k]);

            if (!isfinite(mu.x) || !isfinite(mu.y) || !isfinite(q11) || !isfinite(q12) || !isfinite(q22) || !isfinite(op)) {
                continue;
            }
            const float dx = x - mu.x;
            const float dy = y - mu.y;
            const float sigma = 0.5f * (q11*dx*dx + 2.0f*q12*dx*dy + q22*dy*dy);

            if (sigma > sigma_cut) continue;

            float a = max(0.0f, op) * exp(-sigma);
            a = min(0.999f, a);

            C += T * a * col;
            T *= (1.0f - a);

            if (T < eps || !isfinite(T) || !isfinite(C.x) || !isfinite(C.y) || !isfinite(C.z)) {
                early_exit = true;
                break;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

C.x += T * bg_color[0];
C.y += T * bg_color[1];
C.z += T * bg_color[2];

const uint out_idx = (tile_id * P + pid) * 3;
out_rgb[out_idx + 0] = C.x;
out_rgb[out_idx + 1] = C.y;
out_rgb[out_idx + 2] = C.z;
"""  # noqa: E501


def _build_selected_ids_fast(
    gauss_ids_sorted: np.ndarray,
    starts: np.ndarray,
    clamp_counts: np.ndarray,
    total: int,
) -> np.ndarray:
    selected_ids = np.empty(total, dtype=np.int32)
    num_tiles = len(starts)

    dst_starts = np.zeros(num_tiles, dtype=np.int64)
    if num_tiles > 0:
        dst_starts[1:] = np.cumsum(clamp_counts[:-1])

    for t in range(num_tiles):
        c = int(clamp_counts[t])
        if c <= 0:
            continue
        s = int(starts[t])
        d = int(dst_starts[t])
        selected_ids[d : d + c] = gauss_ids_sorted[s : s + c]

    return selected_ids


@profile
class _MetalRasterizer:
    """MLX Metal kernel rasterizer."""

    def __init__(self) -> None:
        self.kernel = mx.fast.metal_kernel(
            name="splat_tile_fwd_optimized",
            input_names=[
                "gauss_xy",
                "gauss_conic",
                "gauss_opacity",
                "gauss_color",
                "tile_origin",
                "tile_gstart",
                "tile_gcount",
                "tx",
                "ty",
                "width",
                "height",
                "sigma_cut",
                "eps",
                "bg_color",
            ],
            output_names=["out_rgb"],
            source=SPLAT_TILE_FWD_OPTIMIZED_SRC,
        )

    def prepare_tile_data(
        self,
        projected: _ProjectedGaussians,
        tile_bins: _TileBins,
        tile_width: int,
        tile_height: int,
        max_gaussians_per_tile: int = 4000,
        tx: int = 16,
        ty: int = 16,
    ) -> dict[str, mx.array]:
        num_tiles = tile_width * tile_height

        starts = np.array(tile_bins.starts, dtype=np.int64)
        ends = np.array(tile_bins.ends, dtype=np.int64)
        valid = starts >= 0
        counts = np.where(valid, ends - starts, 0)
        clamp_counts = np.minimum(counts, max_gaussians_per_tile).astype(np.int64)

        need_topk = counts > max_gaussians_per_tile
        num_need_topk = int(need_topk.sum())

        if num_need_topk == 0:
            total = int(clamp_counts.sum())

            if total > 0:
                tile_gstarts_np = np.zeros(num_tiles, dtype=np.int64)
                if num_tiles > 0:
                    tile_gstarts_np[1:] = np.cumsum(clamp_counts[:-1])

                gauss_ids_sorted_np = np.array(tile_bins.gauss_ids_sorted, dtype=np.int32)
                selected_ids = _build_selected_ids_fast(
                    gauss_ids_sorted_np,
                    starts,
                    clamp_counts,
                    total,
                )

                sel_mx = mx.array(selected_ids, dtype=mx.int32)
                gauss_xy = mx.take(projected.xys, sel_mx, axis=0)
                gauss_conic = mx.take(projected.conic, sel_mx, axis=0)
                gauss_opacity = mx.take(projected.opacity, sel_mx, axis=0)
                if gauss_opacity.ndim > 1:
                    gauss_opacity = gauss_opacity.squeeze(-1)
                gauss_color = mx.take(projected.color, sel_mx, axis=0)
            else:
                tile_gstarts_np = np.zeros(num_tiles, dtype=np.int64)
                gauss_xy = mx.zeros((1, 2), dtype=mx.float32)
                gauss_conic = mx.zeros((1, 3), dtype=mx.float32)
                gauss_opacity = mx.zeros((1,), dtype=mx.float32)
                gauss_color = mx.zeros((1, 3), dtype=mx.float32)
        else:
            tile_gstarts_np = np.zeros(num_tiles, dtype=np.int64)
            if num_tiles > 0:
                tile_gstarts_np[1:] = np.cumsum(clamp_counts[:-1])
            total = int(clamp_counts.sum())

            if total > 0:
                gauss_ids_sorted = np.array(tile_bins.gauss_ids_sorted, dtype=np.int32)
                selected_ids = np.empty(total, dtype=np.int32)
                write_ptr = 0

                for t in range(num_tiles):
                    c = int(clamp_counts[t])
                    if c <= 0:
                        continue
                    s = int(starts[t])
                    count_full = int(counts[t])

                    if count_full <= max_gaussians_per_tile:
                        selected_ids[write_ptr : write_ptr + c] = gauss_ids_sorted[s : s + c]
                    else:
                        tile_slice = gauss_ids_sorted[s : s + count_full]
                        tx_idx = t % tile_width
                        ty_idx = t // tile_width
                        center_x = float(tx_idx * tx + tx * 0.5)
                        center_y = float(ty_idx * ty + ty * 0.5)
                        tile_ids_mx = mx.array(tile_slice, dtype=mx.int32)
                        tile_xy_mx = mx.take(projected.xys, tile_ids_mx, axis=0)
                        tile_conic_mx = mx.take(projected.conic, tile_ids_mx, axis=0)
                        tile_opacity_mx = mx.take(projected.opacity, tile_ids_mx, axis=0)
                        tile_depths_mx = mx.take(projected.depths, tile_ids_mx, axis=0)
                        if tile_opacity_mx.ndim > 1:
                            tile_opacity_mx = tile_opacity_mx.squeeze(-1)
                        dx = mx.array(center_x, dtype=mx.float32) - tile_xy_mx[:, 0]
                        dy = mx.array(center_y, dtype=mx.float32) - tile_xy_mx[:, 1]
                        sigma = 0.5 * (
                            tile_conic_mx[:, 0] * dx * dx
                            + 2.0 * tile_conic_mx[:, 1] * dx * dy
                            + tile_conic_mx[:, 2] * dy * dy
                        )
                        sigma = mx.minimum(sigma, 12.0)
                        scores = mx.maximum(0.0, tile_opacity_mx) * mx.exp(-sigma)
                        order_desc = mx.argsort(-scores)
                        top_k_idx = order_desc[:c]

                        selected_depths = tile_depths_mx[top_k_idx]
                        depth_order = mx.argsort(selected_depths)
                        top_idx_sorted = top_k_idx[depth_order]

                        chosen_mx = mx.take(tile_ids_mx, top_idx_sorted, axis=0)
                        chosen = np.array(chosen_mx, dtype=np.int32)
                        selected_ids[write_ptr : write_ptr + c] = chosen

                    write_ptr += c

                sel_mx = mx.array(selected_ids, dtype=mx.int32)
                gauss_xy = mx.take(projected.xys, sel_mx, axis=0)
                gauss_conic = mx.take(projected.conic, sel_mx, axis=0)
                gauss_opacity = mx.take(projected.opacity, sel_mx, axis=0)
                if gauss_opacity.ndim > 1:
                    gauss_opacity = gauss_opacity.squeeze(-1)
                gauss_color = mx.take(projected.color, sel_mx, axis=0)
            else:
                gauss_xy = mx.zeros((1, 2), dtype=mx.float32)
                gauss_conic = mx.zeros((1, 3), dtype=mx.float32)
                gauss_opacity = mx.zeros((1,), dtype=mx.float32)
                gauss_color = mx.zeros((1, 3), dtype=mx.float32)

        tile_indices = np.arange(num_tiles, dtype=np.int32)
        tile_x_idx: np.ndarray = (tile_indices % tile_width).astype(np.int32)
        tile_y_idx: np.ndarray = (tile_indices // tile_width).astype(np.int32)
        tile_origins: np.ndarray = np.empty(num_tiles * 2, dtype=np.uint32)
        tile_origins[0::2] = (tile_x_idx * tx).astype(np.uint32)
        tile_origins[1::2] = (tile_y_idx * ty).astype(np.uint32)

        return {
            "gauss_xy": gauss_xy.astype(mx.float32),
            "gauss_conic": gauss_conic.astype(mx.float32),
            "gauss_opacity": gauss_opacity.astype(mx.float32),
            "gauss_color": gauss_color.astype(mx.float32),
            "tile_origin": mx.array(tile_origins, dtype=mx.uint32),
            "tile_gstart": mx.array(tile_gstarts_np, dtype=mx.uint32),
            "tile_gcount": mx.array(clamp_counts, dtype=mx.uint32),
        }

    def rasterize(
        self,
        tile_data: dict[str, mx.array],
        width: int,
        height: int,
        tile_size: tuple[int, int] = (16, 16),
        background: tuple[float, float, float] = (0.0, 0.0, 0.0),
        sigma_cut: float = 12.0,
        eps: float = 1e-3,
    ) -> mx.array:
        tx, ty = tile_size
        tile_width = (width + tx - 1) // tx
        tile_height = (height + ty - 1) // ty
        num_tiles = tile_width * tile_height
        P = tx * ty

        bg_color = mx.array(background, dtype=mx.float32)

        total_threads = num_tiles * P
        flat_output = self.kernel(
            inputs=[
                tile_data["gauss_xy"],
                tile_data["gauss_conic"],
                tile_data["gauss_opacity"],
                tile_data["gauss_color"],
                tile_data["tile_origin"],
                tile_data["tile_gstart"],
                tile_data["tile_gcount"],
                mx.array(tx, dtype=mx.uint32),
                mx.array(ty, dtype=mx.uint32),
                mx.array(width, dtype=mx.uint32),
                mx.array(height, dtype=mx.uint32),
                mx.array(sigma_cut, dtype=mx.float32),
                mx.array(eps, dtype=mx.float32),
                bg_color,
            ],
            output_shapes=[(num_tiles * P, 3)],
            output_dtypes=[mx.float32],
            grid=(int(total_threads), 1, 1),
            threadgroup=(min(P, 256), 1, 1),
        )[0]

        image = mx.full((height, width, 3), bg_color, dtype=mx.float32)

        for tile_id in range(num_tiles):
            ty_idx = tile_id // tile_width
            tx_idx = tile_id % tile_width
            y0 = ty_idx * ty
            x0 = tx_idx * tx
            y1 = min(y0 + ty, height)
            x1 = min(x0 + tx, width)

            tile_pixels = flat_output[tile_id * P : (tile_id + 1) * P]
            tile_image = tile_pixels.reshape(ty, tx, 3)

            tile_h = y1 - y0
            tile_w = x1 - x0
            image[y0:y1, x0:x1] = tile_image[:tile_h, :tile_w]

        return image


_metal_rasterizer_optimized = None


def _get_metal_rasterizer() -> _MetalRasterizer:
    global _metal_rasterizer_optimized

    if _metal_rasterizer_optimized is None:
        _metal_rasterizer_optimized = _MetalRasterizer()
    return _metal_rasterizer_optimized


@profile
def _rasterize_gaussians_metal(
    projected: _ProjectedGaussians,
    tile_bins: _TileBins,
    camera: _Camera,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tile_size: tuple[int, int] = (16, 16),
    max_gaussians_per_tile: int = 4000,
    verbose: bool = False,
    **kwargs: Any,
) -> mx.array:
    width, height = camera.width, camera.height
    tx, ty = tile_size
    tile_width = (width + tx - 1) // tx
    tile_height = (height + ty - 1) // ty

    if verbose:
        num_tiles = tile_width * tile_height
        print("Metal rasterization:")
        print(f"  Image: {width}x{height}")
        print(f"  Tiles: {tile_width}x{tile_height} = {num_tiles}")
        print(f"  Tile size: {tx}x{ty}")

    if int(projected.xys.shape[0]) == 0:
        if verbose:
            print("  Empty scene detected, returning background")
        bg_color = mx.array(background)
        return mx.full((height, width, 3), bg_color)

    rasterizer = _get_metal_rasterizer()

    t0 = time.perf_counter()
    tile_data = rasterizer.prepare_tile_data(
        projected,
        tile_bins,
        tile_width,
        tile_height,
        max_gaussians_per_tile,
        tx,
        ty,
    )
    t_prepare_ms = (time.perf_counter() - t0) * 1000.0

    if verbose:
        total_gaussians = int(tile_data["gauss_xy"].shape[0])
        print(f"  Total gaussians in tiles: {total_gaussians}")

    if int(tile_data["gauss_xy"].shape[0]) == 0:
        if verbose:
            print("  No gaussians in tiles, returning background")
        bg_color = mx.array(background)
        return mx.full((height, width, 3), bg_color)

    t1 = time.perf_counter()
    image = rasterizer.rasterize(
        tile_data,
        width,
        height,
        tile_size,
        background,
        sigma_cut=kwargs.get("sigma_cut", 12.0),
        eps=kwargs.get("eps", 1e-3),
    )
    t_kernel_ms = (time.perf_counter() - t1) * 1000.0

    if verbose:
        try:
            g_counts_np = np.array(tile_data["tile_gcount"])
            avg_g = float(g_counts_np.mean())
            max_g = int(g_counts_np.max())
        except Exception:
            avg_g = max_g = -1
        print(f"  Timing: prepare={t_prepare_ms:.2f} ms, kernel+compose={t_kernel_ms:.2f} ms")
        if avg_g >= 0:
            print(f"  G per tile: avg={avg_g:.1f}, max={max_g}")

    return mx.clip(image, 0.0, 1.0)


# """MLX-based Apple Silicon renderer for Gaussian Splatting using Python MLX ops."""

# from dataclasses import dataclass

# import mlx.core as mx
# import torch

# from gaussian_splatting.structures.renderers.base_renderer import BaseRenderer, ScreenSpaceGaussians
# from gaussian_splatting.utils.profiler import profile


# @dataclass
# class AppleSiliconRendererParams:
#     width: int
#     height: int
#     focal_length: float
#     near_plane: float = 1e-4
#     covariance_regularization: float = 0.3
#     tile_size: tuple[int, int] = (16, 16)
#     max_gaussians_per_tile: int = 4000
#     sigma_cut: float = 12.0
#     eps: float = 1e-3
#     verbose: bool = False

#     @classmethod
#     def from_dict(
#         cls,
#         config_dict: dict,
#     ) -> "AppleSiliconRendererParams":
#         if not isinstance(config_dict, dict):
#             raise ValueError(f"AppleSiliconRendererParams must be a dictionary, got '{type(config_dict).__name__}'.")

#         mandatory_fields = {
#             "width",
#             "height",
#             "focal_length",
#         }
#         if not set(config_dict.keys()).issuperset(mandatory_fields):
#             missing_fields = mandatory_fields - set(config_dict.keys())
#             raise ValueError(
#                 f"AppleSiliconRendererParams is missing the following mandatory fields: {', '.join(missing_fields)}, "
#                 f"got {', '.join(config_dict.keys())}."
#             )

#         return AppleSiliconRendererParams(
#             width=config_dict["width"],
#             height=config_dict["height"],
#             focal_length=config_dict["focal_length"],
#             near_plane=config_dict.get("near_plane", 1e-4),
#             covariance_regularization=config_dict.get("covariance_regularization", 0.3),
#             tile_size=tuple(config_dict.get("tile_size", (16, 16))),
#             max_gaussians_per_tile=config_dict.get("max_gaussians_per_tile", 4000),
#             sigma_cut=config_dict.get("sigma_cut", 12.0),
#             eps=config_dict.get("eps", 1e-3),
#             verbose=config_dict.get("verbose", False),
#         )


# class AppleSiliconRenderer(BaseRenderer):
#     @dataclass
#     class _Camera:
#         width: int
#         height: int

#     @dataclass
#     class _ProjectedGaussians:
#         xys: mx.array
#         conic: mx.array
#         opacity: mx.array
#         color: mx.array
#         depths: mx.array

#     @dataclass
#     class _TileBins:
#         starts: list[int]
#         ends: list[int]
#         gauss_ids_sorted: list[int]

#     @profile
#     def _splat_gaussians_vectorized(
#         self,
#         image: torch.Tensor,
#         gaussians: ScreenSpaceGaussians,
#         sorted_indices: torch.Tensor,
#         image_height: int,
#         image_width: int,
#     ) -> None:
#         if len(gaussians.means_2d) == 0:
#             return

#         projected = self._create_projected_gaussians(
#             gaussians=gaussians,
#             sorted_indices=sorted_indices,
#         )

#         camera = self._Camera(
#             width=image_width,
#             height=image_height,
#         )
#         tile_bins = self._create_tile_bins(
#             projected=projected,
#             camera=camera,
#         )

#         rendered_image = self._rasterize_gaussians_mlx(
#             projected=projected,
#             tile_bins=tile_bins,
#             camera=camera,
#             tile_size=self.config.tile_size,
#             max_gaussians_per_tile=self.config.max_gaussians_per_tile,
#             verbose=self.config.verbose,
#             sigma_cut=self.config.sigma_cut,
#             eps=self.config.eps,
#         )
#         mx.eval(rendered_image)

#         image.copy_(
#             torch.tensor(
#                 rendered_image.tolist(),
#                 device=self.device,
#                 dtype=torch.float32,
#             )
#         )

#     @profile
#     def _create_projected_gaussians(
#         self,
#         gaussians: ScreenSpaceGaussians,
#         sorted_indices: torch.Tensor,
#     ) -> "_ProjectedGaussians":
#         means_2d = gaussians.means_2d[sorted_indices]
#         covariances_2d = gaussians.covariances_2d[sorted_indices]
#         colors = gaussians.colors[sorted_indices]
#         opacities = gaussians.opacities[sorted_indices]
#         depths = gaussians.depths[sorted_indices]

#         a = covariances_2d[:, 0, 0] + self.config.covariance_regularization
#         b = covariances_2d[:, 0, 1]
#         c = covariances_2d[:, 1, 1] + self.config.covariance_regularization
#         det = torch.clamp(a * c - b * b, min=1e-10)
#         inv_det = 1.0 / det
#         conic = torch.stack(
#             [
#                 c * inv_det,
#                 -b * inv_det,
#                 a * inv_det,
#             ],
#             dim=1,
#         )

#         if opacities.dim() > 1:
#             opacities = opacities.squeeze(-1)

#         return self._ProjectedGaussians(
#             xys=mx.array(means_2d.detach().cpu().tolist(), dtype=mx.float32),
#             conic=mx.array(conic.detach().cpu().tolist(), dtype=mx.float32),
#             opacity=mx.array(opacities.detach().cpu().tolist(), dtype=mx.float32),
#             color=mx.array(colors.detach().cpu().tolist(), dtype=mx.float32),
#             depths=mx.array(depths.detach().cpu().tolist(), dtype=mx.float32),
#         )

#     @profile
#     def _create_tile_bins(
#         self,
#         projected: "_ProjectedGaussians",
#         camera: "_Camera",
#     ) -> "_TileBins":
#         tx, ty = self.config.tile_size
#         tile_width = (camera.width + tx - 1) // tx
#         tile_height = (camera.height + ty - 1) // ty
#         num_tiles = tile_width * tile_height

#         starts = [-1] * num_tiles
#         ends = [-1] * num_tiles
#         gauss_ids_sorted: list[int] = []

#         if int(projected.xys.shape[0]) == 0:
#             return self._TileBins(
#                 starts=starts,
#                 ends=ends,
#                 gauss_ids_sorted=gauss_ids_sorted,
#             )

#         tile_entries: list[tuple[int, float, int]] = []
#         xys = projected.xys.tolist()
#         depths = projected.depths.tolist()

#         for gaussian_id, ((x, y), depth) in enumerate(zip(xys, depths, strict=False)):
#             tile_x = int(x // tx)
#             tile_y = int(y // ty)
#             if 0 <= tile_x < tile_width and 0 <= tile_y < tile_height:
#                 tile_id = tile_y * tile_width + tile_x
#                 tile_entries.append((tile_id, float(depth), gaussian_id))

#         if not tile_entries:
#             return self._TileBins(
#                 starts=starts,
#                 ends=ends,
#                 gauss_ids_sorted=gauss_ids_sorted,
#             )

#         tile_entries.sort(key=lambda entry: (entry[0], entry[1]))
#         gauss_ids_sorted = [entry[2] for entry in tile_entries]

#         current_tile = tile_entries[0][0]
#         starts[current_tile] = 0

#         for index, (tile_id, _, _) in enumerate(tile_entries):
#             if tile_id != current_tile:
#                 ends[current_tile] = index
#                 current_tile = tile_id
#                 starts[current_tile] = index

#         ends[current_tile] = len(tile_entries)

#         return self._TileBins(
#             starts=starts,
#             ends=ends,
#             gauss_ids_sorted=gauss_ids_sorted,
#         )

#     def _select_tile_gaussian_ids(
#         self,
#         projected: "_ProjectedGaussians",
#         tile_bins: "_TileBins",
#         tile_id: int,
#         tile_width: int,
#     ) -> mx.array | None:
#         start = tile_bins.starts[tile_id]
#         end = tile_bins.ends[tile_id]

#         if start < 0 or end <= start:
#             return None

#         tile_ids = tile_bins.gauss_ids_sorted[start:end]
#         if len(tile_ids) <= self.config.max_gaussians_per_tile:
#             return mx.array(tile_ids, dtype=mx.int32)

#         tx, ty = self.config.tile_size
#         tile_x_idx = tile_id % tile_width
#         tile_y_idx = tile_id // tile_width
#         center_x = float(tile_x_idx * tx + tx * 0.5)
#         center_y = float(tile_y_idx * ty + ty * 0.5)

#         tile_ids_mx = mx.array(tile_ids, dtype=mx.int32)
#         tile_xy = mx.take(projected.xys, tile_ids_mx, axis=0)
#         tile_conic = mx.take(projected.conic, tile_ids_mx, axis=0)
#         tile_opacity = mx.take(projected.opacity, tile_ids_mx, axis=0)
#         tile_depths = mx.take(projected.depths, tile_ids_mx, axis=0)

#         if tile_opacity.ndim > 1:
#             tile_opacity = tile_opacity.squeeze(-1)

#         dx = mx.array(center_x, dtype=mx.float32) - tile_xy[:, 0]
#         dy = mx.array(center_y, dtype=mx.float32) - tile_xy[:, 1]
#         sigma = 0.5 * (tile_conic[:, 0] * dx * dx + 2.0 * tile_conic[:, 1] * dx * dy + tile_conic[:, 2] * dy * dy)
#         scores = mx.maximum(tile_opacity, 0.0) * mx.exp(-mx.minimum(sigma, self.config.sigma_cut))
#         top_k_indices = mx.argsort(-scores)[: self.config.max_gaussians_per_tile]
#         top_k_depths = tile_depths[top_k_indices]
#         depth_order = mx.argsort(top_k_depths)

#         return mx.take(tile_ids_mx, top_k_indices[depth_order], axis=0)

#     def _rasterize_tile(
#         self,
#         projected: "_ProjectedGaussians",
#         gaussian_ids: mx.array | None,
#         x0: int,
#         y0: int,
#         x1: int,
#         y1: int,
#         background: tuple[float, float, float],
#         sigma_cut: float,
#         eps: float,
#     ) -> mx.array:
#         tile_height = y1 - y0
#         tile_width = x1 - x0
#         background_color = mx.array(background, dtype=mx.float32)

#         if gaussian_ids is None or int(gaussian_ids.shape[0]) == 0:
#             return mx.full((tile_height, tile_width, 3), background_color, dtype=mx.float32)

#         tile_xy = mx.take(projected.xys, gaussian_ids, axis=0)
#         tile_conic = mx.take(projected.conic, gaussian_ids, axis=0)
#         tile_opacity = mx.take(projected.opacity, gaussian_ids, axis=0)
#         tile_color = mx.take(projected.color, gaussian_ids, axis=0)

#         if tile_opacity.ndim > 1:
#             tile_opacity = tile_opacity.squeeze(-1)

#         pixel_x = (mx.arange(x0, x1, dtype=mx.float32) + 0.5).reshape(1, tile_width, 1)
#         pixel_y = (mx.arange(y0, y1, dtype=mx.float32) + 0.5).reshape(tile_height, 1, 1)

#         dx = pixel_x - tile_xy[:, 0].reshape(1, 1, -1)
#         dy = pixel_y - tile_xy[:, 1].reshape(1, 1, -1)

#         sigma = 0.5 * (
#             tile_conic[:, 0].reshape(1, 1, -1) * dx * dx
#             + 2.0 * tile_conic[:, 1].reshape(1, 1, -1) * dx * dy
#             + tile_conic[:, 2].reshape(1, 1, -1) * dy * dy
#         )
#         alpha = mx.where(
#             sigma <= sigma_cut,
#             mx.minimum(
#                 mx.maximum(tile_opacity.reshape(1, 1, -1), 0.0) * mx.exp(-sigma),
#                 0.999,
#             ),
#             0.0,
#         )

#         one_minus_alpha = 1.0 - alpha
#         transmittance_prefix = mx.concatenate(
#             [
#                 mx.ones((tile_height, tile_width, 1), dtype=mx.float32),
#                 mx.cumprod(one_minus_alpha[..., :-1], axis=-1),
#             ],
#             axis=-1,
#         )

#         if eps > 0.0:
#             transmittance_prefix = mx.where(transmittance_prefix >= eps, transmittance_prefix, 0.0)

#         color = (transmittance_prefix[..., None] * alpha[..., None] * tile_color.reshape(1, 1, -1, 3)).sum(axis=2)
#         final_transmittance = mx.prod(one_minus_alpha, axis=-1, keepdims=True)

#         return color + final_transmittance * background_color.reshape(1, 1, 3)

#     @profile
#     def _rasterize_gaussians_mlx(
#         self,
#         projected: "_ProjectedGaussians",
#         tile_bins: "_TileBins",
#         camera: "_Camera",
#         background: tuple[float, float, float] = (0.0, 0.0, 0.0),
#         tile_size: tuple[int, int] = (16, 16),
#         max_gaussians_per_tile: int = 4000,
#         verbose: bool = False,
#         sigma_cut: float = 12.0,
#         eps: float = 1e-3,
#     ) -> mx.array:
#         width, height = camera.width, camera.height
#         tx, ty = tile_size
#         tiles_x = (width + tx - 1) // tx
#         tiles_y = (height + ty - 1) // ty

#         if verbose:
#             num_tiles = tiles_x * tiles_y
#             print("Python MLX rasterization:")
#             print(f"  Image: {width}x{height}")
#             print(f"  Tiles: {tiles_x}x{tiles_y} = {num_tiles}")
#             print(f"  Tile size: {tx}x{ty}")

#         if int(projected.xys.shape[0]) == 0:
#             background_color = mx.array(background, dtype=mx.float32)
#             return mx.full((height, width, 3), background_color, dtype=mx.float32)

#         image = mx.full(
#             (height, width, 3),
#             mx.array(background, dtype=mx.float32),
#             dtype=mx.float32,
#         )
#         gaussian_counts: list[int] = []

#         for tile_id in range(tiles_x * tiles_y):
#             gaussian_ids = self._select_tile_gaussian_ids(
#                 projected=projected,
#                 tile_bins=tile_bins,
#                 tile_id=tile_id,
#                 tile_width=tiles_x,
#             )
#             if gaussian_ids is None:
#                 continue

#             gaussian_counts.append(int(gaussian_ids.shape[0]))

#             tile_y_idx = tile_id // tiles_x
#             tile_x_idx = tile_id % tiles_x
#             y0 = tile_y_idx * ty
#             x0 = tile_x_idx * tx
#             y1 = min(y0 + ty, height)
#             x1 = min(x0 + tx, width)

#             image[y0:y1, x0:x1] = self._rasterize_tile(
#                 projected=projected,
#                 gaussian_ids=gaussian_ids,
#                 x0=x0,
#                 y0=y0,
#                 x1=x1,
#                 y1=y1,
#                 background=background,
#                 sigma_cut=sigma_cut,
#                 eps=eps,
#             )

#         if verbose and gaussian_counts:
#             avg_gaussians = sum(gaussian_counts) / len(gaussian_counts)
#             print(f"  Total gaussians in tiles: {sum(gaussian_counts)}")
#             print(f"  G per tile: avg={avg_gaussians:.1f}, max={max(gaussian_counts)}")

#         return mx.clip(image, 0.0, 1.0)
