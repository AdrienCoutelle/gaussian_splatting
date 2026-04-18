"""MLX-based Apple Silicon renderer for Gaussian Splatting.

Algorithm switching via Protocol + Registry pattern:
- naive: Sequential per-tile processing (debug/validation)
- batch: vmap+einsum parallel processing (pure MLX fast version)
- metal: Metal kernel (production best)
"""

import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import mlx.core as mx
import numpy as np
import torch
from tqdm import tqdm

from gaussian_splatting.structures.renderers.base_renderer import BaseRenderer, ScreenSpaceGaussians
from gaussian_splatting.utils.profiler import profile


@dataclass
class AppleSiliconRendererParams:
    width: int
    height: int
    focal_length: float
    near_plane: float = 1e-4
    covariance_regularization: float = 0.3
    algorithm: Literal["naive", "batch", "metal"] = "metal"
    tile_size: tuple[int, int] = (16, 16)
    max_gaussians_per_tile: int = 4000
    batch_size: int = 64
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
            algorithm=config_dict.get("algorithm", "metal"),
            tile_size=tuple(config_dict.get("tile_size", (16, 16))),
            max_gaussians_per_tile=config_dict.get("max_gaussians_per_tile", 4000),
            batch_size=config_dict.get("batch_size", 64),
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


def _get_tile_gaussians(
    tile_bins: _TileBins,
    projected: _ProjectedGaussians,
    tile_id: int,
) -> dict[str, Any]:
    start = tile_bins.starts[tile_id]
    end = tile_bins.ends[tile_id]

    if start < 0 or end < 0:
        return {
            "count": 0,
            "xys": mx.zeros((0, 2)),
            "conic": mx.zeros((0, 3)),
            "opacity": mx.zeros((0, 1)),
            "color": mx.zeros((0, 3)),
            "depths": mx.zeros((0,)),
        }

    indices = tile_bins.gauss_ids_sorted[start:end]
    indices_mx = mx.array(indices, dtype=mx.int32)

    return {
        "count": len(indices),
        "xys": mx.take(projected.xys, indices_mx, axis=0),
        "conic": mx.take(projected.conic, indices_mx, axis=0),
        "opacity": mx.take(projected.opacity, indices_mx, axis=0).reshape(-1, 1),
        "color": mx.take(projected.color, indices_mx, axis=0),
        "depths": mx.take(projected.depths, indices_mx, axis=0),
    }


class _RasterizationAlgorithm(Protocol):
    """Protocol for rasterization algorithms."""

    def __call__(
        self,
        projected: _ProjectedGaussians,
        tile_bins: _TileBins,
        camera: _Camera,
        background: tuple[float, float, float],
        tile_size: tuple[int, int],
        verbose: bool,
        **kwargs: Any,
    ) -> mx.array: ...


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
"""


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


@mx.compile
def _render_tiles_fixed(
    pixels_b_p2: mx.array,
    xys_b_g2: mx.array,
    conic_b_g3: mx.array,
    opacity_b_g: mx.array,
    color_b_g3: mx.array,
    pixel_mask_b_p: mx.array,
    gauss_mask_b_g: mx.array,
    bg_color: mx.array,
    sigma_cut: float = 12.0,
) -> mx.array:
    def render_one(
        pixels_p2,
        xys_g2,
        conic_g3,
        opacity_g,
        color_g3,
        p_mask_p,
        g_mask_g,
    ):
        d = pixels_p2[None, :, :] - xys_g2[:, None, :]
        dx, dy = d[..., 0], d[..., 1]

        q11, q12, q22 = conic_g3[:, 0:1], conic_g3[:, 1:2], conic_g3[:, 2:3]
        sigma = 0.5 * (q11 * dx * dx + 2.0 * q12 * dx * dy + q22 * dy * dy)

        alpha = opacity_g[:, None] * mx.exp(-mx.minimum(sigma, sigma_cut))
        alpha = mx.minimum(alpha * g_mask_g[:, None], 0.999).astype(mx.float32)

        T_incl = mx.cumprod(1.0 - alpha, axis=0)
        T_before = mx.concatenate([mx.ones_like(T_incl[0:1, :]), T_incl[:-1, :]], axis=0)
        w = T_before * alpha
        T_final = T_incl[-1, :]

        C = mx.einsum("gp,gc->pc", w, color_g3)
        C = C + T_final[:, None] * bg_color[None, :]

        return mx.where(p_mask_p[:, None] > 0.5, C, 0.0)

    f = mx.vmap(render_one, in_axes=(0, 0, 0, 0, 0, 0, 0))
    return f(
        pixels_b_p2,
        xys_b_g2,
        conic_b_g3,
        opacity_b_g,
        color_b_g3,
        pixel_mask_b_p,
        gauss_mask_b_g,
    )


def _rasterize_tile_mlx_compiled(
    pixels: mx.array,
    gauss_xy: mx.array,
    gauss_conic: mx.array,
    gauss_opacity: mx.array,
    gauss_color: mx.array,
    bg_color: mx.array,
) -> mx.array:
    n_pixels: int = int(pixels.shape[0])
    n_gaussians: int = int(gauss_xy.shape[0])

    if n_gaussians == 0:
        return mx.broadcast_to(bg_color, (n_pixels, 3))

    dx = pixels[:, None, 0] - gauss_xy[None, :, 0]
    dy = pixels[:, None, 1] - gauss_xy[None, :, 1]

    sigma = 0.5 * (gauss_conic[:, 0] * dx * dx + 2.0 * gauss_conic[:, 1] * dx * dy + gauss_conic[:, 2] * dy * dy)

    valid_mask = sigma <= 12.0
    sigma_clipped = mx.where(valid_mask, sigma, 0.0)
    gauss_weights = mx.where(valid_mask, mx.exp(-sigma_clipped), 0.0)

    gauss_opacity_flat = gauss_opacity.squeeze(-1) if gauss_opacity.ndim > 1 else gauss_opacity
    alphas = gauss_opacity_flat[None, :] * gauss_weights
    alphas = mx.clip(alphas, 0.0, 0.999)

    transmittance = mx.cumprod(1.0 - alphas, axis=1)

    ones = mx.ones((n_pixels, 1))
    if n_gaussians > 1:
        transmittance_shifted = mx.concatenate([ones, transmittance[:, :-1]], axis=1)
    else:
        transmittance_shifted = ones

    contributions = alphas * transmittance_shifted

    pixel_colors = mx.sum(contributions[:, :, None] * gauss_color[None, :, :], axis=1)

    final_transmittance = transmittance[:, -1] if n_gaussians > 0 else mx.ones(n_pixels)

    pixel_colors = pixel_colors + final_transmittance[:, None] * bg_color[None, :]

    return pixel_colors


@profile
def _rasterize_naive(
    projected: _ProjectedGaussians,
    tile_bins: _TileBins,
    camera: _Camera,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tile_size: tuple[int, int] = (16, 16),
    verbose: bool = False,
    **kwargs: Any,
) -> mx.array:
    width, height = camera.width, camera.height
    tx, ty = tile_size
    tile_width = (width + tx - 1) // tx
    tile_height = (height + ty - 1) // ty

    bg_color = mx.array(background)

    image = mx.zeros((height, width, 3))

    total_tiles = tile_width * tile_height
    max_gaussians_per_tile = 0

    if verbose:
        pbar = tqdm(total=total_tiles, desc="Rasterizing (naive)")

    for tile_y in range(tile_height):
        for tile_x in range(tile_width):
            tile_id = tile_y * tile_width + tile_x

            y_start = tile_y * ty
            y_end = min(y_start + ty, height)
            x_start = tile_x * tx
            x_end = min(x_start + tx, width)

            if y_start >= y_end or x_start >= x_end:
                continue

            tile_data = _get_tile_gaussians(tile_bins, projected, tile_id)
            n_gaussians: int = int(tile_data["count"])
            max_gaussians_per_tile = max(max_gaussians_per_tile, n_gaussians)

            if n_gaussians == 0:
                tile_h = y_end - y_start
                tile_w = x_end - x_start
                if isinstance(background, list | tuple):
                    background_array = mx.array(background)
                else:
                    background_array = background
                image[y_start:y_end, x_start:x_end, :] = background_array
                if verbose:
                    pbar.update(1)
                continue

            y_coords = mx.arange(y_start, y_end, dtype=mx.float32) + 0.5
            x_coords = mx.arange(x_start, x_end, dtype=mx.float32) + 0.5

            yy, xx = mx.meshgrid(y_coords, x_coords, indexing="ij")
            pixels = mx.stack([xx.flatten(), yy.flatten()], axis=1)

            pixel_colors = _rasterize_tile_mlx_compiled(
                pixels,
                tile_data["xys"],
                tile_data["conic"],
                tile_data["opacity"],
                tile_data["color"],
                bg_color,
            )

            tile_h = y_end - y_start
            tile_w = x_end - x_start
            pixel_colors_reshaped = pixel_colors.reshape(tile_h, tile_w, 3)

            image[y_start:y_end, x_start:x_end, :] = pixel_colors_reshaped

            if verbose:
                pbar.update(1)

    if verbose:
        print(f"Max gaussians per tile: {max_gaussians_per_tile}")
        pbar.close()

    return image


@profile
def _rasterize_batch(
    projected: _ProjectedGaussians,
    tile_bins: _TileBins,
    camera: _Camera,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tile_size: tuple[int, int] = (16, 16),
    verbose: bool = False,
    **kwargs: Any,
) -> mx.array:
    batch_size = kwargs.get("batch_size", 64)
    K = kwargs.get("K", 4000)
    sigma_cut = kwargs.get("sigma_cut", 12.0)

    width, height = camera.width, camera.height
    tx, ty = tile_size
    T_w = (width + tx - 1) // tx
    T_h = (height + ty - 1) // ty
    T = T_w * T_h

    P = tx * ty
    G = K

    bg_color = mx.array(background, dtype=mx.float32)
    image = mx.zeros((height, width, 3), dtype=mx.float32)

    yy, xx = mx.meshgrid(
        mx.arange(ty, dtype=mx.float32) + 0.5,
        mx.arange(tx, dtype=mx.float32) + 0.5,
        indexing="ij",
    )
    tile_local = mx.stack([xx.flatten(), yy.flatten()], axis=1)

    if verbose:
        pbar = tqdm(total=T, desc=f"Rasterizing (batch, B={batch_size}, K={K})")

    tile_id = 0
    while tile_id < T:
        pixels_list = []
        xys_list = []
        conic_list = []
        opacity_list = []
        color_list = []
        pixel_mask_list = []
        gauss_mask_list = []
        coords = []

        for _i in range(batch_size):
            if tile_id >= T:
                pixels_list.append(mx.zeros((P, 2), dtype=mx.float32))
                xys_list.append(mx.zeros((G, 2), dtype=mx.float32))
                conic_list.append(mx.zeros((G, 3), dtype=mx.float32))
                opacity_list.append(mx.zeros(G, dtype=mx.float32))
                color_list.append(mx.zeros((G, 3), dtype=mx.float32))
                pixel_mask_list.append(mx.zeros(P, dtype=mx.float32))
                gauss_mask_list.append(mx.zeros(G, dtype=mx.float32))
                coords.append((0, 0, 0, 0))
                continue

            ty_idx, tx_idx = divmod(tile_id, T_w)
            y0, x0 = ty_idx * ty, tx_idx * tx
            y1, x1 = min(y0 + ty, height), min(x0 + tx, width)

            pixels = tile_local + mx.array([x0, y0], dtype=mx.float32)
            p_mask = ((pixels[:, 0] < x1) & (pixels[:, 1] < y1)).astype(mx.float32)

            td = _get_tile_gaussians(tile_bins, projected, tile_id)
            cnt = int(td["count"])

            xys_g = mx.zeros((G, 2), dtype=mx.float32)
            conic_g = mx.zeros((G, 3), dtype=mx.float32)
            opacity_g = mx.zeros(G, dtype=mx.float32)
            color_g = mx.zeros((G, 3), dtype=mx.float32)
            gauss_mask_g = mx.zeros(G, dtype=mx.float32)

            if cnt > 0:
                op = td["opacity"].squeeze(-1) if td["opacity"].ndim == 2 else td["opacity"]

                if cnt > G:
                    center = mx.array([x0 + 0.5 * tx, y0 + 0.5 * ty], dtype=mx.float32)
                    dx_center = center[0] - td["xys"][:, 0]
                    dy_center = center[1] - td["xys"][:, 1]
                    s = 0.5 * (
                        td["conic"][:, 0] * dx_center * dx_center
                        + 2.0 * td["conic"][:, 1] * dx_center * dy_center
                        + td["conic"][:, 2] * dy_center * dy_center
                    )
                    score = op * mx.exp(-mx.minimum(s, 12.0))
                    top_k_idx = mx.argsort(-score)[:G]

                    selected_depths = td["depths"][top_k_idx]
                    depth_order = mx.argsort(selected_depths)
                    idx = top_k_idx[depth_order]
                else:
                    idx = mx.arange(cnt)

                g = int(idx.shape[0])
                xys_g = (
                    mx.concatenate([td["xys"][idx], mx.zeros((G - g, 2), dtype=mx.float32)], axis=0)
                    if g < G
                    else td["xys"][idx][:G]
                )
                conic_g = (
                    mx.concatenate([td["conic"][idx], mx.zeros((G - g, 3), dtype=mx.float32)], axis=0)
                    if g < G
                    else td["conic"][idx][:G]
                )
                opacity_g = (
                    mx.concatenate([op[idx], mx.zeros(G - g, dtype=mx.float32)], axis=0) if g < G else op[idx][:G]
                )
                color_g = (
                    mx.concatenate([td["color"][idx], mx.zeros((G - g, 3), dtype=mx.float32)], axis=0)
                    if g < G
                    else td["color"][idx][:G]
                )
                gauss_mask_g = (
                    mx.concatenate([mx.ones(g, dtype=mx.float32), mx.zeros(G - g, dtype=mx.float32)], axis=0)
                    if g < G
                    else mx.ones(G, dtype=mx.float32)
                )

            pixels_list.append(pixels)
            xys_list.append(xys_g)
            conic_list.append(conic_g)
            opacity_list.append(opacity_g)
            color_list.append(color_g)
            pixel_mask_list.append(p_mask)
            gauss_mask_list.append(gauss_mask_g)
            coords.append((y0, y1, x0, x1))
            tile_id += 1

        pixels_b_p2 = mx.stack(pixels_list, axis=0)
        xys_b_g2 = mx.stack(xys_list, axis=0)
        conic_b_g3 = mx.stack(conic_list, axis=0)
        opacity_b_g = mx.stack(opacity_list, axis=0)
        color_b_g3 = mx.stack(color_list, axis=0)
        pixel_mask_b_p = mx.stack(pixel_mask_list, axis=0)
        gauss_mask_b_g = mx.stack(gauss_mask_list, axis=0)

        batch_out = _render_tiles_fixed(
            pixels_b_p2,
            xys_b_g2,
            conic_b_g3,
            opacity_b_g,
            color_b_g3,
            pixel_mask_b_p,
            gauss_mask_b_g,
            bg_color,
            sigma_cut,
        )

        for i, (y0, y1, x0, x1) in enumerate(coords):
            if y0 == y1 or x0 == x1:
                continue
            tile_h, tile_w = y1 - y0, x1 - x0
            tile_rgb = batch_out[i].reshape(ty, tx, 3)[:tile_h, :tile_w, :]
            image[y0:y1, x0:x1, :] = tile_rgb

        if verbose:
            pbar.update(min(batch_size, T - (tile_id - batch_size)))

    if verbose:
        pbar.close()

    return image


@profile
def _rasterize_metal(
    projected: _ProjectedGaussians,
    tile_bins: _TileBins,
    camera: _Camera,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tile_size: tuple[int, int] = (16, 16),
    verbose: bool = False,
    **kwargs: Any,
) -> mx.array:
    return _rasterize_gaussians_metal(
        projected,
        tile_bins,
        camera,
        background,
        tile_size,
        max_gaussians_per_tile=kwargs.get("max_gaussians_per_tile", 4000),
        sigma_cut=kwargs.get("sigma_cut", 12.0),
        eps=kwargs.get("eps", 1e-3),
        verbose=verbose,
    )


_RASTERIZATION_ALGORITHMS: dict[str, _RasterizationAlgorithm] = {
    "naive": _rasterize_naive,
    "batch": _rasterize_batch,
    "metal": _rasterize_metal,
}


@profile
def _rasterize_gaussians(
    projected: _ProjectedGaussians,
    tile_bins: _TileBins,
    camera: _Camera,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tile_size: tuple[int, int] = (16, 16),
    verbose: bool = False,
    algorithm: str = "metal",
    **kwargs: Any,
) -> mx.array:
    if algorithm not in _RASTERIZATION_ALGORITHMS:
        available = list(_RASTERIZATION_ALGORITHMS.keys())
        raise ValueError(f"Unknown rasterization algorithm: {algorithm}. Available: {available}")
    return _RASTERIZATION_ALGORITHMS[algorithm](
        projected,
        tile_bins,
        camera,
        background,
        tile_size,
        verbose,
        **kwargs,
    )
