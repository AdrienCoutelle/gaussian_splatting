from pathlib import Path

import mlx.core as mx


class Rasterizer:
    """Wraps the Metal per-pixel alpha compositing kernel."""

    def __init__(self) -> None:
        source = (Path(__file__).parent / "kernels" / "splat.metal").read_text()
        self._kernel = mx.fast.metal_kernel(
            name="splat_gaussians",
            input_names=[
                "gauss_xy",
                "gauss_conic",
                "gauss_opacity",
                "gauss_color",
                "tile_origins",
                "tile_gstart",
                "tile_gcount",
                "tx",
                "ty",
                "width",
                "height",
                "sigma_cut",
                "eps",
            ],
            output_names=["out_rgb"],
            source=source,
        )

    def rasterize(
        self,
        gauss_xy: mx.array,
        gauss_conic: mx.array,
        gauss_opacity: mx.array,
        gauss_color: mx.array,
        tile_origins: mx.array,
        tile_gstart: mx.array,
        tile_gcount: mx.array,
        image_width: int,
        image_height: int,
        tile_size: tuple[int, int],
        sigma_cut: float,
        eps: float,
    ) -> mx.array:
        tx, ty = tile_size
        tile_width = (image_width + tx - 1) // tx
        tile_height = (image_height + ty - 1) // ty
        num_tiles = tile_width * tile_height
        P = tx * ty

        flat_output = self._kernel(
            inputs=[
                gauss_xy,
                gauss_conic,
                gauss_opacity,
                gauss_color,
                tile_origins,
                tile_gstart,
                tile_gcount,
                mx.array(tx, dtype=mx.uint32),
                mx.array(ty, dtype=mx.uint32),
                mx.array(image_width, dtype=mx.uint32),
                mx.array(image_height, dtype=mx.uint32),
                mx.array(sigma_cut, dtype=mx.float32),
                mx.array(eps, dtype=mx.float32),
            ],
            output_shapes=[(num_tiles * P, 3)],
            output_dtypes=[mx.float32],
            grid=(num_tiles * P, 1, 1),
            threadgroup=(min(P, 256), 1, 1),
        )[0]  # (num_tiles * P, 3)

        # Reshape tile-linear output to (H, W, 3):
        # (num_tiles * P, 3) -> (tile_height, tile_width, ty, tx, 3)
        # -> transpose -> (tile_height * ty, tile_width * tx, 3) -> slice
        tiled = flat_output.reshape(tile_height, tile_width, ty, tx, 3)
        tiled = mx.transpose(tiled, (0, 2, 1, 3, 4))  # (tile_height, ty, tile_width, tx, 3)
        full_image = tiled.reshape(tile_height * ty, tile_width * tx, 3)
        return full_image[:image_height, :image_width, :]
