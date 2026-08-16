from pathlib import Path

import mlx.core as mx


def _load_kernel_source(filename: str) -> str:
    return Path(__file__).with_name(filename).read_text(encoding="utf-8")


_forward_kernel = mx.fast.metal_kernel(
    name="gaussian_rasterize_forward",
    input_names=[
        "pixels",
        "means",
        "conics",
        "colors",
        "opacities",
        "extents",
        "tile_indices",
        "pixel_tile_indices",
    ],
    output_names=["image"],
    source=_load_kernel_source("rasterize_forward.metal"),
)

_backward_kernel = mx.fast.metal_kernel(
    name="gaussian_rasterize_backward",
    input_names=[
        "pixels",
        "means",
        "conics",
        "colors",
        "opacities",
        "extents",
        "tile_indices",
        "pixel_tile_indices",
        "cotangent",
    ],
    output_names=[
        "pixels_grad",
        "means_grad",
        "conics_grad",
        "colors_grad",
        "opacities_grad",
        "extents_grad",
        "tile_indices_grad",
        "pixel_tile_indices_grad",
    ],
    source=_load_kernel_source("rasterize_backward.metal"),
    atomic_outputs=True,
)


@mx.custom_function
def rasterize_image(
    pixels: mx.array,
    means: mx.array,
    conics: mx.array,
    colors: mx.array,
    opacities: mx.array,
    extents: mx.array,
    tile_indices: mx.array,
    pixel_tile_indices: mx.array,
) -> mx.array:
    pixel_count = pixels.shape[0]
    outputs = _forward_kernel(
        inputs=[pixels, means, conics, colors, opacities, extents, tile_indices, pixel_tile_indices],
        output_shapes=[(pixel_count, 3)],
        output_dtypes=[mx.float32],
        grid=(pixel_count, 1, 1),
        threadgroup=(min(pixel_count, 256), 1, 1),
    )
    return outputs[0]


@rasterize_image.vjp
def _rasterize_image_vjp(
    primals: tuple[mx.array, ...],
    cotangent: mx.array,
    _: mx.array,
) -> tuple[mx.array, ...]:
    pixels, means, conics, colors, opacities, extents, tile_indices, pixel_tile_indices = primals
    pixel_count = pixels.shape[0]
    outputs = _backward_kernel(
        inputs=[
            pixels,
            means,
            conics,
            colors,
            opacities,
            extents,
            tile_indices,
            pixel_tile_indices,
            cotangent,
        ],
        output_shapes=[
            pixels.shape,
            means.shape,
            conics.shape,
            colors.shape,
            opacities.shape,
            extents.shape,
            tile_indices.shape,
            pixel_tile_indices.shape,
        ],
        output_dtypes=[mx.float32] * 8,
        grid=(pixel_count, 1, 1),
        threadgroup=(min(pixel_count, 256), 1, 1),
        init_value=0,
    )
    return tuple(outputs)
