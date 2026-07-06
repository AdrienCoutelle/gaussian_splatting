/**
 * Per-pixel alpha compositing kernel for 2D Gaussian splatting.
 *
 * Grid: (num_tiles * tx * ty, 1, 1) — one thread per pixel across all tiles.
 *
 * Inputs (flat buffers, row-major):
 *   gauss_xy      float32 (M, 2)  — 2D means of sorted Gaussians
 *   gauss_conic   float32 (M, 3)  — upper-tri of inverse 2D covariance (q11, q12, q22)
 *   gauss_opacity float32 (M,)    — per-Gaussian opacity
 *   gauss_color   float32 (M, 3)  — per-Gaussian RGB color
 *   tile_origins  uint32  (T, 2)  — pixel (x0, y0) of top-left corner for each tile
 *   tile_gstart   uint32  (T,)    — index into sorted arrays where each tile starts
 *   tile_gcount   uint32  (T,)    — number of Gaussians assigned to each tile
 *   tx, ty        uint32          — tile width and height in pixels
 *   width, height uint32          — image dimensions
 *   sigma_cut     float32         — exponent threshold; skip Gaussians with sigma > sigma_cut
 *   eps           float32         — early-exit transmittance threshold
 *
 * Output:
 *   out_rgb  float32 (num_tiles * tx * ty, 3)  — tile-linear RGB output
 */

const uint P    = tx * ty;
const uint gtid = thread_position_in_grid.x;

const uint tile_id = gtid / P;
const uint pid     = gtid % P;

const uint px = pid % tx;
const uint py = pid / tx;

const uint x0 = tile_origins[tile_id * 2 + 0];
const uint y0 = tile_origins[tile_id * 2 + 1];

const uint img_x = x0 + px;
const uint img_y = y0 + py;

const uint out_idx = (tile_id * P + pid) * 3;

if (img_x >= width || img_y >= height) {
    out_rgb[out_idx + 0] = 0.0f;
    out_rgb[out_idx + 1] = 0.0f;
    out_rgb[out_idx + 2] = 0.0f;
    return;
}

const float cx = (float)img_x + 0.5f;
const float cy = (float)img_y + 0.5f;

const uint gstart = tile_gstart[tile_id];
const uint gcount = tile_gcount[tile_id];

float T = 1.0f;
float r = 0.0f, g = 0.0f, b = 0.0f;

for (uint i = 0; i < gcount; ++i) {
    const uint idx = gstart + i;

    const float mu_x = gauss_xy[idx * 2 + 0];
    const float mu_y = gauss_xy[idx * 2 + 1];
    const float q11  = gauss_conic[idx * 3 + 0];
    const float q12  = gauss_conic[idx * 3 + 1];
    const float q22  = gauss_conic[idx * 3 + 2];
    const float op   = gauss_opacity[idx];

    const float dx    = cx - mu_x;
    const float dy    = cy - mu_y;
    const float sigma = 0.5f * (q11 * dx * dx + 2.0f * q12 * dx * dy + q22 * dy * dy);

    if (sigma > sigma_cut || !isfinite(sigma)) continue;

    const float alpha = fmin(0.999f, fmax(0.0f, op) * exp(-sigma));

    r += T * alpha * gauss_color[idx * 3 + 0];
    g += T * alpha * gauss_color[idx * 3 + 1];
    b += T * alpha * gauss_color[idx * 3 + 2];
    T *= (1.0f - alpha);

    if (T < eps) break;
}

out_rgb[out_idx + 0] = r;
out_rgb[out_idx + 1] = g;
out_rgb[out_idx + 2] = b;
