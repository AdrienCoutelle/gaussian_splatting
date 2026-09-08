uint pixel_idx = thread_position_in_grid.x;
uint tile_idx = uint(pixel_tile_indices[pixel_idx]);
uint tile_start = uint(tile_offsets[tile_idx]);
uint tile_end = uint(tile_offsets[tile_idx + 1]);

float pixel_x = pixels[pixel_idx * 2];
float pixel_y = pixels[pixel_idx * 2 + 1];
float transmittance = 1.0f;
float3 color = float3(0.0f);

for (uint slot_idx = tile_start; slot_idx < tile_end; ++slot_idx) {
    uint gaussian_idx = uint(tile_indices[slot_idx]);
    float dx = pixel_x - means[gaussian_idx * 2];
    float dy = pixel_y - means[gaussian_idx * 2 + 1];
    float distance_squared = dx * dx + dy * dy;
    float extent = extents[gaussian_idx];

    float conic_a = conics[gaussian_idx * 3];
    float conic_b = conics[gaussian_idx * 3 + 1];
    float conic_c = conics[gaussian_idx * 3 + 2];
    float power = -0.5f * (conic_a * dx * dx + 2.0f * conic_b * dx * dy + conic_c * dy * dy);

    if (distance_squared <= extent * extent && power <= 0.0f) {
        float alpha = metal::min(0.99f, opacities[gaussian_idx] * metal::exp(power));
        if (alpha < 1.0f / 255.0f) {
            continue;
        }

        float next_transmittance = transmittance * (1.0f - alpha);
        if (next_transmittance < 0.0001f) {
            break;
        }

        float weight = transmittance * alpha;
        color += weight * float3(
            colors[gaussian_idx * 3],
            colors[gaussian_idx * 3 + 1],
            colors[gaussian_idx * 3 + 2]
        );
        transmittance = next_transmittance;
    }
}

image[pixel_idx * 3] = color.x;
image[pixel_idx * 3 + 1] = color.y;
image[pixel_idx * 3 + 2] = color.z;