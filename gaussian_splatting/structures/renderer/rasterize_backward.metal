uint pixel_idx = thread_position_in_grid.x;
uint tile_idx = uint(pixel_tile_indices[pixel_idx]);
uint gaussians_per_tile = tile_indices_shape[1];

float pixel_x = pixels[pixel_idx * 2];
float pixel_y = pixels[pixel_idx * 2 + 1];
float transmittance = 1.0f;

for (uint slot_idx = 0; slot_idx < gaussians_per_tile; ++slot_idx) {
    uint gaussian_idx = uint(tile_indices[tile_idx * gaussians_per_tile + slot_idx]);
    float dx = pixel_x - means[gaussian_idx * 2];
    float dy = pixel_y - means[gaussian_idx * 2 + 1];
    float extent = extents[gaussian_idx];
    float conic_a = conics[gaussian_idx * 3];
    float conic_b = conics[gaussian_idx * 3 + 1];
    float conic_c = conics[gaussian_idx * 3 + 2];
    float power = -0.5f * (conic_a * dx * dx + 2.0f * conic_b * dx * dy + conic_c * dy * dy);
    bool is_active = dx * dx + dy * dy <= extent * extent && power <= 0.0f;
    float alpha = is_active ? opacities[gaussian_idx] * metal::exp(power) : 0.0f;

    transmittance *= 1.0f - alpha;
}

float3 cotangent_color = float3(
    cotangent[pixel_idx * 3],
    cotangent[pixel_idx * 3 + 1],
    cotangent[pixel_idx * 3 + 2]
);
float3 suffix_color = float3(0.0f);
float pixel_grad_x = 0.0f;
float pixel_grad_y = 0.0f;

for (int slot_idx = int(gaussians_per_tile) - 1; slot_idx >= 0; --slot_idx) {
    uint gaussian_idx = uint(tile_indices[tile_idx * gaussians_per_tile + uint(slot_idx)]);
    float dx = pixel_x - means[gaussian_idx * 2];
    float dy = pixel_y - means[gaussian_idx * 2 + 1];
    float extent = extents[gaussian_idx];
    float conic_a = conics[gaussian_idx * 3];
    float conic_b = conics[gaussian_idx * 3 + 1];
    float conic_c = conics[gaussian_idx * 3 + 2];
    float power = -0.5f * (conic_a * dx * dx + 2.0f * conic_b * dx * dy + conic_c * dy * dy);
    bool is_active = dx * dx + dy * dy <= extent * extent && power <= 0.0f;
    float alpha = is_active ? opacities[gaussian_idx] * metal::exp(power) : 0.0f;
    float gaussian_transmittance = transmittance / metal::max(1.0f - alpha, 1e-8f);
    float3 gaussian_color = float3(
        colors[gaussian_idx * 3],
        colors[gaussian_idx * 3 + 1],
        colors[gaussian_idx * 3 + 2]
    );

    float color_weight = gaussian_transmittance * alpha;
    atomic_fetch_add_explicit(
        &colors_grad[gaussian_idx * 3],
        color_weight * cotangent_color.x,
        memory_order_relaxed
    );
    atomic_fetch_add_explicit(
        &colors_grad[gaussian_idx * 3 + 1],
        color_weight * cotangent_color.y,
        memory_order_relaxed
    );
    atomic_fetch_add_explicit(
        &colors_grad[gaussian_idx * 3 + 2],
        color_weight * cotangent_color.z,
        memory_order_relaxed
    );

    if (is_active) {
        float alpha_grad = gaussian_transmittance * dot(cotangent_color, gaussian_color - suffix_color);
        float exp_power = metal::exp(power);
        float power_grad = alpha_grad * alpha;
        float mean_grad_x = power_grad * (conic_a * dx + conic_b * dy);
        float mean_grad_y = power_grad * (conic_b * dx + conic_c * dy);

        atomic_fetch_add_explicit(&means_grad[gaussian_idx * 2], mean_grad_x, memory_order_relaxed);
        atomic_fetch_add_explicit(&means_grad[gaussian_idx * 2 + 1], mean_grad_y, memory_order_relaxed);
        atomic_fetch_add_explicit(
            &conics_grad[gaussian_idx * 3],
            -0.5f * power_grad * dx * dx,
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &conics_grad[gaussian_idx * 3 + 1],
            -power_grad * dx * dy,
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &conics_grad[gaussian_idx * 3 + 2],
            -0.5f * power_grad * dy * dy,
            memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            &opacities_grad[gaussian_idx],
            alpha_grad * exp_power,
            memory_order_relaxed
        );

        pixel_grad_x -= mean_grad_x;
        pixel_grad_y -= mean_grad_y;
    }

    suffix_color = alpha * gaussian_color + (1.0f - alpha) * suffix_color;
    transmittance = gaussian_transmittance;
}

atomic_fetch_add_explicit(&pixels_grad[pixel_idx * 2], pixel_grad_x, memory_order_relaxed);
atomic_fetch_add_explicit(&pixels_grad[pixel_idx * 2 + 1], pixel_grad_y, memory_order_relaxed);