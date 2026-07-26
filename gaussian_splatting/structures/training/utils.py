import mlx.core as mx


def gaussian_kernel(size: int = 11, sigma: float = 1.5):
    coords = mx.arange(size) - size // 2
    g = mx.exp(-(coords**2) / (2 * sigma**2))
    g = g / mx.sum(g)

    kernel = mx.outer(g, g)
    kernel = kernel / mx.sum(kernel)
    return kernel


def gaussian_blur(x: mx.array, kernel_size=11, sigma=1.5):
    """
    x: [H, W, C] or [N, H, W, C] — MLX channels-last format.
    Returns array with the same shape as input.
    """
    kernel = gaussian_kernel(kernel_size, sigma)

    # Add batch dim if needed: [H, W, C] -> [1, H, W, C]
    squeeze = x.ndim == 3
    if squeeze:
        x = x[None]

    C = x.shape[-1]  # channels-last: [N, H, W, C]

    weight = mx.zeros((C, kernel_size, kernel_size, 1))
    for c in range(C):
        weight[c, :, :, 0] = kernel

    out = mx.conv2d(
        x,
        weight,
        stride=1,
        padding=kernel_size // 2,
        groups=C,
    )

    return out[0] if squeeze else out


def ssim(
    img1: mx.array,
    img2: mx.array,
    kernel_size: int = 11,
    sigma: float = 1.5,
):
    C1 = 0.01**2
    C2 = 0.03**2

    mu1 = gaussian_blur(img1, kernel_size, sigma)
    mu2 = gaussian_blur(img2, kernel_size, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_blur(img1 * img1, kernel_size, sigma) - mu1_sq
    sigma2_sq = gaussian_blur(img2 * img2, kernel_size, sigma) - mu2_sq
    sigma12 = gaussian_blur(img1 * img2, kernel_size, sigma) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator

    return mx.mean(ssim_map)
