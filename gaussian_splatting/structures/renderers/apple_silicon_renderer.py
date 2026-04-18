import torch
from tqdm import tqdm

from gaussian_splatting.structures.renderers.base_renderer import BaseRenderer, ScreenSpaceGaussian


class NaiveRenderer(BaseRenderer):
    def _splat_gaussians_vectorized(
        self,
        image: torch.Tensor,
        gaussians: list[ScreenSpaceGaussian],
        sorted_indices: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> None:
        y_coords, x_coords = torch.meshgrid(
            torch.arange(image_height, device=self.device, dtype=torch.float32),
            torch.arange(image_width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        pixel_positions = torch.stack([x_coords, y_coords], dim=-1) + 0.5

        transmittance = torch.ones((image_height, image_width), device=self.device, dtype=torch.float32)

        for idx in tqdm(sorted_indices, desc="Splatting Gaussians", leave=False):
            gaussian = gaussians[idx.item()]

            mean_2d = gaussian.mean_2d
            cov_2d = gaussian.covariance_2d

            # Compute bounding box based on gaussian extent
            # Use conservative estimate: sqrt(max diagonal element + off-diagonal contribution)
            max_variance = torch.max(cov_2d[0, 0], cov_2d[1, 1]) + torch.abs(cov_2d[0, 1])
            max_extent = self.config.gaussian_extent * torch.sqrt(max_variance)

            min_x = int(torch.floor(mean_2d[0] - max_extent).item())
            max_x = int(torch.ceil(mean_2d[0] + max_extent).item())
            min_y = int(torch.floor(mean_2d[1] - max_extent).item())
            max_y = int(torch.ceil(mean_2d[1] + max_extent).item())

            # Clip to image bounds
            min_x = max(0, min_x)
            max_x = min(image_width, max_x)
            min_y = max(0, min_y)
            max_y = min(image_height, max_y)

            if min_x >= max_x or min_y >= max_y:
                continue

            # Extract relevant pixel positions
            pixels = pixel_positions[min_y:max_y, min_x:max_x]

            # Compute difference from mean
            delta = pixels - mean_2d

            # Compute inverse covariance
            cov_inv = torch.linalg.inv(cov_2d)

            # Mahalanobis distance: (x-μ)^T Σ^-1 (x-μ)
            mahalanobis = torch.sum(delta @ cov_inv * delta, dim=-1)

            # Gaussian weight
            weight = torch.exp(-0.5 * mahalanobis)

            # Apply opacity and clamp to valid range
            alpha = torch.clamp(weight * gaussian.opacity, 0.0, 1.0)

            # Alpha compositing (front to back)
            current_transmittance = transmittance[min_y:max_y, min_x:max_x]
            contribution = alpha * current_transmittance

            # Update image
            image[min_y:max_y, min_x:max_x] += contribution.unsqueeze(-1) * gaussian.color

            # Update transmittance
            transmittance[min_y:max_y, min_x:max_x] *= 1.0 - alpha
