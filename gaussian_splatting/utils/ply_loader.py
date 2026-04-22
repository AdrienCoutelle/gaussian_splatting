import numpy as np
import torch
from plyfile import PlyData

from gaussian_splatting.structures.gaussian import GaussianCollection


def load_ply_gaussians(ply_path: str) -> GaussianCollection:
    """
    Load Gaussian splatting data from a PLY file.

    The PLY file contains:
    - Position (x, y, z)
    - Spherical harmonics (f_dc_0-2 for DC, f_rest_0-44 for higher orders)
    - Opacity
    - Scale (scale_0-2)
    - Rotation quaternion (rot_0-3)
    """
    ply_data = PlyData.read(ply_path)
    vertices = ply_data["vertex"]

    # Extract positions
    positions = np.stack([
        vertices['x'],
        vertices['y'],
        vertices['z'],
    ], axis=1)  # fmt:skip

    # Extract spherical harmonics DC component (base color)
    sh_dc = np.stack([
        vertices['f_dc_0'],
        vertices['f_dc_1'],
        vertices['f_dc_2'],
    ], axis=1)  # fmt:skip

    # Convert spherical harmonics to RGB
    # SH DC coefficient C0 = 1/(2*sqrt(π)) ≈ 0.28209479177387814
    C0 = 0.28209479177387814
    colors_rgb = np.clip(0.5 + sh_dc * C0, 0.0, 1.0)

    # Extract scale
    scales = np.stack([
        vertices['scale_0'],
        vertices['scale_1'],
        vertices['scale_2'],
    ], axis=1)  # fmt:skip

    # Extract rotation quaternion
    rotations = np.stack([
        vertices['rot_0'],
        vertices['rot_1'],
        vertices['rot_2'],
        vertices['rot_3'],
    ], axis=1)  # fmt:skip

    # Extract opacity
    opacities = vertices["opacity"]

    # Convert to tensors
    means = torch.from_numpy(positions).float()

    # Convert scale from log space to actual scale using exp
    scales_log = torch.from_numpy(scales).float()
    scales_tensor = torch.exp(scales_log)

    rotations_tensor = torch.from_numpy(rotations).float()
    colors = torch.from_numpy(colors_rgb).float()

    # Convert opacity from logit space to [0, 1] using sigmoid
    opacities_logit = torch.from_numpy(opacities).float()
    opacities_tensor = torch.sigmoid(opacities_logit).unsqueeze(1)

    # Compute covariance matrices from scale and rotation
    covariances = compute_covariance_from_scale_rotation(
        scale=scales_tensor,
        rotation=rotations_tensor,
    )

    # Compute and print statistics
    mean_position = means.mean(dim=0)
    distances_from_center = torch.norm(means - mean_position, dim=1)
    radius = distances_from_center.max().item()

    print(f"\n{'=' * 60}")
    print("Gaussian Splatting Object Statistics")
    print(f"{'=' * 60}")
    print(f"Number of Gaussians: {len(means)}")
    print(f"Mean position (centroid): [{mean_position[0]:.4f}, {mean_position[1]:.4f}, {mean_position[2]:.4f}]")
    print(f"Object radius (max distance from centroid): {radius:.4f}")
    print("Bounding box:")
    print(f"  X: [{means[:, 0].min():.4f}, {means[:, 0].max():.4f}]")
    print(f"  Y: [{means[:, 1].min():.4f}, {means[:, 1].max():.4f}]")
    print(f"  Z: [{means[:, 2].min():.4f}, {means[:, 2].max():.4f}]")
    print("\nColor statistics (RGB, clamped to [0,1]):")
    print(f"  R: [{colors[:, 0].min():.4f}, {colors[:, 0].max():.4f}], mean: {colors[:, 0].mean():.4f}")
    print(f"  G: [{colors[:, 1].min():.4f}, {colors[:, 1].max():.4f}], mean: {colors[:, 1].mean():.4f}")
    print(f"  B: [{colors[:, 2].min():.4f}, {colors[:, 2].max():.4f}], mean: {colors[:, 2].mean():.4f}")
    print("\nOpacity statistics (after sigmoid conversion):")
    print(f"  Range: [{opacities_tensor.min():.4f}, {opacities_tensor.max():.4f}], mean: {opacities_tensor.mean():.4f}")
    print("\nScale statistics (after exp conversion):")
    print(
        f"  Scale 0: [{scales_tensor[:, 0].min():.6f}, "
        f"{scales_tensor[:, 0].max():.6f}], mean: {scales_tensor[:, 0].mean():.6f}"
    )
    print(
        f"  Scale 1: [{scales_tensor[:, 1].min():.6f}, "
        f"{scales_tensor[:, 1].max():.6f}], mean: {scales_tensor[:, 1].mean():.6f}"
    )
    print(
        f"  Scale 2: [{scales_tensor[:, 2].min():.6f}, "
        f"{scales_tensor[:, 2].max():.6f}], mean: {scales_tensor[:, 2].mean():.6f}"
    )
    print(f"{'=' * 60}\n")

    return GaussianCollection.from_tensors(
        means=means,
        covariances=covariances,
        colors=colors,
        opacities=opacities_tensor,
    )


def compute_covariance_from_scale_rotation(
    scale: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """
    Compute 3x3 covariance matrices from scale and rotation quaternion.

    Covariance = R * S * S^T * R^T
    where R is rotation matrix and S is diagonal scale matrix.
    """
    # Normalize quaternions
    rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)

    # Extract quaternion components (w, x, y, z)
    w = rotation[:, 0]
    x = rotation[:, 1]
    y = rotation[:, 2]
    z = rotation[:, 3]

    # Build rotation matrices from quaternions
    R = torch.zeros(rotation.shape[0], 3, 3, dtype=rotation.dtype)

    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)

    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - w * x)

    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    # Create diagonal scale matrices
    S = torch.diag_embed(scale)

    # Compute covariance: R * S * S^T * R^T = R * S^2 * R^T
    S_squared = S @ S.transpose(-1, -2)
    covariance = R @ S_squared @ R.transpose(-1, -2)

    return covariance
