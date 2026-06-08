import numpy as np
import torch
from plyfile import PlyData

from gaussian_splatting.structures.gaussian import GaussianCollection


def load_ply_gaussians(ply_path: str) -> GaussianCollection:
    ply_data = PlyData.read(ply_path)
    vertices = ply_data["vertex"]
    property_names = {property_.name for property_ in vertices.properties}

    full_gaussian_properties = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "opacity",
    }
    colmap_properties = {
        "x",
        "y",
        "z",
        "red",
        "green",
        "blue",
    }

    if property_names.issuperset(full_gaussian_properties):
        return _load_full_gaussian_ply(vertices=vertices)

    if property_names.issuperset(colmap_properties):
        return _load_colmap_point_cloud_ply(vertices=vertices)

    raise ValueError(
        f"Unsupported PLY schema for '{ply_path}'. "
        "Expected either full Gaussian Splatting properties "
        "(f_dc_*, scale_*, rot_*, opacity) or COLMAP properties (x,y,z,red,green,blue)."
    )


def _load_full_gaussian_ply(vertices) -> GaussianCollection:
    """Load a PLY file that already contains Gaussian Splatting attributes."""

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
        quaternions=rotations_tensor,
        scales=scales_tensor,
        colors=colors,
        opacities=opacities_tensor,
    )


def _load_colmap_point_cloud_ply(vertices) -> GaussianCollection:
    """Load a COLMAP point-cloud PLY and synthesize Gaussian attributes."""
    positions = np.stack([
        vertices["x"],
        vertices["y"],
        vertices["z"],
    ], axis=1)  # fmt:skip

    colors_rgb = np.stack([
        vertices["red"],
        vertices["green"],
        vertices["blue"],
    ], axis=1).astype(np.float32) / 255.0  # fmt:skip

    means = torch.from_numpy(positions).float()
    colors = torch.from_numpy(colors_rgb).float()

    sigma = _estimate_colmap_sigmas(
        means=means,
        vertices=vertices,
    )
    # Isotropic scale: same sigma in all 3 directions
    scales = sigma.unsqueeze(1).repeat(1, 3)  # (N, 3)
    # Identity rotation quaternion [w=1, x=0, y=0, z=0]
    n_points = means.shape[0]
    quaternions = torch.zeros((n_points, 4), dtype=torch.float32)
    quaternions[:, 0] = 1.0

    opacities = _estimate_colmap_opacities(vertices=vertices).unsqueeze(1)

    return GaussianCollection.from_tensors(
        means=means,
        quaternions=quaternions,
        scales=scales,
        colors=colors,
        opacities=opacities,
    )


def _estimate_colmap_sigmas(
    means: torch.Tensor,
    vertices,
) -> torch.Tensor:
    min_corner = means.min(dim=0).values
    max_corner = means.max(dim=0).values
    scene_extent = torch.norm(max_corner - min_corner)
    n_points = max(int(means.shape[0]), 1)

    base_sigma = torch.clamp(scene_extent / np.sqrt(float(n_points)), min=1e-4)
    sigma = torch.full((n_points,), fill_value=base_sigma, dtype=torch.float32)

    vertex_properties = {property_.name for property_ in vertices.properties}
    if "track_length" in vertex_properties:
        track_length = torch.from_numpy(np.asarray(vertices["track_length"])).float()
        median_track = torch.clamp(track_length.median(), min=1.0)
        confidence = torch.clamp(track_length / median_track, min=0.5, max=2.0)
        sigma = sigma / torch.sqrt(confidence)

    if "reprojection_error" in vertex_properties:
        reprojection_error = torch.from_numpy(np.asarray(vertices["reprojection_error"])).float()
        median_error = torch.clamp(reprojection_error.median(), min=1e-3)
        error_ratio = torch.clamp(reprojection_error / median_error, min=0.5, max=2.0)
        sigma = sigma * torch.sqrt(error_ratio)

    return torch.clamp(sigma, min=1e-4)


def _estimate_colmap_opacities(vertices) -> torch.Tensor:
    n_points = len(vertices["x"])
    quality = torch.ones((n_points,), dtype=torch.float32)

    vertex_properties = {property_.name for property_ in vertices.properties}
    if "track_length" in vertex_properties:
        track_length = torch.from_numpy(np.asarray(vertices["track_length"])).float()
        max_track_length = torch.clamp(track_length.max(), min=1.0)
        quality = quality * torch.clamp(track_length / max_track_length, min=0.1, max=1.0)

    if "reprojection_error" in vertex_properties:
        reprojection_error = torch.from_numpy(np.asarray(vertices["reprojection_error"])).float()
        median_error = torch.clamp(reprojection_error.median(), min=1e-3)
        error_quality = median_error / (median_error + reprojection_error)
        quality = quality * torch.clamp(error_quality, min=0.1, max=1.0)

    return torch.clamp(0.15 + 0.85 * quality, min=0.05, max=0.98)


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
