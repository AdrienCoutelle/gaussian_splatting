import math

import mlx.core as mx

_SH_C0 = 0.5 * math.sqrt(1.0 / math.pi)
_SH_C1 = 0.5 * math.sqrt(3.0 / math.pi)
_SH_C2 = [
    0.5 * math.sqrt(15.0 / math.pi),
    -0.5 * math.sqrt(15.0 / math.pi),
    0.25 * math.sqrt(5.0 / math.pi),
    -0.5 * math.sqrt(15.0 / math.pi),
    0.25 * math.sqrt(15.0 / math.pi),
]
_SH_C3 = [
    -0.25 * math.sqrt(35.0 / (2.0 * math.pi)),
    0.5 * math.sqrt(105.0 / math.pi),
    -0.25 * math.sqrt(21.0 / (2.0 * math.pi)),
    0.25 * math.sqrt(7.0 / math.pi),
    -0.25 * math.sqrt(21.0 / (2.0 * math.pi)),
    0.25 * math.sqrt(105.0 / math.pi),
    -0.25 * math.sqrt(35.0 / (2.0 * math.pi)),
]


def _evaluate_sh(
    sh_coeffs: mx.array,
    directions: mx.array,
) -> mx.array:
    """
    Evaluate spherical harmonics at unit viewing directions.

    :param sh_coeffs: SH coefficients of shape (N, num_coeffs, 3), supporting degrees 0–3.
    :param directions: Unit viewing directions of shape (N, 3), from Gaussian toward the camera.
    :return: RGB colors of shape (N, 3), clamped to [0, 1].
    """
    num_coeffs = sh_coeffs.shape[1]

    result = _SH_C0 * sh_coeffs[:, 0, :]  # (N, 3)

    if num_coeffs > 1:
        x = directions[:, 0:1]
        y = directions[:, 1:2]
        z = directions[:, 2:3]
        result = (
            result - _SH_C1 * y * sh_coeffs[:, 1, :] + _SH_C1 * z * sh_coeffs[:, 2, :] - _SH_C1 * x * sh_coeffs[:, 3, :]
        )

    if num_coeffs > 4:
        x = directions[:, 0:1]
        y = directions[:, 1:2]
        z = directions[:, 2:3]
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = (
            result
            + _SH_C2[0] * xy * sh_coeffs[:, 4, :]
            + _SH_C2[1] * yz * sh_coeffs[:, 5, :]
            + _SH_C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[:, 6, :]
            + _SH_C2[3] * xz * sh_coeffs[:, 7, :]
            + _SH_C2[4] * (xx - yy) * sh_coeffs[:, 8, :]
        )

    if num_coeffs > 9:
        x = directions[:, 0:1]
        y = directions[:, 1:2]
        z = directions[:, 2:3]
        xx, yy, zz = x * x, y * y, z * z
        xy = x * y
        result = (
            result
            + _SH_C3[0] * y * (3 * xx - yy) * sh_coeffs[:, 9, :]
            + _SH_C3[1] * xy * z * sh_coeffs[:, 10, :]
            + _SH_C3[2] * y * (4 * zz - xx - yy) * sh_coeffs[:, 11, :]
            + _SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_coeffs[:, 12, :]
            + _SH_C3[4] * x * (4 * zz - xx - yy) * sh_coeffs[:, 13, :]
            + _SH_C3[5] * z * (xx - yy) * sh_coeffs[:, 14, :]
            + _SH_C3[6] * x * (xx - 3 * yy) * sh_coeffs[:, 15, :]
        )

    return mx.clip(result + 0.5, 0.0, 1.0)


def _quaternions_to_rotation_matrices(quaternions: mx.array) -> mx.array:
    """Convert (N, 4) quaternions [w, x, y, z] to (N, 3, 3) rotation matrices."""
    norms = mx.clip(mx.sqrt(mx.sum(quaternions**2, axis=1, keepdims=True)), 1e-12, None)
    quaternions = quaternions / norms
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    return mx.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        axis=1,
    ).reshape(-1, 3, 3)
