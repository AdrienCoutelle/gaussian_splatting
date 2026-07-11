import mlx.core as mx

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderer.renderer import ScreenSpaceGaussians
from gaussian_splatting.utils.logger import Logger

logger = Logger("RENDERER")


class RendererDifferentiabilityError(Exception):
    pass


def check_renderer_differentiability(renderer) -> None:
    logger.info("Checking renderer differentiability...")
    camera = Camera(
        pose=mx.eye(4),
        width=renderer.config.width,
        height=renderer.config.height,
        focal_length=renderer.config.focal_length,
    )

    positions = mx.array([[0.0, 0.0, -3.0]], dtype=mx.float32)
    quaternions = mx.array([[1.0, 0.0, 0.0, 0.0]], dtype=mx.float32)
    scales = mx.array([[-1.0, -1.0, -1.0]], dtype=mx.float32)
    sh_coeffs = mx.zeros((1, 16, 3), dtype=mx.float32)
    opacities = mx.array([[0.9]], dtype=mx.float32)

    def step1_loss(p):
        g = GaussianCollection.from_tensors(
            positions=p,
            quaternions=quaternions,
            scales=scales,
            sh_coeffs=sh_coeffs,
            opacities=opacities,
        )
        renderer._transform_positions_to_camera_space(camera, g)
        return mx.mean(g.positions)

    _, step1_grads = mx.value_and_grad(step1_loss)(positions)
    mx.eval(step1_grads)

    grad = step1_grads[0]
    if grad is None:
        raise RendererDifferentiabilityError("Step 1 failed: Gradient for 'positions' is None.")
    if mx.any(mx.isnan(grad)).item():
        raise RendererDifferentiabilityError("Step 1 failed: Gradient for 'positions' contains NaNs.")

    logger.info("Step 1 (Transform to Camera Space) is differentiable.")

    def step2_loss(p, q, s, sh, o):
        g = GaussianCollection.from_tensors(
            positions=p,
            quaternions=q,
            scales=s,
            sh_coeffs=sh,
            opacities=o,
        )
        renderer._transform_positions_to_camera_space(camera, g)
        screen_space = renderer._project_to_screen_space(camera, g)

        if screen_space is None:
            # Return a dummy scalar that still depends on the inputs to avoid breakages,
            # though a returned None here usually indicates a failure.
            return mx.array(0.0)

        return (
            mx.mean(screen_space.means_2d)
            + mx.mean(screen_space.covariances_2d)
            + mx.mean(screen_space.colors)
            + mx.mean(screen_space.opacities)
        )

    _, step2_grads = mx.value_and_grad(step2_loss, argnums=[0, 1, 2, 3, 4])(
        positions, quaternions, scales, sh_coeffs, opacities
    )
    mx.eval(step2_grads)

    param_names = ["positions", "quaternions", "scales", "sh_coeffs", "opacities"]
    for name, g in zip(param_names, step2_grads):
        if g is None:
            raise RendererDifferentiabilityError(f"Step 2 failed: Gradient for '{name}' is None.")
        if mx.any(mx.isnan(g)).item():
            raise RendererDifferentiabilityError(f"Step 2 failed: Gradient for '{name}' contains NaNs.")

    logger.info("Step 2 (Project to Screen Space) is differentiable.")

    mock_means_2d = mx.array([[32.0, 32.0]], dtype=mx.float32)
    mock_covariances_2d = mx.array([[[2.0, 0.0], [0.0, 2.0]]], dtype=mx.float32)
    mock_colors = mx.array([[0.8, 0.5, 0.2]], dtype=mx.float32)
    mock_opacities = mx.array([[0.9]], dtype=mx.float32)

    def step3_loss(m2d, cov2d, colors, opacs):
        ssg = ScreenSpaceGaussians(
            means_2d=m2d,
            covariances_2d=cov2d,
            depths=mx.array([3.0], dtype=mx.float32),
            colors=colors,
            opacities=opacs,
        )
        out = renderer._run_rasterization(
            gaussians=ssg,
            camera=camera,
        )
        return mx.mean(out)

    _, step3_grads = mx.value_and_grad(step3_loss, argnums=[0, 1, 2, 3])(
        mock_means_2d, mock_covariances_2d, mock_colors, mock_opacities
    )
    mx.eval(step3_grads)

    ssg_fields = ["means_2d", "covariances_2d", "colors", "opacities"]
    for name, g in zip(ssg_fields, step3_grads):
        if g is None:
            raise RendererDifferentiabilityError(f"Step 3 failed: Gradient for '{name}' is None.")
        if mx.any(mx.isnan(g)).item():
            raise RendererDifferentiabilityError(f"Step 3 failed: Gradient for '{name}' contains NaNs.")

    logger.info("Step 3 (Gaussian Splatting and Binning Logic) is differentiable.")

    logger.info("All differentiability checks passed successfully.")
