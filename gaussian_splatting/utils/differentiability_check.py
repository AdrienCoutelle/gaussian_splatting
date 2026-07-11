import mlx.core as mx

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderer.renderer import Renderer, RendererConfig, ScreenSpaceGaussians


def debug_renderer_differentiability(renderer) -> None:
    print("=== STARTING STEP-BY-STEP DIFFERENTIABILITY DIAGNOSTIC ===\n")

    # 1. Setup Camera
    try:
        camera = Camera(
            pose=mx.eye(4),
            width=renderer.config.width,
            height=renderer.config.height,
            focal_length=renderer.config.focal_length,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize Camera. Verify your Camera constructor: {e}")
        return

    # 2. Setup Base Test Inputs
    positions = mx.array([[0.0, 0.0, -3.0]], dtype=mx.float32)
    quaternions = mx.array([[1.0, 0.0, 0.0, 0.0]], dtype=mx.float32)
    scales = mx.array([[-1.0, -1.0, -1.0]], dtype=mx.float32)
    sh_coeffs = mx.zeros((1, 16, 3), dtype=mx.float32)
    opacities = mx.array([[0.9]], dtype=mx.float32)

    # =========================================================================
    # DIAGNOSTIC 1: _transform_positions_to_camera_space
    # =========================================================================
    print("Checking Step 1: Position Transformation...")
    try:

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

        if step1_grads[0] is not None and not mx.any(mx.isnan(step1_grads[0])).item():
            print("  ✅ Step 1 (Transform to Camera Space) is differentiable.\n")
        else:
            print("  ❌ Step 1 failed. Gradients are None or NaN.\n")
    except Exception as e:
        print(f"  ❌ Step 1 crashed with error: {e}\n")

    # =========================================================================
    # DIAGNOSTIC 2: _project_to_screen_space
    # =========================================================================
    print("Checking Step 2: Screen Space Projection...")
    try:

        def step2_loss(p, q, s, sh, o):
            g = GaussianCollection.from_tensors(
                positions=p,
                quaternions=q,
                scales=s,
                sh_coeffs=sh,
                opacities=o,
            )
            # Re-run transformation in-place
            renderer._transform_positions_to_camera_space(camera, g)
            screen_space = renderer._project_to_screen_space(camera, g)

            if screen_space is None:
                return mx.array(0.0)

            # Combine all output fields to check backward path on all variables
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

        step2_passed = True
        param_names = ["positions", "quaternions", "scales", "sh_coeffs", "opacities"]
        for name, g in zip(param_names, step2_grads):
            if g is None or mx.any(mx.isnan(g)).item():
                print(f"  ❌ Step 2 failed for input: {name} (None or NaN gradient)")
                step2_passed = False
        if step2_passed:
            print("  ✅ Step 2 (Projection to Screen Space) is differentiable.\n")
        else:
            print("")
    except Exception as e:
        print(f"  ❌ Step 2 crashed with error: {e}\n")

    # =========================================================================
    # DIAGNOSTIC 3: _splat_gaussians
    # =========================================================================
    print("Checking Step 3: Gaussian Splatting and Binning Logic...")
    try:
        # Use clean mock variables to isolate this test from Step 2 failures
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

        step3_passed = True
        ssg_fields = ["means_2d", "covariances_2d", "colors", "opacities"]
        for name, g in zip(ssg_fields, step3_grads):
            if g is None or mx.any(mx.isnan(g)).item():
                print(f"  ❌ Step 3 failed for screen space variable: {name}")
                step3_passed = False
        if step3_passed:
            print("  ✅ Step 3 (_splat_gaussians) is differentiable.\n")
        else:
            print("")
    except Exception as e:
        print(f"  ❌ Step 3 crashed with error: {e}\n")

    print("=== DIAGNOSTIC COMPLETE ===")


# Run diagnostic
renderer_config = RendererConfig(
    width=64,
    height=64,
    focal_length=50.0,
)
renderer = Renderer(renderer_config)
debug_renderer_differentiability(renderer)
