# import mlx.core as mx

# from gaussian_splatting.structures.camera import Camera
# from gaussian_splatting.structures.gaussian import GaussianCollection
# from gaussian_splatting.structures.renderer.renderer import Renderer, RendererConfig


# def assert_renderer_differentiable(renderer) -> None:
#     """
#     Verifies that the renderer's computational graph is fully differentiable.
#     Generates internal dummy structures and raises a RuntimeError if gradients
#     are missing, non-finite, or disconnected.
#     """
#     print("[INFO] Initializing differentiability check...")

#     try:
#         pose = mx.eye(4)

#         camera = Camera(
#             pose=pose,
#             width=renderer.config.width,
#             height=renderer.config.height,
#             focal_length=renderer.config.focal_length,
#         )
#     except Exception as e:
#         raise RuntimeError(f"Failed to initialize Camera for the test. Verify your Camera signature. Error: {e}")

#     # 2. Define differentiable inputs for a single Gaussian positioned in front of the camera
#     positions = mx.array([[0.0, 0.0, -3.0]], dtype=mx.float32)  # 3 units in front
#     quaternions = mx.array([[1.0, 0.0, 0.0, 0.0]], dtype=mx.float32)
#     scales = mx.array([[-1.0, -1.0, -1.0]], dtype=mx.float32)
#     sh_coeffs = mx.zeros((1, 16, 3), dtype=mx.float32)
#     opacities = mx.array([[0.9]], dtype=mx.float32)

#     # 3. Define a simple loss function over the rendered output
#     def loss_fn(p, q, s, sh, o):
#         gaussians = GaussianCollection.from_tensors(
#             positions=p,
#             quaternions=q,
#             scales=s,
#             sh_coeffs=sh,
#             opacities=o,
#         )
#         rendered = renderer.render_tensor(camera=camera, gaussians=gaussians)
#         # Using the mean of the output tensor as a target-free loss
#         return mx.mean(rendered)

#     grad_fn = mx.value_and_grad(loss_fn, argnums=[0, 1, 2, 3, 4])

#     try:
#         loss_val, grads = grad_fn(positions, quaternions, scales, sh_coeffs, opacities)
#         mx.eval(loss_val, grads)
#     except Exception as e:
#         raise RuntimeError(f"Execution failed during the forward or backward pass: {e}")

#     # 6. Verify gradient validity
#     param_names = ["positions", "quaternions", "scales", "sh_coeffs", "opacities"]
#     failed_params = []

#     for name, grad in zip(param_names, grads):
#         if grad is None:
#             print(f"[ERROR] Gradient for '{name}' is None (disconnected graph).")
#             failed_params.append(name)
#         elif mx.any(mx.isnan(grad)).item():
#             print(f"[ERROR] Gradient for '{name}' contains NaNs.")
#             failed_params.append(name)
#         elif mx.max(mx.abs(grad)).item() == 0.0:
#             print(f"[WARNING] Gradient for '{name}' is zero. Verify projection boundaries if unexpected.")
#         else:
#             grad_norm = mx.linalg.norm(grad).item()
#             print(f"[SUCCESS] '{name}' gradient is valid. Norm: {grad_norm:.6f}")

#     if failed_params:
#         raise RuntimeError(f"Renderer is not fully differentiable. Broken parameter paths: {failed_params}")

#     print("[SUCCESS] All gradient checks passed. The renderer is differentiable.")


# renderer_config = RendererConfig(
#     width=64,
#     height=64,
#     focal_length=50.0,
# )

# renderer = Renderer(renderer_config)

# assert_renderer_differentiable(renderer)

import mlx.core as mx

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderer.rasterizer import Rasterizer
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
    # DIAGNOSTIC 3: _splat_gaussians_vectorized
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
            sorted_indices = mx.array([0], dtype=mx.int32)
            out = renderer._splat_gaussians_vectorized(
                gaussians=ssg,
                sorted_indices=sorted_indices,
                image_height=camera.h,
                image_width=camera.w,
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
            print("  ✅ Step 3 (_splat_gaussians_vectorized) is differentiable.\n")
        else:
            print("")
    except Exception as e:
        print(f"  ❌ Step 3 crashed with error: {e}\n")

    # =========================================================================
    # DIAGNOSTIC 4: Raw Rasterizer Object
    # =========================================================================
    print("Checking Step 4: Raw Rasterizer Custom Kernel Integration...")
    try:
        # Mock values directly matching Rasterizer interface
        xy = mx.array([[32.0, 32.0]], dtype=mx.float32)
        conic = mx.array([[0.5, 0.0, 0.5]], dtype=mx.float32)
        opacity = mx.array([0.9], dtype=mx.float32)
        color = mx.array([[1.0, 0.5, 0.2]], dtype=mx.float32)

        def step4_loss(x, c, o, col):
            out = Rasterizer().rasterize(
                gauss_xy=x,
                gauss_conic=c,
                gauss_opacity=o,
                gauss_color=col,
                tile_origins=mx.array([[0, 0]], dtype=mx.uint32),
                tile_gstart=mx.array([0], dtype=mx.uint32),
                tile_gcount=mx.array([1], dtype=mx.uint32),
                image_width=renderer.config.width,
                image_height=renderer.config.height,
                tile_size=renderer.config.tile_size,
                sigma_cut=renderer.config.sigma_cut,
                eps=renderer.config.eps,
            )
            return mx.mean(out)

        _, step4_grads = mx.value_and_grad(step4_loss, argnums=[0, 1, 2, 3])(xy, conic, opacity, color)
        mx.eval(step4_grads)
        print("  ✅ Step 4 (Raw Rasterizer) is differentiable.\n")
    except Exception as e:
        print(f"  ❌ Step 4 crashed with error: {e}\n")

    print("=== DIAGNOSTIC COMPLETE ===")


# Run diagnostic
renderer_config = RendererConfig(
    width=64,
    height=64,
    focal_length=50.0,
)
renderer = Renderer(renderer_config)
debug_renderer_differentiability(renderer)
