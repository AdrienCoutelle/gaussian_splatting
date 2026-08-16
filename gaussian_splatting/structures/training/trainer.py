import datetime
import os
from typing import Annotated

import mlx.core as mx
import mlx.optimizers as opt
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from gaussian_splatting.structures.dataset import GaussianSplattingDataset
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderer.renderer import Renderer
from gaussian_splatting.utils.differentiability_check import check_renderer_differentiability
from gaussian_splatting.utils.image import stack_images_horizontally
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply.ply_saver import PLYSaver
from gaussian_splatting.utils.profiler import profile
from gaussian_splatting.utils.tensorboard import TensorBoardWriter

logger = Logger("TRAINER")


class LearningRatesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr_positions: float
    lr_opacities: float
    lr_scales: float
    lr_quaternions: float
    lr_sh_coeffs: float


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: Annotated[int, Field(gt=0)]
    learning_rates: LearningRatesConfig
    save_every_n_epochs: int = 50
    render_scale: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    gradient_accumulation_steps: Annotated[int, Field(gt=0)] = 1
    max_gaussians_per_step: int = 0  # 0 = no limit; positive = random subsample per step
    log_every_n_epochs: int = 5  # log metrics + image to TensorBoard every N epochs

    # Simplified Densification Settings
    densification_interval: int = 10  # Run densification every N steps
    prune_opacity_threshold: float = 0.005
    densify_grad_threshold: float = 0.0002
    split_scale_threshold: float = 0.01


@profile
class Trainer:
    def __init__(
        self,
        gaussians_collection: GaussianCollection,
        renderer: Renderer,
        dataset: GaussianSplattingDataset,
        output_folder: str,
        configuration: TrainerConfig,
    ) -> None:
        self.renderer = renderer
        self.dataset = dataset
        self.output_folder = output_folder
        self.configuration = configuration

        check_renderer_differentiability(self.renderer)

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, "checkpoints"))

        tb_log_dir = os.path.join(output_folder, "tensorboard")
        self.tensorboard_writer = TensorBoardWriter(tb_log_dir)
        logger.info(f"TensorBoard logs: {tb_log_dir}")

        self.params = {
            "positions": mx.array(gaussians_collection.positions, dtype=mx.float32),
            "quaternions": mx.array(gaussians_collection.quaternions, dtype=mx.float32),
            "scales": mx.array(gaussians_collection.scales, dtype=mx.float32),
            "sh_coeffs": mx.array(gaussians_collection.sh_coeffs, dtype=mx.float32),
            "opacities": mx.array(gaussians_collection.opacities, dtype=mx.float32),
        }

        self.optimizers = {
            "positions": opt.Adam(learning_rate=configuration.learning_rates.lr_positions),
            "opacities": opt.Adam(learning_rate=configuration.learning_rates.lr_opacities),
            "scales": opt.Adam(learning_rate=configuration.learning_rates.lr_scales),
            "quaternions": opt.Adam(learning_rate=configuration.learning_rates.lr_quaternions),
            "sh_coeffs": opt.Adam(learning_rate=configuration.learning_rates.lr_sh_coeffs),
        }

        # Gradient and step accumulation variables
        num_gaussians = self.params["positions"].shape[0]
        self.pos_grad_accum = mx.zeros((num_gaussians,), dtype=mx.float32)
        self.denom = mx.zeros((num_gaussians,), dtype=mx.float32)

        # Track maximum screen space size in pixels for each Gaussian
        self.max_pixel_sizes = mx.zeros((num_gaussians,), dtype=mx.float32)

        self.step_count = 0

    def run(self) -> None:
        logger.info(f"Starting training for {self.configuration.epochs} epochs over {len(self.dataset)} cameras.")

        for epoch in tqdm(range(1, self.configuration.epochs + 1), desc="Training"):
            avg_loss = self._run_epoch(epoch)

            self.tensorboard_writer.log_scalar("Loss/train_epoch", avg_loss, epoch)

            self._run_validation(epoch)

            if epoch % self.configuration.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)

        self.tensorboard_writer.close()

    def _save_checkpoint(
        self,
        epoch: int | None = None,
    ) -> None:
        gaussians = self._build_gaussian_collection(self.params)

        file_name = (
            f"checkpoint_epoch_{epoch}"
            if epoch is not None
            else "checkpoint_final"
        )  # fmt:skip

        ply_saver = PLYSaver(os.path.join(self.output_folder, "checkpoints", f"{file_name}.ply"))
        ply_saver.save_gaussians(gaussians)

    def _run_validation(
        self,
        epoch: int,
    ) -> None:
        gt_image, camera = self.dataset.validation_item

        gaussians = self._build_gaussian_collection(self.params)

        t0 = datetime.datetime.now()
        image = self.renderer.render_tensor(
            camera=camera,
            gaussians=gaussians,
        )
        mx.eval(image)
        render_time = (datetime.datetime.now() - t0).total_seconds()

        val_loss = self._loss_fn(image, gt_image)
        stacked_validation_image = stack_images_horizontally(
            left_image=np.array(gt_image),
            right_image=np.array(image),
        )

        self.tensorboard_writer.log_scalar("Loss/validation", val_loss.item(), epoch)
        self.tensorboard_writer.log_scalar("Stats/num_gaussians(kilo)", self.params["positions"].shape[0] / 1000, epoch)
        self.tensorboard_writer.log_scalar("Stats/render_time", render_time, epoch)
        self.tensorboard_writer.log_image("Validation/Image", stacked_validation_image, epoch)

    def _estimate_pixel_sizes(self, positions: mx.array, scales: mx.array, camera) -> mx.array:
        """Estimates the projected screen space size (radius) of Gaussians in pixels."""
        # 1. Resolve camera center coordinates in world space
        if hasattr(camera, "camera_center"):
            cam_center = mx.array(camera.camera_center, dtype=mx.float32)
        elif hasattr(camera, "position"):
            cam_center = mx.array(camera.position, dtype=mx.float32)
        else:
            cam_center = mx.zeros((3,), dtype=mx.float32)

        # 2. Get focal length in pixels
        if hasattr(camera, "focal_x"):
            focal = float(camera.focal_x)
        elif hasattr(camera, "FocalX"):
            focal = float(camera.FocalX)
        elif hasattr(camera, "fov_x") and hasattr(camera, "image_width"):
            fov_x = float(camera.fov_x)
            # If fov is in degrees, convert to radians
            if fov_x > 3.14159:
                fov_x = fov_x * np.pi / 180.0
            width = float(camera.image_width)
            focal = width / (2.0 * np.tan(fov_x / 2.0))
        else:
            focal = 60.0  # Fallback value

        # 3. Calculate distance (depth approximation) to camera center
        delta = positions - cam_center
        depths = mx.linalg.norm(delta, axis=-1)
        depths = mx.maximum(depths, 1e-5)  # Avoid division by zero

        # 4. Extract maximum scale value from 3D log scales
        max_scale_3d = mx.exp(mx.max(scales, axis=-1))

        # 5. Approximate screen space radius in pixels: (3D scale * focal length) / depth
        pixel_sizes = (max_scale_3d * focal) / depths
        return pixel_sizes

    def _run_epoch(
        self,
        epoch: int,
    ) -> float:
        indices = mx.random.permutation(len(self.dataset)).tolist()

        epoch_loss = 0.0
        num_rendered = 0

        def loss_fn(params, camera, gt_image):
            gaussians = self._build_gaussian_collection(params)
            image = self.renderer.render_tensor(
                camera=camera,
                gaussians=gaussians,
            )
            return self._loss_fn(image, gt_image)

        loss_and_grad_fn = mx.value_and_grad(loss_fn)

        for idx in tqdm(indices, desc=f"Running epoch {epoch}", leave=False):
            self.step_count += 1
            gt_image, camera = self.dataset[idx]

            loss, grads = loss_and_grad_fn(self.params, camera, gt_image)

            mx.eval(loss)

            epoch_loss += loss.item()
            num_rendered += 1

            for k, optimizer in self.optimizers.items():
                # Wrap the parameter and gradient in single-item dictionaries
                param_dict = {k: self.params[k]}
                grad_dict = {k: grads[k]}

                # MLX modifies param_dict in-place using its .update() method
                optimizer.update(param_dict, grad_dict)

                # Store the updated array back into your parameters
                self.params[k] = param_dict[k]

            # Estimate pixel sizes for the current view and track the maximum observed size
            current_pixel_sizes = self._estimate_pixel_sizes(self.params["positions"], self.params["scales"], camera)
            self.max_pixel_sizes = mx.maximum(self.max_pixel_sizes, current_pixel_sizes)

            mx.eval(self.params, self.max_pixel_sizes)

            # Accumulate spatial position gradient norms
            pos_grads = grads["positions"]
            grad_norms = mx.linalg.norm(pos_grads, axis=-1)
            self.pos_grad_accum = self.pos_grad_accum + grad_norms
            self.denom = self.denom + 1.0

            # Perform densification/pruning check every N steps
            if self.step_count % self.configuration.densification_interval == 0:
                self._refine_gaussians()

        return epoch_loss / num_rendered if num_rendered > 0 else 0.0

    def _refine_gaussians(self) -> None:
        N = self.params["positions"].shape[0]
        grads_norm = self.pos_grad_accum / mx.maximum(self.denom, 1.0)

        # 1. Identify Gaussians to prune
        opacity_prune = self.params["opacities"].squeeze() < self.configuration.prune_opacity_threshold

        # New size-based conditions
        too_big = self.max_pixel_sizes > 20.0
        too_small = self.max_pixel_sizes < 0.1

        should_prune = opacity_prune | too_big | too_small

        # 2. Identify Gaussians to clone or split based on average gradient
        should_densify = grads_norm > self.configuration.densify_grad_threshold

        # Assume scales are stored as log-scales (standard in 3DGS)
        split_cond = mx.max(mx.exp(self.params["scales"]), axis=-1) > self.configuration.split_scale_threshold

        should_split = should_densify & split_cond & ~should_prune
        should_clone = should_densify & ~split_cond & ~should_prune

        # Convert masks to NumPy to bypass MLX boolean indexing limitations
        should_prune_np = np.array(should_prune)
        should_split_np = np.array(should_split)
        should_clone_np = np.array(should_clone)

        indices_seq_np = np.arange(N)
        keep_idx_np = indices_seq_np[~should_prune_np & ~should_split_np]
        clone_idx_np = indices_seq_np[should_clone_np]
        split_idx_np = indices_seq_np[should_split_np]

        # Convert back to MLX arrays
        keep_idx = mx.array(keep_idx_np, dtype=mx.int32)
        clone_idx = mx.array(clone_idx_np, dtype=mx.int32)
        split_idx = mx.array(split_idx_np, dtype=mx.int32)

        num_keep = keep_idx.shape[0]
        num_clone = clone_idx.shape[0]
        num_split = split_idx.shape[0]

        # Combine indices to index existing tensors
        new_indices = mx.concat([keep_idx, clone_idx, split_idx, split_idx], axis=0)

        split_start_1 = num_keep + num_clone
        split_end_1 = split_start_1 + num_split
        split_start_2 = split_end_1
        split_end_2 = split_start_2 + num_split

        new_params = {}
        for key, val in self.params.items():
            new_params[key] = val[new_indices]

        # Split execution and scale adjustment
        if num_split > 0:
            scale_reduction = mx.log(mx.array(1.6))
            new_params["scales"] = mx.concatenate(
                [
                    new_params["scales"][:split_start_1],
                    new_params["scales"][split_start_1:split_end_1] - scale_reduction,
                    new_params["scales"][split_start_2:split_end_2] - scale_reduction,
                ],
                axis=0,
            )

            # Sample random noise for positional offset
            noise1 = mx.random.normal(shape=(num_split, 3))
            noise2 = mx.random.normal(shape=(num_split, 3))

            split_scales = mx.exp(self.params["scales"][split_idx])
            scaled_noise1 = noise1 * split_scales
            scaled_noise2 = noise2 * split_scales

            split_quats = new_params["quaternions"][split_start_1:split_end_1]
            split_quats = split_quats / mx.linalg.norm(split_quats, axis=-1, keepdims=True)

            def rotate_by_quaternion(v: mx.array, q: mx.array) -> mx.array:
                w = q[:, 0:1]
                xyz = q[:, 1:]

                def cross_product(a, b):
                    return mx.stack(
                        [
                            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
                        ],
                        axis=-1,
                    )

                cross_xyz_v = cross_product(xyz, v)
                return v + 2.0 * cross_product(xyz, cross_xyz_v + w * v)

            offset1 = rotate_by_quaternion(scaled_noise1, split_quats)
            offset2 = rotate_by_quaternion(scaled_noise2, split_quats)

            new_params["positions"] = mx.concatenate(
                [
                    new_params["positions"][:split_start_1],
                    new_params["positions"][split_start_1:split_end_1] + offset1,
                    new_params["positions"][split_start_2:split_end_2] + offset2,
                ],
                axis=0,
            )

        # Apply newly built parameter set
        self.params = new_params
        new_num_gaussians = self.params["positions"].shape[0]

        # Reset tracking registers for the modified shape
        self.pos_grad_accum = mx.zeros((new_num_gaussians,), dtype=mx.float32)
        self.denom = mx.zeros((new_num_gaussians,), dtype=mx.float32)
        self.max_pixel_sizes = mx.zeros((new_num_gaussians,), dtype=mx.float32)

        # Recursive updater to process the nested optimizer states dynamically
        def update_state(state, indices, original_size):
            if isinstance(state, mx.array):
                if state.ndim > 0 and state.shape[0] == original_size:
                    return state[indices]
                return state
            elif isinstance(state, dict):
                return {k: update_state(v, indices, original_size) for k, v in state.items()}
            elif isinstance(state, list):
                return [update_state(v, indices, original_size) for v in state]
            return state

        for k, optimizer in self.optimizers.items():
            optimizer.state = update_state(optimizer.state, new_indices, N)

        mx.eval(self.params, self.max_pixel_sizes)

    def _loss_fn(
        self,
        image: mx.array,
        gt_image: mx.array,
    ) -> mx.array:
        return mx.mean(mx.abs(image - gt_image))

    def _build_gaussian_collection(
        self,
        params: dict[str, mx.array],
    ) -> GaussianCollection:
        return GaussianCollection.from_tensors(
            positions=params["positions"],
            quaternions=params["quaternions"],
            scales=params["scales"],
            sh_coeffs=params["sh_coeffs"],
            opacities=params["opacities"],
        )
