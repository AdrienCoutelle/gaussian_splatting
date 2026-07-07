import os
from typing import Annotated

import cv2
import mlx.core as mx
import mlx.optimizers as opt
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.dataset import GaussianSplattingDataset
from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.structures.renderer.rasterizer import Rasterizer
from gaussian_splatting.structures.renderer.renderer import Renderer
from gaussian_splatting.utils.logger import Logger
from gaussian_splatting.utils.ply.ply_saver import PLYSaver
from gaussian_splatting.utils.profiler import profile

logger = Logger("TRAINER")


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: Annotated[int, Field(gt=0)]
    learning_rate: float = 1e-3
    save_every_n_epochs: int = 10
    render_scale: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    gradient_accumulation_steps: Annotated[int, Field(gt=0)] = 1


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

        os.makedirs(output_folder, exist_ok=True)

        # Disable caching entirely using the non-deprecated MLX API
        mx.set_cache_limit(0)

        # 16GB M1 Safety Limits:
        # Capping the batch size at 128 and tiles at 512 keeps loop allocations small.
        self.renderer.config.max_gaussians_per_batch = min(self.renderer.config.max_gaussians_per_batch, 128)
        self.renderer.config.max_gaussians_per_tile = min(self.renderer.config.max_gaussians_per_tile, 512)

        self.renderer.rasterizer = Rasterizer(
            gaussian_extent=self.renderer.config.gaussian_extent,
            tile_size=self.renderer.config.tile_size,
            max_gaussians_per_batch=self.renderer.config.max_gaussians_per_batch,
        )

        self.params = {
            "positions": mx.array(gaussians_collection.positions, dtype=mx.float16),
            "quaternions": mx.array(gaussians_collection.quaternions, dtype=mx.float16),
            "scales": mx.array(gaussians_collection.scales, dtype=mx.float16),
            "sh_coeffs": mx.array(gaussians_collection.sh_coeffs, dtype=mx.float16),
            "opacities": mx.array(gaussians_collection.opacities, dtype=mx.float16),
        }

        self.optimizer = opt.Adam(learning_rate=configuration.learning_rate)

    def _build_gaussian_collection(self, params: dict[str, mx.array]) -> GaussianCollection:
        return GaussianCollection.from_tensors(
            positions=params["positions"],
            quaternions=params["quaternions"],
            scales=params["scales"],
            sh_coeffs=params["sh_coeffs"],
            opacities=params["opacities"],
        )

    def _load_training_image(
        self,
        image_name: str,
        target_height: int,
        target_width: int,
    ) -> mx.array | None:
        for ext in [".png", ".jpg", ".jpeg"]:
            path = os.path.join(self.dataset.images_folder_path, image_name + ext)
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != target_height or img.shape[1] != target_width:
                img = cv2.resize(img, (target_width, target_height))
            return mx.array(img / 255.0, dtype=mx.float16)
        return None

    def _save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        checkpoint_path = os.path.join(self.output_folder, f"checkpoint_epoch_{epoch:04d}.ply")
        gaussians_collection = self._build_gaussian_collection(self.params)
        PLYSaver(checkpoint_path).save_gaussians(gaussians_collection)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def run(self) -> None:
        logger.info(f"Starting training for {self.configuration.epochs} epochs over {len(self.dataset)} cameras.")

        camera_items = list(self.dataset.items)
        scale = self.configuration.render_scale

        def render_from_params(params: dict[str, mx.array], camera: Camera) -> mx.array:
            gaussians = self._build_gaussian_collection(params)
            return self.renderer.render_tensor(camera=camera, gaussians=gaussians)

        # Gradient checkpointing to bypass holding backward activations in memory
        checkpointed_render = mx.checkpoint(render_from_params)

        def loss_fn(params: dict[str, mx.array], camera: Camera, gt_image: mx.array, valid_idx: mx.array) -> mx.array:
            sliced_params = {k: v[valid_idx] for k, v in params.items()}
            rendered = checkpointed_render(sliced_params, camera)
            return mx.mean(mx.abs(rendered - gt_image))

        loss_and_grad_fn = mx.value_and_grad(loss_fn)

        for epoch in tqdm(range(self.configuration.epochs), desc="Training"):
            epoch_loss = 0.0
            num_rendered = 0

            indices = mx.random.permutation(len(camera_items)).tolist()
            accum_steps = self.configuration.gradient_accumulation_steps

            accumulated_grads = {k: mx.zeros_like(v) for k, v in self.params.items()}

            for step, idx in enumerate(indices):
                logger.info(
                    f"Epoch {epoch + 1}/{self.configuration.epochs} — rendering camera {idx + 1}/{len(camera_items)}"
                )
                image_name, camera = camera_items[idx]

                render_h = max(1, round(camera.h * scale))
                render_w = max(1, round(camera.w * scale))

                # Safe resolution capping for 16GB limits (~400x300 maximum footprint)
                # This drops intermediate matrices from ~500MB to ~60MB.
                max_pixels = 400 * 300
                current_pixels = render_h * render_w
                if current_pixels > max_pixels:
                    adjust_scale = np.sqrt(max_pixels / current_pixels)
                    render_h = max(1, round(render_h * adjust_scale))
                    render_w = max(1, round(render_w * adjust_scale))
                    active_focal_length = camera.focal_length * scale * adjust_scale
                else:
                    active_focal_length = camera.focal_length * scale

                render_camera = Camera(
                    pose=camera.pose,
                    focal_length=active_focal_length,
                    width=render_w,
                    height=render_h,
                )

                gt_image = self._load_training_image(
                    image_name=image_name,
                    target_height=render_h,
                    target_width=render_w,
                )
                if gt_image is None:
                    continue

                # Eager Frustum Culling (executed outside of computation graph)
                positions = self.params["positions"]
                r_world_to_camera = mx.array(render_camera.pose[:3, :3].T)
                camera_center = mx.array(render_camera.pose[:3, 3:4])

                mx.eval(positions)  # Ensure coordinates are evaluated before calculation

                cam_positions = (r_world_to_camera @ (positions.T - camera_center)).T
                depths = -cam_positions[:, 2]

                # Keep Gaussians within depth window
                depth_mask = (depths > 0.1) & (depths < 20.0)

                # Keep Gaussians projected within or near screen boundaries
                principal_point_x, principal_point_y = render_camera.principal_point
                means_2d_x = active_focal_length * (cam_positions[:, 0] / mx.maximum(depths, 1e-4)) + principal_point_x
                means_2d_y = -active_focal_length * (cam_positions[:, 1] / mx.maximum(depths, 1e-4)) + principal_point_y

                margin = 80.0
                frustum_mask = (
                    (means_2d_x >= -margin)
                    & (means_2d_x <= render_camera.width + margin)
                    & (means_2d_y >= -margin)
                    & (means_2d_y <= render_camera.height + margin)
                )

                valid_mask = depth_mask & frustum_mask

                # Convert the boolean mask to NumPy to extract active indices eagerly,
                # then return them as an MLX integer index array.
                valid_mask_np = np.array(valid_mask)
                valid_idx_np = np.where(valid_mask_np)[0]

                valid_idx = mx.array(valid_idx_np, dtype=mx.int32)
                mx.eval(valid_idx)  # Materialize index list

                logger.info(f"Culling: keeping {valid_idx.shape[0]} / {positions.shape[0]} visible Gaussians.")

                if valid_idx.shape[0] == 0:
                    continue

                # Forward and backward pass using active parameters
                loss, grads = loss_and_grad_fn(self.params, render_camera, gt_image, valid_idx)

                for k in self.params:
                    accumulated_grads[k] = accumulated_grads[k] + grads[k] / accum_steps

                mx.eval(loss, accumulated_grads)

                scaled_loss_val = loss.item()
                logger.info(f"Epoch {epoch + 1}/{self.configuration.epochs} — L1 loss: {scaled_loss_val:.6f}")

                is_last_step = step == len(indices) - 1
                if (step + 1) % accum_steps == 0 or is_last_step:
                    self.optimizer.update(self.params, accumulated_grads)
                    mx.eval(self.params, self.optimizer.state)

                    accumulated_grads = {k: mx.zeros_like(v) for k, v in self.params.items()}
                    mx.eval(accumulated_grads)
                    logger.info("Ran backpropagation and optimizer step.")

                epoch_loss += scaled_loss_val
                num_rendered += 1

                del loss, grads, gt_image, valid_idx
                mx.metal.clear_cache()

            if num_rendered > 0:
                avg_loss = epoch_loss / num_rendered
                logger.info(f"Epoch {epoch + 1}/{self.configuration.epochs} — avg L1 loss: {avg_loss:.6f}")

            if (epoch + 1) % self.configuration.save_every_n_epochs == 0:
                self._save_checkpoint(epoch + 1)

            logger.info(f"Completed epoch {epoch + 1}/{self.configuration.epochs}.")

        self._save_checkpoint(self.configuration.epochs)
