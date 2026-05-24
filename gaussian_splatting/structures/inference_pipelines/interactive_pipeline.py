from dataclasses import dataclass

import cv2
import numpy as np
import pygame
import torch

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.gaussian import Gaussian
from gaussian_splatting.structures.inference_pipelines.base_pipeline import (
    BaseInferencePipeline,
    InferencePipelineParams,
)
from gaussian_splatting.structures.renderers.base_renderer import BaseRenderer


class StopInferenceException(Exception):
    """Raised when the user wants to stop the interactive inference."""

    pass


class InteractiveInputHandler:
    def __init__(
        self,
        initial_position: list[float],
        initial_look_at: list[float],
        width: int,
        height: int,
        mouse_sensitivity: float = 0.0025,
        movement_speed: float = 0.05,
    ):
        self.position = np.array(initial_position, dtype=np.float32)
        self.look_at = np.array(initial_look_at, dtype=np.float32)

        forward = self.look_at - self.position
        forward = forward / np.linalg.norm(forward)
        self.yaw = np.arctan2(forward[1], forward[0])
        self.pitch = np.arcsin(forward[2])

        self.mouse_sensitivity = mouse_sensitivity
        self.movement_speed = movement_speed

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Gaussian Splatting Interactive Viewer")
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        self.clock = pygame.time.Clock()
        self.frame_count = 0
        self._cached_pose: torch.Tensor | None = None

        print(f"Initial pitch: {np.degrees(self.pitch):.1f}°, yaw: {np.degrees(self.yaw):.1f}°")

    def get_next_pose(
        self,
        compute_pose_fn: callable,
    ) -> tuple[torch.Tensor, pygame.Surface, bool]:
        """
        Process input events and return the updated pose.

        Args:
            compute_pose_fn: Function that computes pose from position, look_at, and world_up

        Returns:
            Tuple of (pose tensor, pygame surface to render to, pose changed flag)

        Raises:
            StopInferenceException: When the user wants to quit
        """
        pose_changed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise StopInferenceException("User closed the window")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise StopInferenceException("User pressed ESC")
            elif event.type == pygame.MOUSEMOTION:
                pose_changed = (
                    self._handle_mouse_movement(
                        dx=event.rel[0],
                        dy=event.rel[1],
                    )
                    or pose_changed
                )
            elif event.type == pygame.MOUSEWHEEL:
                pose_changed = self._handle_forward_movement(step=float(event.y)) or pose_changed

        pose_changed = self._handle_keyboard_input() or pose_changed

        if pose_changed:
            self._update_look_at()

        if (
            pose_changed
            or self._cached_pose is None
        ):  # fmt:skip
            self._cached_pose = torch.from_numpy(
                compute_pose_fn(
                    position=self.position,
                    look_at=self.look_at,
                    world_up=np.array([0, 0, 1]),
                )
            )

        return self._cached_pose, self.screen, pose_changed

    def update_display(self) -> None:
        """Update the display and tick the clock."""
        pygame.display.flip()
        self.clock.tick(60)

        self.frame_count += 1
        if self.frame_count % 60 == 0:
            fps = self.clock.get_fps()
            print(f"FPS: {fps:.1f} | Pitch: {np.degrees(self.pitch):.1f}° | Yaw: {np.degrees(self.yaw):.1f}°")

    def cleanup(self) -> None:
        """Clean up pygame resources."""
        pygame.quit()

    def _handle_mouse_movement(
        self,
        dx: int,
        dy: int,
    ) -> bool:
        if (
            dx == 0
            and dy == 0
        ):  # fmt:skip
            return False

        self.yaw -= dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity

        max_pitch = np.pi / 2 - 0.001
        self.pitch = np.clip(self.pitch, -max_pitch, max_pitch)
        return True

    def _update_look_at(self) -> None:
        forward = np.array(
            [
                np.cos(self.pitch) * np.cos(self.yaw),
                np.cos(self.pitch) * np.sin(self.yaw),
                np.sin(self.pitch),
            ],
            dtype=np.float32,
        )
        self.look_at = self.position + forward

    def _handle_forward_movement(
        self,
        step: float,
    ) -> bool:
        if step == 0.0:
            return False

        forward = self.look_at - self.position
        forward = forward / np.linalg.norm(forward)
        self.position += forward * self.movement_speed * step
        return True

    def _handle_keyboard_input(self) -> bool:
        keys = pygame.key.get_pressed()
        moved = False

        forward = self.look_at - self.position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, np.array([0, 0, 1]))
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-6:
            right = right / right_norm
        else:
            right = np.array([1, 0, 0])

        up = np.cross(right, forward)

        if keys[pygame.K_f]:
            self.position += forward * self.movement_speed
            moved = True
        if keys[pygame.K_b]:
            self.position -= forward * self.movement_speed
            moved = True
        if keys[pygame.K_LEFT]:
            self.position -= right * self.movement_speed
            moved = True
        if keys[pygame.K_RIGHT]:
            self.position += right * self.movement_speed
            moved = True
        if keys[pygame.K_UP]:
            self.position += up * self.movement_speed
            moved = True
        if keys[pygame.K_DOWN]:
            self.position -= up * self.movement_speed
            moved = True

        return moved


@dataclass
class InteractiveInferencePipelineParams(InferencePipelineParams):
    initial_look_at: list[float]
    initial_position: list[float]

    @staticmethod
    def from_dict(configuration: dict) -> "InteractiveInferencePipelineParams":
        if not isinstance(configuration, dict):
            raise ValueError(
                f"InteractiveInferencePipelineParams must be a dictionary, got '{type(configuration).__name__}'."
            )

        mandatory_fields = {
            "initial_look_at",
            "initial_position",
        }

        if not set(configuration.keys()).issuperset(mandatory_fields):
            missing_fields = mandatory_fields - set(configuration.keys())
            raise ValueError(
                f"InteractiveInferencePipelineParams is missing the following mandatory fields: {missing_fields}, "
                f"got {set(configuration.keys())}."
            )

        look_at = configuration["initial_look_at"]
        if not isinstance(look_at, list) or len(look_at) != 3 or not all(isinstance(c, (int, float)) for c in look_at):
            raise ValueError(
                f"InteractiveInferencePipelineParams 'initial_look_at' "
                f"must be a list of three numbers, got '{look_at}'."
            )

        position = configuration["initial_position"]
        if (
            not isinstance(position, list)
            or len(position) != 3
            or not all(isinstance(c, (int, float)) for c in position)
        ):
            raise ValueError(
                f"InteractiveInferencePipelineParams 'initial_position' "
                f"must be a list of three numbers, got '{position}'."
            )

        return InteractiveInferencePipelineParams(
            initial_look_at=look_at,
            initial_position=position,
        )


class InteractiveInferencePipeline(BaseInferencePipeline):
    def __init__(
        self,
        renderer: BaseRenderer,
        gaussians: list[Gaussian],
        configuration: InteractiveInferencePipelineParams,
        device: torch.device,
        output_folder: str,
        epoch: int | None = None,
    ):
        super().__init__(
            configuration=configuration,
            device=device,
            output_folder=output_folder,
            epoch=epoch,
        )

        self.renderer = renderer
        self.gaussians = gaussians
        self.device = device

        self.input_handler = InteractiveInputHandler(
            initial_position=self.configuration.initial_position,
            initial_look_at=self.configuration.initial_look_at,
            width=self.renderer.config.width,
            height=self.renderer.config.height,
        )

    def run(self) -> None:
        cached_surface: pygame.Surface | None = None

        try:
            with torch.no_grad():
                while True:
                    pose, screen, pose_changed = self.input_handler.get_next_pose(
                        compute_pose_fn=self._compute_pose_look_at,
                    )

                    if (
                        pose_changed
                        or cached_surface is None
                    ):  # fmt:skip
                        image_bgr = self._render_image(pose)
                        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                        cached_surface = pygame.surfarray.make_surface(image_rgb.swapaxes(0, 1))

                    screen.blit(cached_surface, (0, 0))

                    self.input_handler.update_display()
        except StopInferenceException as e:
            print(f"Stopping interactive inference: {e}")
        finally:
            self.input_handler.cleanup()

    def _render_image(
        self,
        pose: torch.Tensor,
    ) -> np.ndarray:
        camera = Camera(
            pose=pose,
            focal_length=self.renderer.config.focal_length,
            width=self.renderer.config.width,
            height=self.renderer.config.height,
        )

        rendered_image = self.renderer.render(
            camera=camera,
            gaussians=self.gaussians,
        )

        image_array = (rendered_image.array * 255).astype(np.uint8)
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
