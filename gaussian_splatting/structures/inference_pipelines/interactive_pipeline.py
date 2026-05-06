import datetime
import os
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

        os.makedirs(self.output_folder, exist_ok=True)

        output_name = (
            f"position_epoch_{self.epoch}.jpg"
            if self.epoch is not None
            else f"single_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )

        self.output_path = os.path.join(
            self.output_folder,
            output_name,
        )

        self.device = device

        self.position = np.array(self.configuration.initial_position, dtype=np.float32)
        self.look_at = np.array(self.configuration.initial_look_at, dtype=np.float32)

        forward = self.look_at - self.position
        forward = forward / np.linalg.norm(forward)
        self.yaw = np.arctan2(forward[1], forward[0])
        self.pitch = np.arcsin(-forward[2])

        self.mouse_sensitivity = 0.002
        self.movement_speed = 0.1

    def run(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode(
            (self.renderer.config.width, self.renderer.config.height),
        )
        pygame.display.set_caption("Gaussian Splatting Interactive Viewer")
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        clock = pygame.time.Clock()
        running = True
        frame_count = 0

        with torch.no_grad():
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_s:
                            self._save_current_view()
                    elif event.type == pygame.MOUSEMOTION:
                        self._handle_mouse_movement(
                            dx=event.rel[0],
                            dy=event.rel[1],
                        )

                self._handle_keyboard_input()

                self._update_look_at()

                pose = torch.from_numpy(
                    self._compute_pose_look_at(
                        position=self.position,
                        look_at=self.look_at,
                        world_up=np.array([0, 0, 1]),
                    )
                )

                image_bgr = self._render_image(pose)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                surface = pygame.surfarray.make_surface(image_rgb.swapaxes(0, 1))
                screen.blit(surface, (0, 0))

                pygame.display.flip()
                clock.tick(60)

                frame_count += 1
                if frame_count % 60 == 0:
                    fps = clock.get_fps()
                    print(f"FPS: {fps:.1f}")

        pygame.quit()

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

    def _handle_mouse_movement(
        self,
        dx: int,
        dy: int,
    ) -> None:
        self.yaw += dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity

        self.pitch = np.clip(self.pitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)

    def _update_look_at(self) -> None:
        forward = np.array(
            [
                np.cos(self.pitch) * np.cos(self.yaw),
                np.cos(self.pitch) * np.sin(self.yaw),
                -np.sin(self.pitch),
            ],
            dtype=np.float32,
        )
        self.look_at = self.position + forward

    def _handle_keyboard_input(self) -> None:
        keys = pygame.key.get_pressed()

        forward = self.look_at - self.position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, np.array([0, 0, 1]))
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-6:
            right = right / right_norm
        else:
            right = np.array([1, 0, 0])

        up = np.cross(right, forward)

        if keys[pygame.K_w]:
            self.position += forward * self.movement_speed
        if keys[pygame.K_s]:
            self.position -= forward * self.movement_speed
        if keys[pygame.K_a]:
            self.position -= right * self.movement_speed
        if keys[pygame.K_d]:
            self.position += right * self.movement_speed
        if keys[pygame.K_SPACE]:
            self.position += up * self.movement_speed
        if keys[pygame.K_LSHIFT]:
            self.position -= up * self.movement_speed

    def _save_current_view(self) -> None:
        pose = torch.from_numpy(
            self._compute_pose_look_at(
                position=self.position,
                look_at=self.look_at,
                world_up=np.array([0, 0, 1]),
            )
        )

        image_bgr = self._render_image(pose)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            self.output_folder,
            f"screenshot_{timestamp}.jpg",
        )

        cv2.imwrite(
            filename=save_path,
            img=image_bgr,
        )

        print(f"Screenshot saved to {save_path}")
