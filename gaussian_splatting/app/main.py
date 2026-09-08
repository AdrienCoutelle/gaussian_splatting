import mlx.core as mx
import numpy as np
import streamlit as st

from gaussian_splatting.structures.camera import Camera
from gaussian_splatting.structures.renderer.rasterizer import RasterizerConfig
from gaussian_splatting.structures.renderer.renderer import Renderer, RendererConfig

from .components.control_panel import ControlPanelComponent
from .components.gaussians_loader import GaussiansLoader
from .components.image_display import ImageDisplayComponent
from .components.image_save import ImageSaveComponent


class GaussianSplattingApp:
    def __init__(self):
        st.set_page_config(layout="wide")
        self.control_panel = ControlPanelComponent()
        self.gaussians_loader = GaussiansLoader()
        self.image_display = ImageDisplayComponent()
        self.image_saver = ImageSaveComponent()

    def apply_custom_styles(self):
        st.markdown(
            """
            <style>
            /* Base font adjustments */
            html, body, [class*="css"], p, span, label, input, button, select, div {
                font-size: 0.82rem !important;
            }
            /* Header scaling */
            h1 { font-size: 1.25rem !important; }
            h2 { font-size: 1.10rem !important; }
            h3 { font-size: 0.95rem !important; }
            /* Tabs adjustment */
            .stTabs [data-baseweb="tab"] {
                font-size: 0.82rem !important;
            }
            /* Metric values adjustment */
            [data-testid="stMetricValue"] {
                font-size: 1.05rem !important;
            }
            [data-testid="stMetricLabel"] {
                font-size: 0.72rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def run(self):
        self.apply_custom_styles()

        col_sidebar, col_main = st.columns([3, 7])

        with col_sidebar:
            gaussians, filename = self.gaussians_loader.render()
            st.markdown("---")
            config = self.control_panel.render(gaussians)
            st.markdown("---")
            save_image_container = st.container()

        cam_pos = config.get(
            "camera_position",
            {"x": 0.0, "y": 0.0, "z": 0.0},
        )
        x, y, z = cam_pos["x"], cam_pos["y"], cam_pos["z"]
        camera_intrinsics = config["camera_intrinsics"]
        focal_length = camera_intrinsics["focal_length"]
        width = camera_intrinsics["width"]
        height = camera_intrinsics["height"]

        look_at = config["look_at"]
        position = np.array([x, y, z], dtype=np.float32)

        dir_vec = look_at - position
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm < 1e-6:
            raise ValueError("Position and look_at are too close together")
        dir_vec /= dir_norm

        right_vec = np.cross(dir_vec, np.array([0, 0, 1]))
        right_norm = np.linalg.norm(right_vec)
        if right_norm < 1e-6:
            # dir_vec and world_up are parallel, use alternative up vector
            alternative_up = np.array([1.0, 0.0, 0.0]) if abs(dir_vec[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            right_vec = np.cross(dir_vec, alternative_up)
            right_norm = np.linalg.norm(right_vec)
        right_vec /= right_norm

        up_vec = np.cross(dir_vec, right_vec)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, 0] = right_vec
        pose[:3, 1] = up_vec
        pose[:3, 2] = dir_vec
        pose[:3, 3] = position

        camera = Camera(
            pose=mx.array(pose),
            width=width,
            height=height,
            focal_length=focal_length,
        )

        print("Camera pose:\n", camera.pose)

        print(f"Gaussians: {gaussians}")

        if gaussians is None:
            image = None
        else:
            renderer = Renderer(
                RendererConfig(
                    rasterizer_config=RasterizerConfig(),
                    draw_axis=config["draw_axis"],
                )
            )
            image = renderer.render(
                camera=camera,
                gaussians=gaussians,
            )

        with col_main:
            self.image_display.render(image)

        with save_image_container:
            self.image_saver.render(image, filename)


if __name__ == "__main__":
    app = GaussianSplattingApp()
    app.run()
