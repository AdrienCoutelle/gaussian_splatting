import numpy as np
import streamlit as st

from gaussian_splatting.structures.gaussian import Gaussians


class ControlPanelComponent:
    def render(
        self,
        gaussians: Gaussians | None,
    ) -> dict:
        config = {}
        st.subheader("Control Panel")

        st.markdown("### Camera Position")
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            x = st.number_input("X", min_value=-20.0, max_value=20.0, value=1.0, step=0.1, format="%.1f")
        with col_y:
            y = st.number_input("Y", min_value=-20.0, max_value=20.0, value=0.0, step=0.1, format="%.1f")
        with col_z:
            z = st.number_input("Z", min_value=-20.0, max_value=20.0, value=0.0, step=0.1, format="%.1f")

        config["camera_position"] = {"x": x, "y": y, "z": z}

        st.markdown("### Look-at Position")
        look_at_mode = st.radio(
            "Look-at target",
            options=["Mean of Gaussians", "Manual"],
            index=0 if gaussians is not None else 1,
            horizontal=True,
        )

        use_mean_look_at = look_at_mode == "Mean of Gaussians" and gaussians is not None
        mean_look_at = (
            np.asarray(gaussians.positions).mean(axis=0)
            if gaussians is not None
            else np.zeros(3, dtype=np.float32)
        )
        look_at_x, look_at_y, look_at_z = st.columns(3)
        with look_at_x:
            target_x = st.number_input(
                "Target X",
                value=float(mean_look_at[0]) if use_mean_look_at else 0.0,
                step=0.1,
                format="%.1f",
                disabled=use_mean_look_at,
            )
        with look_at_y:
            target_y = st.number_input(
                "Target Y",
                value=float(mean_look_at[1]) if use_mean_look_at else 0.0,
                step=0.1,
                format="%.1f",
                disabled=use_mean_look_at,
            )
        with look_at_z:
            target_z = st.number_input(
                "Target Z",
                value=float(mean_look_at[2]) if use_mean_look_at else 0.0,
                step=0.1,
                format="%.1f",
                disabled=use_mean_look_at,
            )
        look_at = (
            mean_look_at
            if use_mean_look_at
            else np.array(
                [target_x, target_y, target_z],
                dtype=np.float32,
            )
        )

        config["look_at"] = look_at

        st.markdown("### Camera Intrinsics")
        focal_length = st.number_input(
            "Focal Length",
            min_value=1.0,
            max_value=500.0,
            value=50.0,
            step=10.0,
        )
        col_width, col_height = st.columns(2)
        with col_width:
            width = st.number_input(
                "Width",
                min_value=16,
                max_value=2048,
                value=512,
                step=16,
            )
        with col_height:
            height = st.number_input(
                "Height",
                min_value=16,
                max_value=2048,
                value=288,
                step=16,
            )

        config["camera_intrinsics"] = {
            "focal_length": focal_length,
            "width": width,
            "height": height,
        }
        config["draw_axis"] = st.checkbox("Draw Axis", value=False)

        return config
