import os

import cv2
import numpy as np
import streamlit as st


class ImageSaveComponent:
    def render(
        self,
        image: np.ndarray | None,
        default_filename: str | None,
    ) -> None:
        st.subheader("Save Image")

        if image is None or default_filename is None:
            st.info("No image to save.")
            return

        image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        success, encoded_image = cv2.imencode(
            ".png",
            cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR),
        )
        if not success:
            st.error("Failed to encode image as PNG.")
            return

        default_png_filename = f"{os.path.splitext(default_filename)[0]}.png"
        st.download_button(
            "Save PNG",
            data=encoded_image.tobytes(),
            file_name=default_png_filename,
            mime="image/png",
            use_container_width=True,
        )
