import cv2
import numpy as np
import streamlit as st


class ImageDisplayComponent:
    def render(
        self,
        image: np.ndarray | None,
    ) -> None:
        if image is not None:
            image = np.clip(image, 0.0, 1.0)
            display_image = (image * 255.0).astype(np.uint8)

            try:
                success, encoded_image = cv2.imencode(
                    ".png",
                    cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR),
                )
                if success:
                    display_data = encoded_image.tobytes()
                else:
                    display_data = display_image
            except Exception:
                display_data = display_image

            st.image(
                display_data,
                width="stretch",
            )
        else:
            st.markdown(
                "<div style='text-align: center;'>Please load a .ply file.</div>",
                unsafe_allow_html=True,
            )
