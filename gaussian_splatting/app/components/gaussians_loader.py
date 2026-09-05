import os
import tempfile

import streamlit as st

from gaussian_splatting.structures.gaussian import GaussianCollection
from gaussian_splatting.utils.ply.ply_loader import PLYLoader


class GaussianCollectionLoader:
    def render(self) -> tuple[GaussianCollection | None, str | None]:
        st.subheader("Model Loader")
        uploaded_file = st.file_uploader(
            "Select PLY File",
            type=["ply"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            original_filename = uploaded_file.name
            st.markdown(f"**Selected file:** `{original_filename}`")

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name

                try:
                    ply_handler = PLYLoader(file_path=tmp_file_path)
                    gaussians = ply_handler.get_gaussians()
                finally:
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)

                return gaussians, original_filename
            except Exception as e:
                st.error(f"Failed to load file: {e}")
                return None, None

        st.markdown("**Selected file:** `No file selected`")
        return None, None
