import streamlit as st

import train
import paths


def render(celeba_trainer: train.Trainer, danbooru_trainer: train.Trainer) -> None:
    session = st.session_state
    st.title(body="Unstable Diffusion")
    celeba_col, danbooru_col = st.columns([1, 1], gap="large")

    with celeba_col:
        st.subheader("CelebA")
        if "celeba_img" in session:
            st.image(session["celeba_img"])
        if st.button(label="Generate", key="celeba_generate_btn"):
            session["celeba_img"] = celeba_trainer.sample_one_for_GUI()
            st.rerun()
    with danbooru_col:
        st.subheader("Danbooru")
        if "danbooru_img" in session:
            st.image(session["danbooru_img"])
        if st.button(label="Generate", key="danbooru_generate_btn"):
            session["danbooru_img"] = danbooru_trainer.sample_one_for_GUI()
            st.rerun()


if __name__ == "__main__":
    celeba_trainer = train.Trainer()
    celeba_trainer.add_paths(paths.celeba)
    celeba_trainer.add_model(
        "/home/l0026@njmcs.local/data/Generative/datasets/celeba/models/sample_30.pth"
    )
    danbooru_trainer = train.Trainer()
    danbooru_trainer.add_paths(paths.danbooru)
    danbooru_trainer.add_model(
        "/home/l0026@njmcs.local/data/Generative/datasets/danbooru/models/sample_30.pth"
    )
    render(celeba_trainer=celeba_trainer, danbooru_trainer=danbooru_trainer)
