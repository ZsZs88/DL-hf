import streamlit as st


def render(celeba_trainer, danbooru_trainer):
    session = st.session_state
    st.title(body="Unstable Diffusion")
    celeba_col, danbooru_col = st.columns([1, 1], gap="large")

    with celeba_col:
        if "celeba_img" in session:
            st.image(session["celeba_img"])
        if st.button(label="Generate", key="celeba_generate_btn"):
            # session["celeba_img"] = celeba_trainer.sample()
            st.rerun()
    with danbooru_col:
        if "danbooru_img" in session:
            st.image(session["danbooru_img"])
        if st.button(label="Generate", key="danbooru_generate_btn"):
            # session["danbooru_img"] = danbooru_trainer.sample()
            st.rerun()



if __name__ == "__main__":
    celeba_trainer = None
    danbooru_trainer = None
    render(celeba_trainer=celeba_trainer, danbooru_trainer=danbooru_trainer)
