import streamlit as st


def render(celeba_trainer, danbooru_trainer):
    st.title(body="Unstable Diffusion")
    celeba_tab, danbooru_tab = st.tabs(["CelebA", "Danbooru"])

    with celeba_tab:
        if st.button(label="Generate", key="celeba_generate_btn"):
            # celeba_img = celeba_trainer.sample()
            # st.image(celeba_img)
            pass
    with danbooru_tab:
        if st.button(label="Generate", key="danbooru_generate_btn"):
            # danbooru_img = danbooru_trainer.sample()
            # st.image(danbooru_trainer)
            pass


if __name__ == "__main__":
    celeba_trainer = None
    danbooru_trainer = None
    render(celeba_trainer=celeba_trainer, danbooru_trainer=danbooru_trainer)
