import streamlit as st
from src.ui.components import render_movie_card
from src.models.recommender import MovieRecommender
from src.ui.translator import Translator


def render_tab(recommender: MovieRecommender, t: Translator):
    st.header(t("tab2_header"))
    st.markdown(t("tab2_sub"))

    user_text = st.text_input(
        "", placeholder=t("input_placeholder"), key="keyword_input"
    )
    profile_weight = (
        st.slider(
            t("personalization_strength"),
            min_value=0,
            max_value=100,
            value=30,
            step=5,
            format="%d%%",
            key="keyword_slider",
        )
        / 100.0
    )
    top_n = st.number_input(
        t("input_top_n"),
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        key="top_n_input_keywords",
    )

    if st.button(t("btn_text_rec"), key="btn2"):
        if user_text.strip():
            recommendations = recommender.recommend_by_keywords(
                user_text.strip(), profile_weight=profile_weight, top_n=top_n
            )
            st.success(t("success_text").format(user_text))

            for row_start in range(0, len(recommendations), 5):
                cols = st.columns(5)
                for i, movie in enumerate(recommendations[row_start : row_start + 5]):
                    render_movie_card(movie, t, cols, total_cols=5, idx=i)
                for _ in range(len(recommendations[row_start : row_start + 5]), 5):
                    cols[_].empty()
        else:
            st.warning(t("warning_input"))
