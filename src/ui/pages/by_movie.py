import streamlit as st
from src.ui.components import render_movie_card
from src.models.recommender import MovieRecommender
from src.ui.translator import Translator


def render_tab(recommender: MovieRecommender, t: Translator):
    st.header(t("tab1_header"))

    selected_movie = st.selectbox(t("select_movie"), recommender.get_all_titles())
    profile_weight = (
        st.slider(
            t("personalization_strength"),
            min_value=0,
            max_value=100,
            value=30,
            step=5,
            format="%d%%",
            key="movie_slider",
        )
        / 100.0
    )
    top_n = st.number_input(
        t("input_top_n"),
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        key="top_n_input_movie",
    )

    if st.button(t("btn_recommend"), key="btn1"):
        recommendations = recommender.recommend_by_movie(
            selected_movie, profile_weight=profile_weight, top_n=top_n
        )

        if recommendations:
            if profile_weight > 0 and not recommender.ratings:
                st.warning(t("no_ratings_warning"))

            st.success(t("success_movie").format(selected_movie))

            for row_start in range(0, len(recommendations), 5):
                cols = st.columns(5)
                for i, movie in enumerate(recommendations[row_start : row_start + 5]):
                    render_movie_card(movie, t, cols, total_cols=5, idx=i)
                for _ in range(len(recommendations[row_start : row_start + 5]), 5):
                    cols[_].empty()
        else:
            st.info("No recommendations found.")
