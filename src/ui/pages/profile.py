import streamlit as st
import pandas as pd
from src.models.recommender import MovieRecommender
from src.ui.translator import Translator
from src.ui.components import render_movie_card


def render_tab(recommender: MovieRecommender, t: Translator):
    st.header(t("tab4_header"))
    st.subheader(t("tab4_rate_movies"))

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        movie_to_rate = st.selectbox(
            t("tab4_select_movie"),
            recommender.get_all_titles(),
            key="rating_select",
        )
    with col2:
        if st.button(t("tab4_like"), width="stretch"):
            recommender.save_rating(movie_to_rate, "like")
            st.success(t("tab4_liked").format(movie_to_rate))
            st.rerun()
    with col3:
        if st.button(t("tab4_dislike"), width="stretch"):
            recommender.save_rating(movie_to_rate, "dislike")
            st.warning(t("tab4_disliked").format(movie_to_rate))
            st.rerun()

    st.divider()

    st.subheader(t("tab4_history"))
    history = recommender.ratings
    if history:
        hist_list = []
        for title, rating in history.items():
            rating_label = (
                t("tab4_rating_liked")
                if rating == "like"
                else t("tab4_rating_disliked")
            )
            hist_list.append(
                {"title": title, "rating": rating_label, "raw_rating": rating}
            )

        hist_df = pd.DataFrame(hist_list)
        st.dataframe(hist_df[["title", "rating"]], width="stretch", hide_index=True)

        remove_title = st.selectbox(
            t("select_movie_to_remove"), list(history.keys()), key="remove_rating"
        )
        if st.button(t("btn_remove_rating"), type="secondary"):
            recommender.remove_rating(remove_title)
            st.success(t("rating_removed").format(remove_title))
            st.rerun()

        if st.button(t("btn_clear_all_ratings"), type="primary"):
            recommender.clear_all_ratings()
            st.success(t("all_ratings_cleared"))
            st.rerun()
    else:
        st.info(t("tab4_no_history"))

    st.divider()

    st.subheader(t("user_added_movies"))
    user_movies = recommender.get_user_added_movies()
    if user_movies:
        user_df = pd.DataFrame(user_movies)[["title", "genres", "overview"]]
        st.dataframe(user_df, width="stretch", hide_index=True)

        remove_movie_title = st.selectbox(
            t("select_movie_to_remove"),
            [m["title"] for m in user_movies],
            key="remove_movie",
        )
        if st.button(t("btn_remove_movie"), type="secondary"):
            success = recommender.remove_user_movie(remove_movie_title)
            if success:
                st.success(t("movie_removed").format(remove_movie_title))
                st.rerun()
            else:
                st.error(t("movie_remove_failed"))

        if st.button(t("btn_clear_all_movies"), type="primary"):
            recommender.clear_all_user_movies()
            st.success(t("all_movies_cleared"))
            st.rerun()
    else:
        st.info(t("no_user_movies"))

    st.divider()

    st.subheader(t("tab4_recommendations"))

    top_n = st.number_input(
        t("input_top_n"),
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        key="top_n_input_personal",
    )

    if st.button(t("tab4_generate_recs"), type="primary"):
        personal_recs = recommender.recommend_personal(top_n=top_n)

        if personal_recs:
            for row_start in range(0, len(personal_recs), 5):
                cols = st.columns(5)
                for i, movie in enumerate(personal_recs[row_start : row_start + 5]):
                    render_movie_card(movie, t, cols, total_cols=5, idx=i)
                for _ in range(len(personal_recs[row_start : row_start + 5]), 5):
                    cols[_].empty()

        else:
            st.warning(t("tab4_no_likes_warning"))
