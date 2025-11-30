import streamlit as st
from src.models.recommender import MovieRecommender
from src.ui.translator import Translator


def render_tab(recommender: MovieRecommender, t: Translator):
    st.header(t("tab3_header"))

    with st.form("add_movie_form"):
        new_title = st.text_input(t("input_title"))
        new_overview = st.text_area(t("input_overview"))
        all_genres = recommender.get_all_genres()
        new_genres = st.multiselect(t("input_genres"), options=all_genres)
        new_keywords = st.text_input(
            t("input_keywords"), placeholder="spy, mission, car chase"
        )

        submitted = st.form_submit_button(t("btn_add"))

        if submitted:
            if new_title.strip() and new_overview.strip() and new_genres:
                success = recommender.add_new_movie(
                    new_title, new_genres, new_keywords, new_overview
                )
                if success:
                    st.success(t("success_add").format(new_title))
                    st.rerun()
                else:
                    st.error(t("error_exists"))
            else:
                st.warning(t("warning_input"))
