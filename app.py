import streamlit as st
import dotenv
from src.models.recommender import MovieRecommender
from src.ui.translator import Translator
from src.ui.pages.by_movie import render_tab as render_by_movie
from src.ui.pages.by_keywords import render_tab as render_by_keywords
from src.ui.pages.add_movie import render_tab as render_add_movie
from src.ui.pages.profile import render_tab as render_profile
import pandas as pd

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

lang_option = st.sidebar.radio("Language / Idioma", ["English", "Português"], index=0)
lang_code = "en" if lang_option == "English" else "pt"
t = Translator(lang_code)


@st.cache_resource
def load_model():
    dataset = (
        dotenv.get_key(dotenv.find_dotenv(), "DATASET_NAME") or "movies_top10k.csv"
    )
    model = MovieRecommender(dataset)
    model.fit()
    return model


try:
    recommender = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.sidebar.title(t("sidebar_title"))
st.sidebar.info(t("sidebar_info"))

if st.sidebar.checkbox(t("show_charts"), value=True):
    st.sidebar.subheader(t("chart_title"))

    top_movies = (
        recommender.df.nlargest(20, "popularity")[["title", "popularity"]]
        .copy()
        .reset_index(drop=True)
    )

    top_movies["title"] = top_movies["title"].astype(str).str[:25]
    top_movies = top_movies.sort_values("popularity", ascending=False)
    top_movies = top_movies.set_index("title")

    top_movies.index = pd.CategoricalIndex(
        top_movies.index,
        categories=top_movies.index,
        ordered=True,
    )

    st.sidebar.bar_chart(top_movies)
    st.sidebar.markdown(t("tech_explanation"))

st.title(t("app_title"))
st.markdown(t("app_subtitle"))

tabs = st.tabs([t("tab1"), t("tab2"), t("tab3"), t("tab4")])

with tabs[0]:
    render_by_movie(recommender, t)
with tabs[1]:
    render_by_keywords(recommender, t)
with tabs[2]:
    render_add_movie(recommender, t)
with tabs[3]:
    render_profile(recommender, t)

st.divider()
st.caption(t("footer"))
