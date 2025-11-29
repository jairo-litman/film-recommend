import streamlit as st
from src.models.recommender import MovieRecommender
import dotenv
import pandas as pd
from src.texts import translations

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

lang_option = st.sidebar.radio(
    "Language / Idioma",
    ["English", "Português"],
    index=0,
)

lang_code = "en" if lang_option == "English" else "pt"
t = translations[lang_code]


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

st.sidebar.title(t["sidebar_title"])
st.sidebar.info(t["sidebar_info"])

if st.sidebar.checkbox(t["show_charts"], value=True):
    st.sidebar.subheader(t["chart_title"])
    st.sidebar.line_chart(recommender.df["popularity"].head(20))
    st.sidebar.markdown(t["tech_explanation"])

st.title(t["app_title"])
st.markdown(t["app_subtitle"])

tab1, tab2 = st.tabs([t["tab1"], t["tab2"]])

with tab1:
    st.header(t["tab1_header"])
    selected_movie = st.selectbox(
        t["select_movie"], recommender.get_all_titles(), index=0
    )

    if st.button(t["btn_recommend"]):
        recommendations = recommender.recommend_by_movie(selected_movie)

        if recommendations:
            st.success(t["success_movie"].format(selected_movie))
            cols = st.columns(5)
            for idx, movie in enumerate(recommendations):
                with cols[idx]:
                    if pd.notna(movie["poster_path"]):
                        full_path = (
                            "https://image.tmdb.org/t/p/w500" + movie["poster_path"]
                        )
                        st.image(full_path, use_container_width=True)
                    st.markdown(f"**{movie['title']}**")
                    st.caption(f"{t['similarity']}: {int(movie['score'] * 100)}%")
                    with st.popover(t["popover"]):
                        st.write(movie["overview"])

with tab2:
    st.header(t["tab2_header"])
    st.markdown(t["tab2_sub"])

    user_text = st.text_input("Input:", placeholder=t["input_placeholder"])

    if st.button(t["btn_text_rec"]):
        if user_text:
            recommendations = recommender.recommend_by_keywords(user_text)

            st.success(t["success_text"].format(user_text))
            cols = st.columns(5)
            for idx, movie in enumerate(recommendations):
                with cols[idx]:
                    if pd.notna(movie["poster_path"]):
                        full_path = (
                            "https://image.tmdb.org/t/p/w500" + movie["poster_path"]
                        )
                        st.image(full_path, use_container_width=True)
                    st.markdown(f"**{movie['title']}**")
                    st.caption(f"Score: {int(movie['score'] * 100)}%")
                    with st.popover(t["popover"]):
                        st.write(movie["overview"])
        else:
            st.warning(t["warning_input"])

# Footer
st.divider()
st.caption(t["footer"])
