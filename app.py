import streamlit as st
from src.models.recommender import MovieRecommender
import dotenv
import pandas as pd

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


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

st.sidebar.title("📊 Análise de Dados")
st.sidebar.info("Dataset: TMDB 10k Most Popular Movies")

if st.sidebar.checkbox("Mostrar Gráficos do Dataset", value=True):
    st.sidebar.subheader("Distribuição de Popularidade")
    st.sidebar.line_chart(recommender.df["popularity"].head(20))
    st.sidebar.markdown(
        "**Técnica Aplicada:** Content-Based Filtering usando Cosine Similarity em vetores de texto (Genres + Keywords)."
    )

st.title("🎬 Sistema de Recomendação de Filmes")
st.markdown("Encontre seu próximo filme favorito usando Inteligência Artificial.")


tab1, tab2 = st.tabs(
    ["🔍 Busca por Filme Similar", "🧠 Busca por Preferências (Gen/Keywords)"]
)

with tab1:
    st.header("Porque você gostou de...")
    selected_movie = st.selectbox(
        "Escolha um filme:", recommender.get_all_titles(), index=0
    )

    if st.button("Recomendar por Filme"):
        recommendations = recommender.recommend_by_movie(selected_movie)

        if recommendations:
            st.success(f"Filmes similares a **{selected_movie}**:")
            cols = st.columns(5)
            for idx, movie in enumerate(recommendations):
                with cols[idx]:
                    # Display Poster if available
                    if pd.notna(movie["poster_path"]):
                        full_path = (
                            "https://image.tmdb.org/t/p/w500" + movie["poster_path"]
                        )
                        st.image(full_path, use_container_width=True)
                    st.markdown(f"**{movie['title']}**")
                    st.caption(f"Semelhança: {int(movie['score'] * 100)}%")
                    with st.popover("Ver Sinopse"):
                        st.write(movie["overview"])


with tab2:
    st.header("Descreva o que você quer ver")
    st.markdown(
        "Digite gêneros ou elementos de enredo (ex: *Action, Space, Robots, Love*)."
    )

    user_text = st.text_input(
        "Suas preferências:", placeholder="Ex: Science Fiction time travel paradox"
    )

    if st.button("Recomendar por Texto"):
        if user_text:
            recommendations = recommender.recommend_by_keywords(user_text)

            st.success(f"Melhores matches para: **'{user_text}'**")
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
                    with st.popover("Ver Sinopse"):
                        st.write(movie["overview"])
        else:
            st.warning("Por favor, digite algo para buscarmos.")

st.divider()
st.caption("Desenvolvido para a disciplina de Inteligência Artificial - BSI")
