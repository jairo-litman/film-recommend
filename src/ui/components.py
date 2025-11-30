import streamlit as st
import pandas as pd
from typing import Dict, Any, List


def render_movie_card(
    movie: Dict[str, Any],
    t,
    cols: List,
    total_cols: int,
    idx: int,
) -> None:
    with cols[idx % total_cols]:
        poster_path = movie.get("poster_path")

        if pd.notna(poster_path):
            full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
            st.image(full_path, width="stretch")

        st.markdown(f"**{movie['title']}**")
        score_pct = int(movie["score"] * 100)

        st.caption(
            f"{t('similarity')}: {score_pct}%"
            if "similarity" in t.all()
            else f"Score: {score_pct}%"
        )

        with st.popover(t("popover")):
            st.write(movie.get("overview", ""))
