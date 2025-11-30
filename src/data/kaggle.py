import pandas as pd
import kagglehub
import os
from dotenv import load_dotenv
import dotenv
import logging


def clean_and_reduce_data():
    path = kagglehub.dataset_download("asaniczka/tmdb-movies-dataset-2023-930k-movies")
    csv_path = os.path.join(path, "TMDB_movie_dataset_v11.csv")

    df = pd.read_csv(csv_path)

    logging.log(f"Original shape: {df.shape}")

    df_filtered = df[(df["vote_count"] > 50) & (~df["adult"])].copy()

    df_filtered = df_filtered.sort_values(by="popularity", ascending=False).head(10000)

    logging.log(f"Reduced shape: {df_filtered.shape}")

    cols_to_keep = [
        "id",
        "title",
        "genres",
        "keywords",
        "overview",
        "release_date",
        "vote_average",
        "vote_count",
        "popularity",
        "runtime",
        "poster_path",
    ]
    df_final = df_filtered[cols_to_keep].copy()

    df_final["genres"] = df_final["genres"].fillna("")
    df_final["keywords"] = df_final["keywords"].fillna("")
    df_final["overview"] = df_final["overview"].fillna("")

    df_final = df_final.sort_values(by="title").reset_index(drop=True)

    output_file = (
        dotenv.get_key(dotenv.find_dotenv(), "DATASET_NAME") or "movies_top10k.csv"
    )
    df_final.to_csv(output_file, index=False)

    logging.log(f"Cleaned data saved to: {output_file}")


if __name__ == "__main__":
    load_dotenv()

    clean_and_reduce_data()
