from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import numpy as np


class MovieRecommender:
    def __init__(
        self,
        csv_path: str,
        json_movies: str = "user_movies.json",
        json_ratings: str = "user_ratings.json",
    ):
        self.csv_path = csv_path
        self.json_movies = json_movies
        self.json_ratings = json_ratings

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
        )
        self.vectors: Optional[np.ndarray] = None
        self.df: pd.DataFrame = self._load_movies()
        self.ratings: Dict[str, str] = self._load_ratings()

    # ----------------------------
    # Data Loading
    # ----------------------------

    def _load_movies(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Base data file not found: {self.csv_path}")

        df_base = pd.read_csv(self.csv_path)

        if os.path.exists(self.json_movies):
            try:
                with open(self.json_movies, "r") as f:
                    user_data = json.load(f)
                    if user_data:
                        user_df = pd.DataFrame(user_data)
                        df_base = pd.concat([df_base, user_df], ignore_index=True)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(
                    f"Warning: Failed to load user movies from {self.json_movies}: {e}"
                )

        return df_base

    def _load_ratings(self) -> Dict[str, str]:
        if not os.path.exists(self.json_ratings):
            return {}

        try:
            with open(self.json_ratings, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Failed to load ratings from {self.json_ratings}: {e}")
            return {}

    # ----------------------------
    # Model Training
    # ----------------------------

    def fit(self) -> None:
        self.df["combined_features"] = (
            self.df["genres"].fillna("")
            + " "
            + self.df["keywords"].fillna("")
            + " "
            + self.df["overview"].fillna("")
        )

        self.vectors = self.vectorizer.fit_transform(
            self.df["combined_features"]
        ).toarray()

    # ----------------------------
    # Recommendation Logic
    # ----------------------------

    def recommend_by_movie(
        self, movie_title: str, top_n: int = 5, profile_weight: float = 0.0
    ) -> List[Dict[str, Any]]:
        if self.vectors is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if movie_title not in self.df["title"].values:
            return []

        movie_idx = self.df[self.df["title"] == movie_title].index[0]
        sim_scores = cosine_similarity([self.vectors[movie_idx]], self.vectors)[0]

        if profile_weight > 0:
            user_vec = self._get_adjusted_user_vector()
            if user_vec is not None:
                profile_scores = cosine_similarity([user_vec], self.vectors)[0]
                final_scores = (
                    1 - profile_weight
                ) * sim_scores + profile_weight * profile_scores
            else:
                final_scores = sim_scores
        else:
            final_scores = sim_scores

        return self._get_top_recommendations(
            final_scores, exclude_idx=movie_idx, top_n=top_n
        )

    def recommend_by_keywords(
        self, keywords: str, top_n: int = 5, profile_weight: float = 0.0
    ) -> List[Dict[str, Any]]:
        if self.vectors is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        split_keywords = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        if not split_keywords:
            return []

        query_vec = self.vectorizer.transform(split_keywords).toarray()
        query_scores = cosine_similarity(query_vec, self.vectors)[0]

        if profile_weight > 0:
            user_vec = self._get_adjusted_user_vector()
            if user_vec is not None:
                profile_scores = cosine_similarity([user_vec], self.vectors)[0]
                final_scores = (
                    1 - profile_weight
                ) * query_scores + profile_weight * profile_scores
            else:
                final_scores = query_scores
        else:
            final_scores = query_scores

        return self._get_top_recommendations(final_scores, top_n=top_n)

    def recommend_personal(self, top_n: int = 5) -> List[Dict[str, Any]]:
        if self.vectors is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        user_vec = self._get_adjusted_user_vector()
        if user_vec is None:
            return []

        scores = cosine_similarity([user_vec], self.vectors)[0]
        watched_titles = set(self.ratings.keys())

        return self._get_top_recommendations(
            scores, exclude_titles=watched_titles, top_n=top_n
        )

    def _get_top_recommendations(
        self,
        scores: np.ndarray,
        top_n: int = 5,
        exclude_idx: Optional[int] = None,
        exclude_titles: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        scored_items = list(enumerate(scores))
        scored_items.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored_items:
            if exclude_idx is not None and idx == exclude_idx:
                continue
            if (
                exclude_titles is not None
                and self.df.iloc[idx]["title"] in exclude_titles
            ):
                continue

            results.append(self._format_result(idx, score))
            if len(results) >= top_n:
                break

        return results

    # ----------------------------
    # User Profile
    # ----------------------------

    def _get_adjusted_user_vector(self, alpha: float = 1.0, beta: float = 0.5):
        if self.vectors is None:
            return None

        liked_titles = [t for t, r in self.ratings.items() if r == "like"]
        disliked_titles = [t for t, r in self.ratings.items() if r == "dislike"]

        like_vec = None
        dislike_vec = None

        if liked_titles:
            like_indices = self.df[self.df["title"].isin(liked_titles)].index
            if not like_indices.empty:
                like_vec = np.mean(self.vectors[like_indices], axis=0)

        if like_vec is None:
            return None

        if disliked_titles:
            dislike_indices = self.df[self.df["title"].isin(disliked_titles)].index
            if not dislike_indices.empty:
                dislike_vec = np.mean(self.vectors[dislike_indices], axis=0)

        user_vec = alpha * like_vec
        if dislike_vec is not None:
            user_vec = user_vec - beta * dislike_vec

        return user_vec

    # ----------------------------
    # Movie Management
    # ----------------------------

    def add_new_movie(
        self, title: str, genres_list: List[str], keywords: str, overview: str
    ) -> bool:
        if title in self.df["title"].values:
            return False

        genres_str = ", ".join(genres_list)
        new_id = (
            int(self.df["id"].max()) + 1
            if "id" in self.df.columns and not self.df["id"].empty
            else 1
        )

        new_movie = {
            "id": new_id,
            "title": title,
            "genres": genres_str,
            "keywords": keywords,
            "overview": overview,
            "popularity": 50,
            "vote_count": 1,
            "vote_average": 0,
            "poster_path": None,
        }

        self._save_movie_to_json(new_movie)

        new_row = pd.DataFrame([new_movie])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.fit()

        return True

    def _save_movie_to_json(self, movie: Dict[str, Any]) -> None:
        data = []
        if os.path.exists(self.json_movies):
            try:
                with open(self.json_movies, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, TypeError):
                data = []

        data.append(movie)
        with open(self.json_movies, "w") as f:
            json.dump(data, f, indent=4)

    def save_rating(self, title: str, rating: str) -> None:
        if rating not in {"like", "dislike"}:
            raise ValueError("Rating must be 'like' or 'dislike'.")

        self.ratings[title] = rating
        with open(self.json_ratings, "w") as f:
            json.dump(self.ratings, f, indent=4)

    def remove_rating(self, title: str) -> bool:
        if title in self.ratings:
            del self.ratings[title]
            with open(self.json_ratings, "w") as f:
                json.dump(self.ratings, f, indent=4)
            return True
        return False

    def remove_user_movie(self, title: str) -> bool:
        if not os.path.exists(self.json_movies):
            return False

        try:
            with open(self.json_movies, "r") as f:
                user_movies = json.load(f)
        except (json.JSONDecodeError, TypeError):
            return False

        if not isinstance(user_movies, list):
            return False

        before_count = len(user_movies)
        user_movies = [m for m in user_movies if m.get("title") != title]
        after_count = len(user_movies)

        if before_count == after_count:
            return False

        with open(self.json_movies, "w") as f:
            json.dump(user_movies, f, indent=4)

        self.df = self._load_movies()
        self.fit()

        return True

    def clear_all_ratings(self) -> None:
        self.ratings = {}
        with open(self.json_ratings, "w") as f:
            json.dump(self.ratings, f, indent=4)

    def clear_all_user_movies(self) -> None:
        if os.path.exists(self.json_movies):
            os.remove(self.json_movies)

        self.df = self._load_movies()
        self.fit()

    # ----------------------------
    # Utilities
    # ----------------------------

    def _format_result(self, idx: int, score: float) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        return {
            "title": row["title"],
            "genres": row["genres"],
            "score": round(float(score), 2),
            "poster_path": row.get("poster_path"),
            "overview": row.get("overview"),
        }

    def get_all_titles(self) -> List[str]:
        return self.df["title"].dropna().tolist()

    def get_all_genres(self) -> List[str]:
        all_genres = set()

        for genres in self.df["genres"].dropna():
            all_genres.update(g.strip() for g in genres.split(","))

        return sorted(all_genres)

    def get_user_added_movies(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.json_movies):
            return []
        try:
            with open(self.json_movies, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
