import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

        self.df["combined_features"] = (
            self.df["genres"] + " " + self.df["keywords"] + " " + self.df["overview"]
        ).fillna("")

        self.vectorizer = CountVectorizer(max_features=5000, stop_words="english")
        self.vectors = None

    def fit(self):
        self.vectors = self.vectorizer.fit_transform(self.df["combined_features"])

    def recommend_by_movie(self, movie_title, top_n=5):
        if self.vectors is None:
            raise Exception("Model not fitted. Call fit() first.")

        if movie_title not in self.df["title"].values:
            return []

        movie_index = self.df[self.df["title"] == movie_title].index[0]

        similarity_scores = cosine_similarity(self.vectors[movie_index], self.vectors)

        sorted_scores = sorted(
            list(enumerate(similarity_scores[0])), reverse=True, key=lambda x: x[1]
        )[1 : top_n + 1]

        return self._get_results_from_indices(sorted_scores)

    def recommend_by_keywords(self, user_keywords, top_n=5):
        if self.vectors is None:
            raise Exception("Model not fitted. Call fit() first.")

        user_vector = self.vectorizer.transform([user_keywords])

        similarity_scores = cosine_similarity(user_vector, self.vectors)

        sorted_scores = sorted(
            list(enumerate(similarity_scores[0])), reverse=True, key=lambda x: x[1]
        )[:top_n]

        return self._get_results_from_indices(sorted_scores)

    def _get_results_from_indices(self, sorted_scores):
        results = []
        for i, score in sorted_scores:
            results.append(
                {
                    "title": self.df.iloc[i]["title"],
                    "genres": self.df.iloc[i]["genres"],
                    "score": round(score, 2),
                    "poster_path": self.df.iloc[i]["poster_path"],
                    "overview": self.df.iloc[i]["overview"],
                }
            )
        return results

    def get_all_titles(self):
        return self.df["title"].values
