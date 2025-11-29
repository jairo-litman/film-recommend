# Movie Recommender AI

This is a **Streamlit** application that recommends movies based on user preferences. The app uses **Content-Based Filtering** with **Cosine Similarity** to suggest movies similar to a selected title or based on user-provided keywords.

---

## Features

-   **Search by Movie**: Select a movie from the dataset to find similar titles.
-   **Search by Preferences**: Enter genres or plot elements to get personalized recommendations.
-   **Interactive Charts**: Visualize the dataset's popularity distribution.
-   **Multilingual Support**: Available in English and Portuguese.

---

## Tech Stack

-   **Python**: Core programming language.
-   **Streamlit**: Framework for building the web app.
-   **Pandas**: Data manipulation and analysis.
-   **Scikit-learn**: Machine learning library for similarity calculations.
-   **kagglehub**: For handling Kaggle datasets.

---

## Dataset

The app uses the [**Full TMDB Movies Dataset 2024 (1M Movies)**](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) dataset. We preprocess the data to focus on the top 10,000 movies based on popularity.

---

## Requirements

-   Python 3+
-   Kaggle API Token
-   Streamlit
-   UV

## Setup Instructions

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/jairo-litman/film-recommend
    cd film-recommend
    ```

2. **Create a Virtual Environment**:

    ```bash
    uv venv
    ```

3. **Install Dependencies**:

    ```bash
    uv sync
    ```

4. **Set Up Environment Variables**:

    - Create a `.env` file in the root directory.
    - Add your Kaggle API token and other configurations as shown in `.env.example`.

5. **Run the Data Preprocessing Script**:

    ```bash
    uv run src/data/kaggle.py
    ```

6. **Launch the Streamlit App**:
    ```bash
    uv run streamlit run app.py
    ```
