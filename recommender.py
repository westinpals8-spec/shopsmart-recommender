"""
recommender.py
--------------
Hybrid E-Commerce Recommendation Engine
Combines content-based filtering (TF-IDF + cosine similarity) with
collaborative filtering (SVD matrix factorization) to generate product
recommendations for users.

Foundation reference:
    Scikit-learn documentation: https://scikit-learn.org
    Sarwar, B. et al. (2001). Item-based collaborative filtering recommendation
    algorithms. Proceedings of WWW '01.
    Ricci, F., Rokach, L., & Shapira, B. (2011). Recommender Systems Handbook.
    Springer.

Modifications and extensions by Westin Pals (2025):
    - Hybrid scoring combining content and collaborative signals
    - Category-aware filtering
    - Confidence scoring based on rating density
    - Fallback to content-only mode for cold-start users
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_data():
    """Load product catalog and user ratings from CSV files."""
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    ratings  = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    return products, ratings


class ContentBasedFilter:
    """
    Content-based recommendation using TF-IDF vectorization.

    Each product is represented as a TF-IDF vector built from its name,
    category, and description. Cosine similarity between products is used
    to find items similar to those a user has rated highly.

    This approach follows the methodology described in Pazzani & Billsus
    (2007), where user profiles are constructed from item features and
    matched against candidate items.
    """

    def __init__(self, products: pd.DataFrame):
        self.products = products.copy()
        self._build_feature_matrix()

    def _build_feature_matrix(self):
        # Combine textual fields into one feature string; weight category 3x
        self.products["features"] = (
            self.products["name"] + " "
            + self.products["category"] + " "
            + self.products["category"] + " "
            + self.products["category"] + " "
            + self.products["description"]
        )
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = tfidf.fit_transform(self.products["features"])
        self.sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def get_similar(self, product_id: int, top_n: int = 10) -> pd.DataFrame:
        """Return the top_n most content-similar products to product_id."""
        idx = self.products.index[self.products["product_id"] == product_id]
        if idx.empty:
            return pd.DataFrame()
        i = idx[0]
        scores = list(enumerate(self.sim_matrix[i]))
        scores = [(j, s) for j, s in scores if j != i]
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_n]
        indices = [x[0] for x in top]
        sims    = [x[1] for x in top]
        result = self.products.iloc[indices][["product_id", "name", "category", "price"]].copy()
        result["content_score"] = sims
        return result.reset_index(drop=True)

    def recommend_for_user(
        self,
        rated_ids: list,
        rated_values: list,
        top_n: int = 10,
        exclude_ids: list = None,
        category_filter: str = None,
    ) -> pd.DataFrame:
        """
        Build a user profile from their rated items and return candidate
        products ranked by content similarity to that profile.
        """
        if not rated_ids:
            return pd.DataFrame()

        # Weight TF-IDF vectors by the user's ratings (higher rating = more weight)
        profile_vecs = []
        for pid, rating in zip(rated_ids, rated_values):
            idx = self.products.index[self.products["product_id"] == pid]
            if not idx.empty:
                vec = self.tfidf_matrix[idx[0]].toarray() * rating
                profile_vecs.append(vec)

        if not profile_vecs:
            return pd.DataFrame()

        user_profile = np.mean(profile_vecs, axis=0)
        scores = cosine_similarity(user_profile, self.tfidf_matrix)[0]

        exclude = set(rated_ids) if exclude_ids is None else set(exclude_ids) | set(rated_ids)
        results = []
        for i, score in enumerate(scores):
            pid = int(self.products.iloc[i]["product_id"])
            if pid in exclude:
                continue
            cat = self.products.iloc[i]["category"]
            if category_filter and cat != category_filter:
                continue
            results.append({"product_id": pid, "content_score": float(score)})

        df = pd.DataFrame(results).sort_values("content_score", ascending=False)
        return df.head(top_n).reset_index(drop=True)


class CollaborativeFilter:
    """
    Collaborative filtering using Truncated SVD (matrix factorization).

    Builds a user-item rating matrix and decomposes it using SVD to discover
    latent factors representing underlying user preferences and product
    characteristics. Predicted ratings are generated by reconstructing the
    original matrix from the latent factors.

    This approach is inspired by the Simon Funk SVD method popularized during
    the Netflix Prize (Koren, Bell, & Volinsky, 2009) and implemented here
    using scikit-learn's TruncatedSVD for efficiency.
    """

    def __init__(self, products: pd.DataFrame, ratings: pd.DataFrame, n_components: int = 12):
        self.products = products
        self.ratings  = ratings
        self.n_components = n_components
        self._build_model()

    def _build_model(self):
        self.matrix = self.ratings.pivot_table(
            index="user_id", columns="product_id", values="rating"
        ).fillna(0)

        self.users    = list(self.matrix.index)
        self.prod_ids = list(self.matrix.columns)

        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        U   = svd.fit_transform(self.matrix.values)
        Vt  = svd.components_
        self.reconstructed = pd.DataFrame(
            np.dot(U, Vt),
            index=self.matrix.index,
            columns=self.matrix.columns,
        )

    def predict(self, user_id: str, product_id: int) -> float:
        """Return the predicted rating for (user_id, product_id)."""
        if user_id not in self.reconstructed.index:
            return 0.0
        if product_id not in self.reconstructed.columns:
            return 0.0
        return float(self.reconstructed.loc[user_id, product_id])

    def recommend_for_user(
        self,
        user_id: str,
        top_n: int = 10,
        exclude_ids: list = None,
        category_filter: str = None,
    ) -> pd.DataFrame:
        """Return top_n predicted-highest-rated unseen products for a user."""
        if user_id not in self.reconstructed.index:
            return pd.DataFrame()

        predicted = self.reconstructed.loc[user_id]
        rated_ids = set(
            self.ratings[self.ratings["user_id"] == user_id]["product_id"].tolist()
        )
        exclude = rated_ids if exclude_ids is None else rated_ids | set(exclude_ids)

        results = []
        for pid, score in predicted.items():
            if pid in exclude:
                continue
            cat_rows = self.products[self.products["product_id"] == pid]
            if cat_rows.empty:
                continue
            cat = cat_rows.iloc[0]["category"]
            if category_filter and cat != category_filter:
                continue
            results.append({"product_id": int(pid), "cf_score": float(score)})

        df = pd.DataFrame(results).sort_values("cf_score", ascending=False)
        return df.head(top_n).reset_index(drop=True)


class HybridRecommender:
    """
    Hybrid recommendation engine.

    Combines content-based and collaborative filtering scores using a
    weighted linear combination. The blend weight (alpha) controls the
    trade-off between the two approaches:

        final_score = alpha * CF_score + (1 - alpha) * CB_score

    For new users with sparse ratings (cold-start problem), alpha is
    automatically reduced to rely more heavily on content signals.

    Extension over baseline: this hybrid approach was added as an
    improvement over single-strategy baselines, following the survey by
    Burke (2002) which showed hybrid methods consistently outperform
    individual techniques.
    """

    def __init__(self, alpha: float = 0.55):
        """
        Parameters
        ----------
        alpha : float
            Weight given to collaborative filtering (0 = pure content, 1 = pure CF).
        """
        self.alpha    = alpha
        self.products = None
        self.ratings  = None
        self.cb       = None
        self.cf       = None
        self._loaded  = False

    def load(self):
        """Load data and fit both sub-models."""
        self.products, self.ratings = load_data()
        self.cb = ContentBasedFilter(self.products)
        self.cf = CollaborativeFilter(self.products, self.ratings)
        self._loaded = True

    def get_users(self) -> list:
        return sorted(self.ratings["user_id"].unique().tolist()) if self._loaded else []

    def get_categories(self) -> list:
        return ["All"] + sorted(self.products["category"].unique().tolist()) if self._loaded else []

    def recommend(
        self,
        user_id: str,
        top_n: int = 8,
        category_filter: str = None,
        alpha_override: float = None,
    ) -> pd.DataFrame:
        """
        Generate top_n hybrid recommendations for a given user.

        Returns a DataFrame with columns:
            product_id, name, category, price, cb_score, cf_score,
            hybrid_score, confidence
        """
        if not self._loaded:
            self.load()

        alpha = alpha_override if alpha_override is not None else self.alpha
        cat   = None if category_filter == "All" else category_filter

        # Retrieve the user's rated products and scores
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        rated_ids    = user_ratings["product_id"].tolist()
        rated_vals   = user_ratings["rating"].tolist()

        # Cold-start adjustment: reduce CF weight for sparse users
        n_ratings = len(rated_ids)
        if n_ratings < 5:
            alpha = max(0.1, alpha - 0.3)

        # --- Content-based scores ---
        cb_df = self.cb.recommend_for_user(
            rated_ids, rated_vals, top_n=top_n * 3, category_filter=cat
        )

        # --- Collaborative filtering scores ---
        cf_df = self.cf.recommend_for_user(
            user_id, top_n=top_n * 3, category_filter=cat
        )

        # Merge on product_id
        all_ids = set(cb_df["product_id"].tolist()) | set(cf_df["product_id"].tolist())
        cb_map  = dict(zip(cb_df["product_id"], cb_df["content_score"])) if not cb_df.empty else {}
        cf_map  = dict(zip(cf_df["product_id"], cf_df["cf_score"]))     if not cf_df.empty else {}

        # Normalize scores independently to [0, 1]
        def normalize(d):
            if not d:
                return d
            mn, mx = min(d.values()), max(d.values())
            rng = mx - mn if mx != mn else 1
            return {k: (v - mn) / rng for k, v in d.items()}

        cb_norm = normalize(cb_map)
        cf_norm = normalize(cf_map)

        rows = []
        for pid in all_ids:
            cb_s = cb_norm.get(pid, 0.0)
            cf_s = cf_norm.get(pid, 0.0)
            # Confidence: 1 if both models agree, lower if only one has signal
            both_present = (pid in cb_norm) and (pid in cf_norm)
            confidence   = "High" if both_present else "Medium"
            hybrid = alpha * cf_s + (1 - alpha) * cb_s
            rows.append({
                "product_id": pid,
                "cb_score":   round(cb_s,   3),
                "cf_score":   round(cf_s,   3),
                "hybrid_score": round(hybrid, 3),
                "confidence": confidence,
            })

        if not rows:
            return pd.DataFrame()

        merged = pd.DataFrame(rows).sort_values("hybrid_score", ascending=False).head(top_n)
        merged = merged.merge(
            self.products[["product_id", "name", "category", "price"]],
            on="product_id", how="left"
        )
        return merged[
            ["product_id", "name", "category", "price",
             "cb_score", "cf_score", "hybrid_score", "confidence"]
        ].reset_index(drop=True)

    def get_rated_products(self, user_id: str) -> pd.DataFrame:
        """Return products already rated by this user."""
        if not self._loaded:
            self.load()
        user_r = self.ratings[self.ratings["user_id"] == user_id]
        return user_r.merge(
            self.products[["product_id", "name", "category", "price"]],
            on="product_id"
        )[["product_id", "name", "category", "rating"]].sort_values("rating", ascending=False)
