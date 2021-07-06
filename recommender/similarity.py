import logging
import numpy as np
import pandas as pd
from typing import List
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances

logging.getLogger().setLevel(logging.INFO)


class UserSimilarityMatrix:
    """Class for building and computing similar users"""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __repr__(self) -> str:
        return f"Dimensions of User-Items Matrix: {self.matrix.shape}"

    def build_user_item_matrix(self, max_users: str, item: str) -> None:
        """Build User/Item Interaction Matrix"""
        matrix = np.zeros(shape=(max_users, max(self.data[item])))
        for _, row in self.data.iterrows():
            matrix[row["user_handle"] - 1, row[item] - 1] = 1
        return matrix

    def get_user_item_matrix(self, max_users: int, features: List[str]):
        """Concatenate Features into One User-Items Matrix"""
        results = []
        for item in features:
            results.append(self.build_user_item_matrix(max_users, item))
        self.matrix = np.hstack(results)

    def _truncatedSVD(self, threshold: float = 0.90) -> np.ndarray:
        """Apply Truncated SVD to Explain 'n'% of total variance"""
        n_components = 2  # minimum components to begin
        ex_var = 0
        while ex_var < threshold:
            pc = TruncatedSVD(n_components=n_components)
            pc.fit_transform(self.matrix)
            ex_var = np.sum(pc.explained_variance_ratio_)
            n_components += 1
        logging.info(
            f"Total components {pc.n_components} with {ex_var:0.2f} variance explained"
        )
        self.matrix= pc.transform(self.matrix)

    def compute_similarity(self, metric: str = "cosine") -> np.ndarray:
        """Compute Similarity Matrix"""
        score = pairwise_distances(self.matrix, metric=metric)
        if metric == "cosine":
            return 1 - score
        return score

def compute_weighted_matrix(
    users: np.ndarray, assessments: np.ndarray, course: np.ndarray, weights: List[float]
) -> np.ndarray:
    """Compute Weighted Similary Matrix where: weight_1 + weight_2 + weight_3 = 1"""
    return (
        (users * float(weights[0]))
        + (assessments * float(weights[1]))
        + (course * float(weights[2]))
    )


def rank_similar_users(X: np.ndarray, top_n: int = 5) -> pd.DataFrame:
    """Apply Custom Pandas Function to Rank Top 'n' Users"""

    def custom_udf(X):
        """
        Custom Pandas function for using index/score to
        generate output results dataframe.
        """
        idx = np.argsort(X.values, axis=0)[::-1][1 : top_n + 1]
        return [
            str({"user": i, "score": X.astype(float).round(4).values[i]}) for i in idx
        ]

    # dimensions: users x top_n
    if isinstance(X,np.ndarray):
        X = pd.DataFrame(X)
    ranking = X.apply(custom_udf).T
    ranking.columns = [f"{i+1}" for i in ranking.columns]
    ranking["user_handle"] = ranking.index
    logging.info(f"User Ranking Dataframe Shape: {ranking.shape}")
    return ranking
