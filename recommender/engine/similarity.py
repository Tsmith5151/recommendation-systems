import numpy as np
import pandas as pd
from typing import List
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances

from recommender.logger import get_logger

logger = get_logger(__name__)


class UserSimilarityMatrix:
    """Class for building and computing similar users"""

    def __init__(self, data: pd.DataFrame):
        """
        Generate user-user similarity metrics 

        Parameters
        ----------
        data: pd.DataFrame
            input datataframe
        """
        self.data = data

    def __repr__(self) -> str:
        return f"Dimensions of User-Items Matrix: {self.matrix.shape}"

    def build_user_item_matrix(self, max_users: str, item: str) -> None:
        """Build User/Item Interaction Matrix

        Parameters
        ----------
        max_users: int
            maximum number of users for creating user-matrix matrix dimensions
        features: str
            input feature column for creating user-items matrix
        """
        matrix = np.zeros(shape=(max_users, max(self.data[item])))
        for _, row in self.data.iterrows():
            matrix[row["user_handle"] - 1, row[item] - 1] = 1
        return matrix

    def get_user_item_matrix(self, max_users: int, features: List[str]):
        """Concatenate Features into One User-Items Matrix
        
        Parameters
        ----------
        max_users: int
            maximum number of users for creating user-matrix matrix dimensions
        features: List[str]
            list of features columns for creating user-items matrix
        """
        results = []
        for item in features:
            results.append(self.build_user_item_matrix(max_users, item))
        self.matrix = np.hstack(results)

    def _truncatedSVD(self, threshold: float = 0.90) -> np.ndarray:
        """Apply Truncated SVD to Explain 'n'% of total variance
        
        Parameters
        ----------
        threshold: float
            minimum variance threshold to explain 
        """
        n_components = 2  # minimum components to begin
        ex_var = 0
        while ex_var < threshold:
            pc = TruncatedSVD(n_components=n_components)
            pc.fit_transform(self.matrix)
            ex_var = np.sum(pc.explained_variance_ratio_)
            n_components += 1
        logger.info(
            f"Total components {pc.n_components} with {ex_var:0.2f} variance explained"
        )
        self.matrix = pc.transform(self.matrix)

    def compute_similarity(self) -> np.ndarray:
        """Compute Pairwise Cosine Distance Matrix
        
        Parameters
        ----------
        None
        """
        return 1 - pairwise_distances(self.matrix, metric="cosine")


def compute_weighted_matrix(
    users: np.ndarray,
    assessments: np.ndarray,
    course: np.ndarray,
    weights: List[float],
) -> np.ndarray:
    """
    Generate weighted user-user matrices for each table:
    Interest/Assessment/Course Views

    Parameters
    ----------
    users: np.ndarray
        input user similarity matrix 
    assessments: np.ndarray
        input assessment similarity matrix 
    course: np.ndarray
        input course similarity matrix 
    weights: List[float]
        input list of weights associated with each matrix 
    
    equation: Aggregated Matrix: weight_1 + weight_2 + weight_3 = 1
    """
    return (
        (users * float(weights[0]))
        + (assessments * float(weights[1]))
        + (course * float(weights[2]))
    )


def rank_similar_users(X: np.ndarray, top_n: int = 5) -> pd.DataFrame:
    """Apply Custom Pandas Function to Rank Top 'n' Users
    
    Parameters
    ----------
    X: np.ndarray
        input user-user similarity matrix 
    top_n: int (default = 5)
        Top number of most similar users to keep for final matrix
    """

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
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    ranking = X.apply(custom_udf).T
    ranking.columns = [f"{i+1}" for i in ranking.columns]
    ranking["user_handle"] = ranking.index
    logger.info(f"User Ranking Dataframe Shape: {ranking.shape}")
    return ranking
