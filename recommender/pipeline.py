import numpy as np
import pandas as pd
from typing import List

from .parse import get_parser
from .utils.tools import timer
from .logger import get_logger
from .utils import tools as util
from .database import utils as db_main
from .engine.similarity import (
    UserSimilarityMatrix,
    compute_weighted_matrix,
    rank_similar_users,
)

logger = get_logger(__name__)


class CosineSimilarityPipeline:
    """Collaborative Filtering Recommender Pipeline
    Class for Recommending Similar Users using Cosine
    Similarity Distance Metric.

    Parameters
    ----------
    env: str
        environment for which database credentials to inherit
    weights: List
        list of weights for applying to similarity matrix
    top_n: int
        top most similar users for ranking
    outout_table: str
        name of output table to write results to SQLite3 Database
    """

    def __init__(
        self,
        env: str = None,
        weights: List[float] = None,
        top_n: int = None,
        output_table: str = None,
    ):
        self.env = env
        self.weights = weights
        self.output_table = output_table
        self.top_n = top_n

    def __repr__(self):
        return """ Collaborative Filtering: Cosine Similarity Pipeline"""

    def data_summary(self, data: dict):
        """Summary of Users/Assessments/Courses/Tags Data

        Parameters
        ----------
        data: dict
            Input dictionary containing dataframes for course,
            assessment, interest, and tags, respectively.
        """
        util.data_summary(data)

    def apply_data_loader(self) -> None:
        """Load Users/Assessments/Course/Tags Data

        Parameters
        ----------
        None
        """
        logger.info("=" * 100)
        logger.info("Loading Data...")
        self.data_raw = util.load_data(self.env)

    def apply_data_prep(self):
        """Preprocess Raw Data

        Parameters
        ----------
        None
        """
        logger.info("=" * 100)
        logger.info("Preprocessing Data...")
        data = util.preprocess(self.data_raw)

        self.assessment = data["assessment"]
        self.interest = data["interest"]
        self.course = data["course_tags"]
        self.max_users = data["max_users"]

    def apply_similarity_calculation(
        self, name: str, data: pd.DataFrame, max_users: int, features: List[str]
    ) -> np.ndarray:
        """Compute User-Items Similarity Matrix
        Steps:
        1.) Convert categorical columns to encodings
        2.) Construct User-Item Binary Vector for each input dataset
        3.) Apply truncatedSVD to determine 'n' components to explain m% of total variance
        4.) Compute cosine similarity

        Parameters
        ----------
        name: str
            name of input data source
        data: pd.DataFrame
            input pandas dataframe: user-items
        max_users: str
            maximum number of users for creating user-matrix matrix dimensions
        features: List[str]
            List of features columns for creating user-items matrix
        """

        logger.info("=" * 100)
        logger.info(f"Computing USER-{name.upper()} Similarity Matrix...")
        logger.info(f"Input Features: {features}")
        SM = UserSimilarityMatrix(data)
        SM.encode_categorical(features)
        SM.get_user_item_matrix(max_users, features)

        logger.info(f"Applying Truncated SVD: Input Shape: {SM.matrix.shape}...")
        SM._truncatedSVD()
        logger.info(f"Reduced User-Item Matrix Shape: {SM.matrix.shape}")
        matrix = SM.compute_similarity()
        return matrix

    def apply_weighted_similarity(
        self, i: np.ndarray, a: np.ndarray, c: np.ndarray, weights: List[float]
    ) -> np.ndarray:
        """Compute Interest/Assessment/Courses Weighted Matrix

        Parameters
        ----------
        i: str
            input user similarity matrix
        a: List[str]
            input assessment similarity matrix
        c: List[str]
            input course similarity matrix
        weights: List[str]
            input list of weights associated with each matrix
        """
        logger.info("=" * 100)
        logger.info("Computing Weighted Similarity Matrix...")
        return compute_weighted_matrix(i, a, c, weights)

    def apply_user_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank Users based on Similarity Distance Metric

        Parameters
        ----------
        df: np.ndarray
            input dataframe
        """
        logger.info("=" * 100)
        logger.info("Ranking similar users...")
        return rank_similar_users(df, self.top_n)

    def save(self, results: pd.DataFrame) -> None:
        """Write Output Data to Table in SQLite Database

        Parameters
        ----------
        df: np.ndarray
            input dataframe
        """
        logger.info("=" * 100)
        logger.info("Updating similarity matrix in SQLite Database...")
        db_main.write_table(self.env, self.output_table, results)

    @timer
    def run(self) -> None:
        """Main method for generating user x content matrix

        Parameters
        ----------
        None
        """
        self.apply_data_loader()
        self.data_summary(self.data_raw)
        self.apply_data_prep()

        user_interest = self.apply_similarity_calculation(
            "interest", self.interest, self.max_users, ["tag"]
        )
        user_assessment = self.apply_similarity_calculation(
            "assessment", self.assessment, self.max_users, ["tag", "score"]
        )
        user_courses = self.apply_similarity_calculation(
            "course_views", self.course, self.max_users, ["tag", "view"]
        )

        weighted_matrix = self.apply_weighted_similarity(
            user_interest, user_assessment, user_courses, self.weights
        )
        rank_matrix = self.apply_user_ranking(weighted_matrix)
        self.save(rank_matrix)
        logger.info("Done!")


if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.method == "cosine":
        pl = CosineSimilarityPipeline(
            args.env, args.weights, args.top_users, args.results_table
        )
        logger.info(pl)
        pl.run()
