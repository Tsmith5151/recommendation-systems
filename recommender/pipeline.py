import numpy as np
import pandas as pd
from typing import List

from .parse import get_parser
from recommender.database import utils as db_main
from recommender.engine.similarity import (
    UserSimilarityMatrix,
    compute_weighted_matrix,
    rank_similar_users,
)
from recommender.utils import timer
from recommender.utils import tools as util

from .logger import get_logger
logger = get_logger(__name__)

class Pipeline:
    """Pipeline Class for Recommending Similar Users"""

    def __init__(
        self,
        env: str = None,
        weights=List[float],
        output_table: str = None,
    ):
        self.env = env
        self.weights = weights
        self.output_table = output_table

    def __repr__(self):
        return """ Pipeline for Generating User Similarities"""

    def data_summary(self, data: dict):
        """Summary of Users/Assessments/Courses/Tags Data"""
        util.data_summary(data)

    def apply_data_loader(self) -> None:
        """Load Users/Assessments/Course/Tags Data"""
        logger.info("=" * 100)
        logger.info("Loading Data...")
        self.data_raw = util.load_data(self.env)

    def apply_data_prep(self):
        """Preprocess Raw Data"""
        logger.info("=" * 100)
        logger.info("Preprocessing Data...")
        self.data = util.preprocess(self.data_raw)

    def apply_similarity_calculation(
        self, name: str, features: List[str]
    ) -> np.ndarray:
        """Compute User-Items Similarity Matrix
        Steps:
            - Construct User-Item Binary Vector for each input dataset
            - Apply truncatedSVD to determine 'n' components to explain m% of total variance
            - Compute cosine similarity
        """
        logger.info("=" * 100)
        logger.info(f"Computing USER-{name.upper()} Similarity Matrix...")
        logger.info(f"Input Features: {features}")
        SM = UserSimilarityMatrix(self.data[name])
        SM.get_user_item_matrix(self.data["max_users"], features)

        logger.info(f"Applying Truncated SVD: Input Shape: {SM.matrix.shape}...")
        SM._truncatedSVD()
        logger.info(f"Reduced User-Item Matrix Shape: {SM.matrix.shape}")
        matrix = SM.compute_similarity()
        return matrix

    def apply_weighted_similarity(
        self, i: np.ndarray, a: np.ndarray, c: np.ndarray, weights: List[float]
    ) -> np.ndarray:
        """Compute Interest/Assessment/Courses Weighted Matrix"""
        logger.info("=" * 100)
        logger.info("Computing Weighted Similarity Matrix...")
        return compute_weighted_matrix(i, a, c, weights)

    def apply_user_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank Users based on Similarity Distance Metric"""
        logger.info("=" * 100)
        logger.info("Ranking similar users...")
        return rank_similar_users(df)

    def save(self, results: pd.DataFrame) -> None:
        """Write Output Data to Table in SQLite Database"""
        logger.info("=" * 100)
        logger.info("Updating similarity matrix in SQLite Database...")
        db_main.write_table(self.env, self.output_table, results)

    @timer
    def run(self) -> None:
        """Main method for generating user x content matrix"""
        self.apply_data_loader()
        self.data_summary(self.data_raw)
        self.apply_data_prep()
        user_interest = self.apply_similarity_calculation("interest", ["tag"])
        user_assessment = self.apply_similarity_calculation(
            "assessment", ["tag", "score"]
        )
        user_courses = self.apply_similarity_calculation("course_tags", ["tag", "view"])
        weighted_matrix = self.apply_weighted_similarity(
            user_interest, user_assessment, user_courses, self.weights
        )
        rank_matrix = self.apply_user_ranking(weighted_matrix)
        self.save(rank_matrix)
        logger.info("Done!")


if __name__ == "__main__":
    args = get_parser().parse_args()
    pl = Pipeline(args.env, args.weights, args.results_table)
    logger.info(pl)
    pl.run()
