import logging
import numpy as np
import pandas as pd
from os import environ
from typing import List

from parse import get_parser
from recommender import similarity
from recommender.database import utils as db_main
from recommender.similarity import UserSimilarityMatrix, compute_weighted_matrix, rank_similar_users
from recommender import utils
from recommender.utils import timer

logging.getLogger().setLevel(logging.INFO)


class Pipeline:
    """Pipeline Class for Recommending Similar Users"""

    def __init__(
        self,
        env: str = None,
        similarity_metric: str = None,
        weights=List[float],
        output_table: str = None,
    ):
        self.env = env
        self.weights = weights
        self.metric = similarity_metric
        self.output_table = output_table

    def __repr__(self):
        return """ Pipeline for Generating User Similarities"""

    def data_summary(self, data: dict):
        """Summary of Users/Assessments/Courses/Tags Data"""
        utils.data_summary(data)

    def apply_data_loader(self) -> None:
        """Load Users/Assessments/Course/Tags Data"""
        logging.info("=" * 50)
        logging.info("Loading Data...")
        self.data_raw = utils.load_data(self.env)

    def apply_data_prep(self):
        """Preprocess Raw Data"""
        logging.info("=" * 50)
        logging.info("Preprocessing Data...")
        self.data = utils.preprocess(self.data_raw)

    def apply_similarity_calculation(self, name: str, features: List[str], metric: str) -> np.ndarray:
        """Compute User-Items Similarity Matrix
        Steps:
            - Construct User-Item Binary Vector for each input dataset
            - Apply truncatedSVD to determine 'n' components to explain m% of total variance
            - Compute cosine similarity
        """
        logging.info("=" * 50)
        logging.info(f"Computing USER-{name.upper()} Similarity Matrix...")
        logging.info(f"Input Feautes: {features}")    
        SM = UserSimilarityMatrix(self.data[name])
        SM.get_user_item_matrix(self.data['max_users'],features)

        logging.info(f"Applying Truncated SVD: Input Shape: {SM.matrix.shape}...")
        SM._truncatedSVD()
        logging.info(f"Reduced User-Item Matrix Shape: {SM.matrix.shape}")

        # Compute pairwise user-similarity
        matrix = SM.compute_similarity(metric=metric)
        return matrix

    def apply_weighted_similarity(
        self, i: np.ndarray, a: np.ndarray, c: np.ndarray, weights: List[float]
    ) -> np.ndarray:
        """Compute Interest/Assessment/Courses Weighted Matrix"""
        logging.info("=" * 50)
        logging.info("Computing Weighted Similarity Matrix...")
        return compute_weighted_matrix(i, a, c, weights)

    def apply_user_ranking(self, df:pd.DataFrame) -> pd.DataFrame:
        """Rank Users based on Similarity Metric"""
        logging.info('=' * 50)
        logging.info("Computing Weighted Similarity Matrix...")
        return rank_similar_users(df)

    def save(self, results: pd.DataFrame) -> None:
        """Write Output Data to Table in SQLite Database"""
        logging.info("=" * 50)
        logging.info("Updating similarity matrix in SQLite Database...")
        df_results = pd.DataFrame(
                results, columns=[i for i in range(1, self.data["max_users"]+1)]
            )
        db_main.write_table(self.env, self.output_table, df_results)

    @timer
    def run(self) -> None:
        """Main method for generating user x content matrix"""
        self.apply_data_loader()
        self.data_summary(self.data_raw)
        self.apply_data_prep()
        user_interest = self.apply_similarity_calculation("interest", ["tag"], self.metric)
        user_assessment = self.apply_similarity_calculation("assessment", ["tag","score"], self.metric)
        user_courses = self.apply_similarity_calculation("course_tags", ["tag","view"], self.metric)
        weighted_matrix = self.apply_weighted_similarity(
            user_interest, user_assessment, user_courses, self.weights
        )
        rank_matrix = self.apply_user_ranking(user_assessment)
        self.save(rank_matrix)
        logging.info("Done!")


if __name__ == "__main__":
    args = get_parser().parse_args()
    pl = Pipeline(args.env, args.similarity_metric, args.weights, args.results_table)
    logging.info(pl)
    pl.run()
