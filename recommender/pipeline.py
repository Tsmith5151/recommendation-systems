import pandas as pd
from typing import List

from .parse import get_parser
from .utils.tools import timer
from .logger import get_logger
from .utils import tools as util
from .database import utils as db_main
from .engine.similarity import UserSimilarityRecommender
from .engine.factorization import RankingFactorizationRecommender

logger = get_logger(__name__)


class Pipeline:
    """Baseline Class for User-User Recommendation Pipeline

    Parameters
    ----------
    env: str
        environment for which database credentials to inherit
    outout_table: str
        name of output table to write results to SQLite3 Database
    """

    def __init__(self, env: str = None, output_table: str = None):
        self.env = env
        self.output_table = output_table

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

    def apply_user_aggregate_ranking(self, *args) -> pd.DataFrame:
        """Aggregate recommended similar users for each unique user id

        Parameters
        ----------
        args: pd.DataFrame
            input dataframe containing similar user recommendations with scores
        """
        logger.info("=" * 100)
        logger.info("Aggregating recommended users dataframe...")
        return util.agg_similar_users_mf(self.user_col, *args)

    def save(self, results: pd.DataFrame, suffix: str) -> None:
        """Write Output Data to Table in SQLite Database

        Parameters
        ----------
        df: np.ndarray
            input dataframe
        """
        logger.info("=" * 100)
        logger.info("Updating user rankings in SQLite Database...")
        db_main.write_table(self.env, self.output_table + suffix, results)


class RankingFactorizationPipeline(Pipeline):
    """Collaborative Filtering Recommender Pipeline
    Class for Recommending Similar Users using Matrix
    Factorization.

    Parameters
    ----------
    env: str
        environment for which database credentials to inherit
    top_n: int
        top most similar users for ranking
    outout_table: str
        name of output table to write results to SQLite3 Database
    """

    def __init__(
        self,
        env: str = None,
        top_n: int = None,
        output_table: str = None,
    ):
        super().__init__(env, output_table)
        self.top_n = top_n
        self.user_col = "user_handle"

    def __repr__(self):
        return """ Collaborative Filtering: Factorization Pipeline"""

    def apply_matrix_factorization(
        self,
        df: pd.DataFrame,
        name: str,
        users_col: str,
        items_col: str,
        extra_cols: List[str],
        top_n: int,
    ) -> pd.DataFrame:
        """Rank Users based on Matrix Factorization

        Parameters
        ----------
        df: np.ndarray
            input dataframe
        name: str
            name of input data source
        users: str
            name of the column in `observation_data` that corresponds to the user id.
        items_col: str
            name of the column in `observation_data` that corresponds to
            the item id.
        extras_col: Optional[str]
            side information for the items.  This SFrame must have a column with
            the same name as what is specified by the `item_id` input parameter.
            `item_data` can provide any amount of additional item-specific
            information.
        """
        logger.info("=" * 100)
        logger.info("Ranking similar users...")
        MF = RankingFactorizationRecommender(name, users_col, items_col, extra_cols)
        MF.fit(df)
        return MF.rank_users(top_n)

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

        user_interest = self.apply_matrix_factorization(
            self.interest, "interest", self.user_col, "tag", None, self.top_n
        )

        user_assessment = self.apply_matrix_factorization(
            self.assessment, "assessment", self.user_col, "tag", ["score"], self.top_n
        )

        user_courses = self.apply_matrix_factorization(
            self.course, "course_tags", self.user_col, "tag", ["view"], self.top_n
        )

        ranking = self.apply_user_aggregate_ranking(
            user_interest, user_assessment, user_courses
        )
        self.save(ranking, suffix="mf")
        logger.info("Done!")


class SimilarityRecommenderPipeline(Pipeline):
    """Collaborative Filtering Recommender Pipeline
    Class that ranks an user according to its similarity to
    other users based on similar content interactions.

    Parameters
    ----------
    env: str
        environment for which database credentials to inherit
    similarity_metric: str
        metric to measure the similarity between users
    top_n: int
        top most similar users for ranking
    outout_table: str
        name of output table to write results to SQLite3 Database
    """

    def __init__(
        self,
        env: str = None,
        similarity_metric: str = None,
        top_n: int = None,
        output_table: str = None,
    ):
        super().__init__(env, output_table)
        self.top_n = top_n
        self.user_col = "user_handle"
        self.similarity_metric = similarity_metric

    def __repr__(self):
        return """ Collaborative Filtering: Cosine Similarity Pipeline"""

    def apply_similarity(
        self,
        df: pd.DataFrame,
        name: str,
        users_col: str,
        items_col: str,
        top_n: int,
        similarity_metric: str,
    ) -> pd.DataFrame:
        """Rank Users based on Matrix Factorization

        Parameters
        ----------
        df: np.ndarray
            input dataframe
        name: str
            name of input data source
        users: str
            name of the column in `observation_data` that corresponds to the user id.
        items_col: str
            name of the column in `observation_data` that corresponds to
            the item id.
        similarity_metric: str
            metric to measure the similarity between users
        """
        logger.info("=" * 100)
        logger.info("Ranking similar users...")
        MF = UserSimilarityRecommender(name, users_col, items_col, similarity_metric)
        MF.fit(df)
        return MF.rank_users(top_n)

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

        user_interest = self.apply_similarity(
            self.interest,
            "interest",
            self.user_col,
            "tag",
            self.top_n,
            self.similarity_metric,
        )
        user_assessment = self.apply_similarity(
            self.assessment,
            "assessment",
            self.user_col,
            "tag",
            self.top_n,
            self.similarity_metric,
        )
        user_courses = self.apply_similarity(
            self.course,
            "course",
            self.user_col,
            "tag",
            self.top_n,
            self.similarity_metric,
        )

        ranking = self.apply_user_aggregate_ranking(
            user_interest, user_assessment, user_courses
        )
        self.save(ranking, suffix=self.similarity_metric)
        logger.info("Done!")


if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.method == "similarity":
        pl = SimilarityRecommenderPipeline(
            args.env, args.similarity_metric, args.top_users, args.results_table
        )
        logger.info(pl)
        pl.run()

    if args.method == "factorization":
        pl = RankingFactorizationPipeline(args.env, args.top_users, args.results_table)
        logger.info(pl)
        pl.run()
