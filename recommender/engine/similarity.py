import pandas as pd
import turicreate as tc


from recommender.logger import get_logger
logger = get_logger(__name__)


class UserSimilarityRecommender:
    def __init__(
        self,
        name: str,
        users_col: str,
        items_col: str,
        similarity_metric: str = None,
    ):
        """
        User Similarity Recommender Recommender Class
        
        Source:
        https://apple.github.io/turicreate/docs/api/generated/turicreate
        
        Note: get_similar_users currently not supported for item similarity
        models. As a workaround, to get the neighborhood of users, train a
        model with the items and users reversed, then call get_similar_items.

        Parameters
        ----------
        name: str
            name of input data source
        users_col: str
            name of the column in `observation_data` that corresponds to the user id.
        items_col: str
            name of the column in `observation_data` that corresponds to
            the item id.
        similarity_metric: str
            metric to measure the similarity between users
        """
        self.name = name
        self.users_col = users_col
        self.items_col = items_col
        self.similarity_metric = similarity_metric

    def convert_dataframe(self, df: pd.DataFrame) -> tc.SFrame:
        """Convert pandas DataFrame to "scalable, tabular, column-mutable
        dataframe object that can scale to big data.

        Parameters
        ----------
        None
        """
        return tc.SFrame(tc.SFrame(df.astype(str)))

    def fit(self, data: pd.DataFrame):
        """Model calculates similarity between users using the observations of
        tags/items interacted between users. The model scores an user 'j' for
        item 'k' using a weighted average of the items previous observations.
        By default, cosine will be used to measure the similarity between two users.

        Parameters
        ----------
        data - pd.DataFrame
            input pandas dataframe
        """
        self.sdf = self.convert_dataframe(data)
        self.matrix = tc.recommender.item_similarity_recommender.create(
            self.sdf,
            user_id=self.items_col,
            item_id=self.users_col,
            similarity_type=self.similarity_metric,
        )

    def rank_users(self, n_top: int) -> pd.DataFrame:
        """
        Obtain the most similar items for each item in items.
        """
        rank = (
            self.matrix.get_similar_items(self.sdf[self.users_col], n_top)
            .to_dataframe()
            .drop_duplicates()
        )

        # groupby and sort scores
        rank["table"] = self.name

        # format score
        rank["score"] = rank["score"].round(5)
        return rank
