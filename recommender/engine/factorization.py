import pandas as pd
from typing import List
import turicreate as tc


class RankingFactorizationRecommender:
    def __init__(
        self, name: str, users_col: str, items_col: str, extras_col: List[str] = None
    ):
        """
        Ranking Factorization Recommender Class
        Source:
        https://apple.github.io/turicreate/docs/api/generated/turicreate.recommender.ranking_factorization_recommender.RankingFactorizationRecommender.html

        Parameters
        ----------
        name: str
            name of input data source
        users_col: str
            name of the column in `observation_data` that corresponds to the user id.
        items_col: str
            name of the column in `observation_data` that corresponds to
            the item id.
        extras_col: List[str]
            side information for the items.  This SFrame must have a column with
            the same name as what is specified by the `item_id` input parameter.
            `item_data` can provide any amount of additional item-specific
            information.
        """
        self.name = name
        self.users_col = users_col
        self.items_col = items_col
        self.extra_col = extras_col

    def get_dataframe(self):
        """Convert pandas DataFrame to "scalable, tabular, column-mutable
        dataframe object that can scale to big data.

        Parameters
        ----------
        None
        """
        return tc.SFrame(tc.SFrame(self.df.astype(str)))

    def fit(self, data: pd.DataFrame):
        """Fit ranking factorization recommender to learn a set of latent
        factors for each user and item and uses them to rank recommended
        items according to the likelihood of observing those pairs.

        Parameters
        ----------
        data - pd.DataFrame
            input pandas dataframe
        """
        self.sdf = self.get_dataframe(data)
        self.extra_cols = self.sdf[self.extra_cols] if not self.extra_cols else None
        self.matrix = tc.ranking_factorization_recommender.create(
            self.sdf,
            user_id=self.users_col,
            item_id=self.items_col,
            item_data=self.extra_cols,
            solver="ials",
        )

    def rank_users(self, n_top: int):
        """Factorization_recommender will return the nearest users based on
        the cosine similarity between latent user factors

        Parameters
        ----------
        top_n: int (default = 5)
            top number of most similar users to keep for final matrix
        """
        rank = (
            self.matrix.get_similar_users(self.sdf[self.users_col], n_top)
            .drop_duplicates(subset=self.sf.column_names())
            .sort(self.user_col)
            .to_dataframe()
        )
        rank["table"] = self.name
        return rank
