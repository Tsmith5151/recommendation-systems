import pandas as pd
from typing import List
import turicreate as tc


class RankingFactorizationRecommender:
    def __init__(
        self, name: str, users_col: str, items_col: str, extras_col: List[str] = None
    ):
        self.name = name
        self.users_col = users_col
        self.items_col = items_col
        self.extra_col = extras_col

    def get_dataframe(self):
        """Convert pandas DataFrame to "scalable, tabular, column-mutable
        dataframe object that can scale to big data.
        """
        return tc.SFrame(tc.SFrame(self.df.astype(str)))

    def fit(self, data: pd.DataFrame):
        """Fit ranking factorization recommender to learn a set of latent
        factors for each user and item and uses them to rank recommended
        items according to the likelihood of observing those pairs.
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
        """
        rank = (
            self.matrix.get_similar_users(self.sdf[self.users_col], n_top)
            .drop_duplicates(subset=self.sf.column_names())
            .sort(self.user_col)
            .to_dataframe()
        )
        rank["table"] = self.name
        return rank
