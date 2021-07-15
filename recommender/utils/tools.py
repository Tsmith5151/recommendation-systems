import functools
import pandas as pd
from time import time

from recommender.database import utils as db_main
from recommender.logger import get_logger

logger = get_logger(__name__)


def load_data(env: str) -> dict:
    """Load Users and Content Data from SQLite

    Parameters
    ----------
    env: str
        Environment for which database credentials to inherit
    """
    df_course = db_main.read_table(env, "select * from user_course_views")
    df_asmt = db_main.read_table(env, "select * from user_assessment_scores")
    df_interest = db_main.read_table(env, "select * from user_interests")
    df_tags = db_main.read_table(env, "select * from course_tags")

    return {
        "course": df_course,
        "assessment": df_asmt,
        "interest": df_interest,
        "tags": df_tags,
    }


def data_summary(data: dict):
    """Print Summary Metrics of Data

    Parameters
    ----------
    data: dict
        Input dictionary containing dataframes for course,
        assessment, interest, and tags, respectively.
    """
    for name, df in data.items():
        logger.info(f"\nDataframe: {name.upper()} -- Shape: {df.shape}")
        for c in df.columns:
            unique = len(df[c].unique())
            is_null = df[df[c].isnull()].shape[0]
            logger.info(f"{c} -- Unique: {unique} -- Null: {is_null}")
    return


def preprocess(data: dict) -> dict:
    """Apply series of perprocessing steps such as
    renaming columns and encoding categorical variables
    for each dataframe.

    Parameters
    ----------
    data: data
        Input dictionary containing dataframes for course,
        assessment, interest, and tags, respectively.
    """

    prep = {}
    for name, df in data.items():
        # drop null values
        df.dropna(axis=1, how="all", inplace=True)  # course tags table
        df.reset_index(drop=True, inplace=True)

        # rename columns in dataframe
        rename = {
            "interest_tag": "tag",
            "assessment_tag": "tag",
            "course_tags": "tag",
            "user_assessment_score": "score",
            "view_time_seconds": "view",
        }
        df.columns = [rename[i] if i in rename.keys() else i for i in df.columns]

        # discretize user assessment scores quantile buckets
        if any("score" in col for col in df.columns):
            df["score"] = pd.qcut(df["score"], q=3, labels=["high", "medium", "low"])

        # discretize user viewing time into quantile buckets
        if any("view" in col for col in df.columns):
            df["view"] = pd.qcut(
                df["view"], q=4, labels=["high", "medium", "low", "very low"]
            )
        # save prep dataframe
        prep[name] = df

    # add key for max users -> used for initializing user-item matrix
    prep["max_users"] = max(
        [max(v["user_handle"]) for k, v in prep.items() if "user_handle" in v.columns]
    )

    # add key containing dataframe for merged course/tags
    prep["course_tags"] = pd.merge(
        prep["course"], prep["tags"], on="course_id", how="left"
    )
    return prep


def agg_similar_users_mf(users_col: str, *args: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duplicated recommended similar users for a given input user.
    The current implementation takes is simplistic and takes the mean score;
    future work could be to perform a weighted aggregation on the various
    content tables.

    Parameters
    ----------
    users_col: str
        name of the column in `observation_data` that corresponds to the user id.
    args: pd.DataFrame
        input dataframe containing similar user recommendations with scores
    """

    # concat multiple user dataframes
    rank = pd.concat([*args], axis=0)

    # aggregate multiple similar users per unique user
    rank_agg = (
        rank.groupby([users_col, "similar"])["score"].mean().reset_index(drop=False)
    )

    # groupby and sort users
    output = (
        rank_agg.groupby(["user_handle"])
        .apply(lambda x: x.sort_values(["score"], ascending=False))
        .reset_index(drop=True)
    )

    return output


def timer(func):
    """
    Wrapper for recording execution time
    Format = H:M:S
    """

    @functools.wraps(func)
    def wrapper_time(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        end = time()
        elapsed_time = end - start
        h, r = divmod(elapsed_time, 3600)
        m, s = divmod(r, 60)
        print(f"Elapsed Time: {h:.0f}H:{m:.0f}M:{s:.0f}s")

    return wrapper_time
