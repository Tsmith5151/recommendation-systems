import logging
import functools
import pandas as pd
from time import time

from recommender.database import utils as db_main

logging.getLogger().setLevel(logging.INFO)


def load_data(env: str) -> dict:
    """Load Users and Content Data from SQLite"""

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
    """Print Summary Metrics of Data"""
    for name, df in data.items():
        logging.info(f"\nDataframe: {name.upper()} -- Shape: {df.shape}")
        for c in df.columns:
            unique = len(df[c].unique())
            is_null = df[df[c].isnull()].shape[0]
            logging.info(f"{c} -- Unique: {unique} -- Null: {is_null}")
    return


def preprocess(data: dict) -> dict:
    """Preprocess input DataFrames"""
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

        # encode categorical columns
        cat_cols = ["tag", "score", "view", "level"]
        for col in df.columns:
            if col in cat_cols:
                df[col] = pd.Categorical(df[col]).codes

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


def timer(func):
    """Wrapper for recording execution time"""

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
