import os
import json
import pandas as pd
from server import app
from flask import request
from recommender.database import utils as db_main

DATABASE_ENV = os.environ["DATABASE_ENV"]
TABLE = os.environ["RESULTS_TABLE"]


class SimilarUsers:
    def __init__(self, user: str):
        self.user = user

    def fetch_user_from_db(self) -> pd.DataFrame:
        """Fetch User Record from SQLite Database"""
        query = f"select * from {TABLE} where user_handle = {self.user}"
        print("Table", TABLE)
        return db_main.read_table(DATABASE_ENV, query)

    def get_payload(self) -> json:
        """Return JSON Payload containing Input User and Top
        Similar Users with associated similarity scores"""
        data = self.fetch_user_from_db()
        if data.shape[0] == 0:
            return {self.user_id: "No records found!"}
        else:
            return {str(self.user): list(data.loc[0].values.flatten()[:-1])}


@app.route("/api/similarity/", methods=["POST", "GET"])
def get_user_similarity():
    user = json.loads(request.get_data())["user_handle"]
    SU = SimilarUsers(user)
    results = SU.get_payload()
    return results
