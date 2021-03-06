import json
import pandas as pd
from server import app
from flask import request
from database import utils as db_main

# Environment Variable
# import os
DATABASE_ENV = "dev"  # os.environ["DATABASE_ENV"]
TABLE = "user_ranking_mf"  # os.environ["RESULTS_TABLE"]


class SimilarUsers:
    def __init__(self, user: str):
        self.user = user

    def fetch_user_from_db(self) -> pd.DataFrame:
        """Fetch User Record from SQLite Database

        Parameters
        ----------
        None
        """
        query = f"select * from {TABLE} where user_handle = {self.user}"
        return db_main.read_table(DATABASE_ENV, query)

    def get_payload(self) -> json:
        """Return JSON Payload containing Input User and Top
        Similar Users with associated similarity scores

        Parameters
        ----------
        None
        """
        data = self.fetch_user_from_db()
        if data.shape[0] == 0:
            return json.dumps({self.user_id: "No records found!"})
        else:
            return json.dumps(
                {str(self.user): data[["similar", "score"]].to_dict(orient="records")}
            )


@app.route("/api/similarity/", methods=["POST", "GET"])
def get_user_similarity():
    user = json.loads(request.get_data())["user_handle"]
    SU = SimilarUsers(user)
    results = SU.get_payload()
    return results
