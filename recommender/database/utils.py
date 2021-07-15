import os
import pandas as pd
from .manager import DatabaseManager


def drop_table(env: str, table: str) -> None:
    """Drop Table from SQLite Database

    Parameters
    ----------
    env: str
        environment for which database credentials to inherit
    table: str
        name of sql table to drop
    """
    with DatabaseManager(env) as conn:
        sql_table = """DROP TABLE IF EXISTS {table}"""
        cur = conn.cursor()
        cur.execute(sql_table)


def create_ranking_table(env: str, table: str) -> None:
    """Query User Similarity Table for SQLite Database

    Parameters
    ----------
    env: str
        environment for which database credentials to inherit
    table: str
        name of sql table create
    """
    with DatabaseManager(env) as conn:
        sql_table = f"""CREATE TABLE IF NOT EXISTS {table}\
            (user_handle TEXT NOT NULL PRIMARY KEY,\
            similar TEXT NOT NULL,\
            score REAL)
            """
        cur = conn.cursor()
        cur.execute(sql_table)


def create_ranking_table_index(
    env: str, table: str, user_alias: str = "user_handle"
) -> None:
    """Create Index on User in the Similarity Table for optimal query
    performance

        Parameters
        ----------
        env: str
            environment for which database credentials to inherit
        table: str
            name of sql table to index
    """
    with DatabaseManager(env) as conn:
        sql_table = (
            f"""CREATE UNIQUE INDEX {user_alias}_index ON {table} ({user_alias})"""
        )
        cur = conn.cursor()
        cur.execute(sql_table)


def read_table(env: str, query: str) -> pd.DataFrame:
    """Query Table from SQLite Database

    Parameters
    ----------
    env: str
        environment for which database credentials to inherit
    query: str
        sql query for reading data from SQLite3 database
    """
    with DatabaseManager(env) as conn:
        cur = conn.cursor()
        cur.execute(query)
        df = pd.DataFrame(
            cur.fetchall(), columns=[column[0] for column in cur.description]
        )
        return df


def write_table(env: str, table: str, df: pd.DataFrame) -> None:
    """Write Table from SQLite Database

    Parameters
    ----------
    env: str
        environment for which database credentials to inherit
    table: str
        name of table to write results to
    df: pd.DataFrame
        dataframe to write to SQLite3 database
    """
    with DatabaseManager(env) as conn:
        df.to_sql(name=table, con=conn, if_exists="replace", index=False)


def ingest_raw_data(env: str, data_dir: str = "data"):
    """Write .csv raw files to SQLite Database

    Parameters
    ----------
    env: str
        environment for which database credentials to inherit
    data_dir: str
        path to read data from local filesystem
    """

    csv_files = [i for i in os.listdir("data") if ".csv" in i]
    for f in csv_files:
        df = pd.read_csv(os.path.join(data_dir, f))
        with DatabaseManager(env) as conn:
            df.to_sql(
                name=f.split(".")[0],
                con=conn,
                if_exists="replace",
                index=False,
            )
