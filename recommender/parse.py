import argparse


def get_parser():
    """Arguments for Machine Learning Pipeline"""
    parser = argparse.ArgumentParser(description="User Recommendation Application")
    parser.add_argument(
        "--data_path",
        default="data",
        type=str,
        help="data path for input files to read from disk",
    )
    parser.add_argument(
        "--env",
        default="dev",
        choices=["dev", "prod"],
        type=str,
        help="Environment for SQLite Database: dev/prod",
    )
    parser.add_argument(
        "--method",
        default="cosine-distance",
        choices=["cosine-distance", "matrix-factorization"],
        type=str,
        help="Collaborative filtering method",
    )
    parser.add_argument(
        "--weights",
        default=["0.60", "0.30", "0.20"],
        action="store",
        nargs="+",
        help="Weights for similarity matrix: interest,assessment,tags",
    )
    parser.add_argument(
        "--results_table",
        default=None,
        type=str,
        help="SQLite3 table containing user similarity metrics",
    )
    parser.add_argument(
        "--top_users",
        default=5,
        type=int,
        help="top most similar users for ranking",
    )
    parser.add_argument(
        "--user_id",
        default=None,
        type=str,
        help="unique user id for scoring similarities",
    )
    parser.add_argument(
        "--hostname",
        default="0.0.0.0",
        type=str,
        help="hostname for serving Flask application",
    )
    parser.add_argument(
        "--port",
        default=5000,
        type=int,
        help="port for serving Flask application",
    )
    return parser
