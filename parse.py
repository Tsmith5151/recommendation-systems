import argparse


def get_parser() -> argparse:
    """Arguments for Machine Learning Pipeline"""
    parser = argparse.ArgumentParser(
        description="PluralSight - Machine Learning Challenge"
    )
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
        "--results_table",
        default=None,
        type=str,
        help="SQLite3 table containing user similarity metrics",
    )
    parser.add_argument(
        "--hostname",
        default="0.0.0.0",
        type=str,
        help="hostname for serving Flask application",
    )
    parser.add_argument(
        "--port", default=5000, type=int, help="port for serving Flask application"
    )
    return parser