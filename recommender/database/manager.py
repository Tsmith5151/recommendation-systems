import os
import yaml
import sqlite3
from recommender.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Context manager for handling connections to SQLite Database"""

    def __init__(self, env: str):
        self.env = env
        self.database_path = os.path.join("data", "sqlite3")

    def __enter__(self):
        try:
            with open("configs.yml") as f:
                params = yaml.safe_load(f)[self.env]
            self.conn = sqlite3.connect(
                os.path.join(self.database_path, f"{params['database']}.db")
            )
            return self.conn
        except Exception as e:
            logger.critical(f"Error connecting to database: {str(e)}")

    def __exit__(self, e_type, e_val, _):
        try:
            self.conn.close()
        except Exception as e:
            logger.critical(f"Error closing database: {str(e)}")
