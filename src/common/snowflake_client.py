import os
import yaml
import snowflake.connector
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def resolve(value):
    """Resolve ${VAR} from environment"""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1])
    return value
    

class SnowflakeClient:
    def __init__(self, config_path="config/snowflake_config.yaml"):

        # ✅ Always load .env FIRST
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")

        # ✅ Load YAML
        config_file = project_root / config_path
        with open(config_file, "r") as f:
            raw_config = yaml.safe_load(f)

        # ✅ Resolve env variables manually
        config = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "role": raw_config["snowflake"]["role"],
            "warehouse": raw_config["snowflake"]["warehouse"],
            "database": raw_config["snowflake"]["database"],
            "schema": raw_config["snowflake"]["schema"],
        }

        # 🚨 Safety check
        for k, v in config.items():
            if v is None:
                raise ValueError(f"Missing environment variable for {k}")

        self.conn = snowflake.connector.connect(
            account=config["account"],
            user=config["user"],
            password=config["password"],
            role=config["role"],
            warehouse=config["warehouse"],
            database=config["database"],
            schema=config["schema"],
        )

    def execute(self, query, params=None):
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
        finally:
            cursor.close()

    def fetchall(self, query):
        cursor = self.conn.cursor()
        try:
            cursor.execute(query)
            return cursor.fetchall()
        finally:
            cursor.close()

    def close(self):
        self.conn.close()