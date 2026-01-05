import pandas as pd
from src.utils.logger import get_logger

from datetime import datetime
from src.common.snowflake_client import SnowflakeClient

logger = get_logger(__name__)

def extract_data(file_path: str) -> pd.DataFrame:
    """
    Extract marketing data from CSV file.

    Args:
        file_path (str): Path to raw CSV file

    Returns:
        pd.DataFrame: Raw data
    """
    logger.info(f"Extracting data from {file_path}")

    df = pd.read_csv(file_path)

    # Add metadata
    sf_df = df.copy()

    sf_df['ingestion_timestamp'] = datetime.utcnow()
    sf_df['source'] = file_path

    # Connect to Snowflake
    sf = SnowflakeClient()

    # Insert rows one by one (simple, safe)
    for _, row in sf_df.iterrows():
        query = f"""
        INSERT INTO RAW_MARKETING_DATA (
            date_raw, sales, tv_spend, digital_spend, search_spend, social_spend,
            promo_flag, holiday_flag, price_index, ingestion_timestamp, source
        )
        VALUES (
            '{row['date']}', {row['sales']}, {row['tv_spend']}, {row['digital_spend']},
            {row['search_spend']}, {row['social_spend']}, {row['promo_flag']},
            {row['holiday_flag']}, {row['price_index']}, '{row['ingestion_timestamp']}', '{row['source']}'
        );
        """
        sf.execute(query)

    sf.close()

    logger.info(f"✅ Loaded {len(sf_df)} rows into RAW_MARKETING_DATA")
    logger.info(f"Data extracted successfully with shape {df.shape}")

    return df