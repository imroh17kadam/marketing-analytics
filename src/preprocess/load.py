import os
from src.utils.logger import get_logger

import pandas as pd
from datetime import datetime
import uuid
from src.common.snowflake_client import SnowflakeClient

logger = get_logger(__name__)

def load_data(df: pd.DataFrame) -> None:
    """
    Save processed data to disk.

    Args:
        df (pd.DataFrame): Processed data
        output_path (str): Output CSV path
    """
    # Add metadata
    df = df.copy()
    df['processing_timestamp'] = datetime.utcnow()
    df['run_id'] = str(uuid.uuid4())  # Unique ID for this run

    # Connect to Snowflake
    sf = SnowflakeClient()

    # Insert rows
    for _, row in df.iterrows():
        query = f"""
        INSERT INTO PROCESSED_MARKETING_DATA (
            date, sales, tv_spend, digital_spend, search_spend, social_spend,
            promo_flag, holiday_flag, price_index, processing_timestamp, run_id
        )
        VALUES (
            '{row['date']}', {row['sales']}, {row['tv_spend']}, {row['digital_spend']},
            {row['search_spend']}, {row['social_spend']}, {row['promo_flag']},
            {row['holiday_flag']}, {row['price_index']}, '{row['processing_timestamp']}', '{row['run_id']}'
        );
        """
        sf.execute(query)

    sf.close()

    logger.info(f"Loaded {len(df)} rows into PROCESSED_MARKETING_DATA")