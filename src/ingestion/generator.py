import numpy as np
import pandas as pd
from pathlib import Path


def generate_marketing_data(
    start_date="2021-01-03",
    n_weeks=156,
    seed=42
) -> pd.DataFrame:
    """
    Generate synthetic weekly marketing mix data
    """
    np.random.seed(seed)

    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W")
    # weekofyear = dates.weekofyear
    weekofyear = dates.isocalendar().week


    # Seasonality
    seasonal_effect = 1 + 0.15 * np.sin(2 * np.pi * weekofyear / 52)

    # Marketing spends
    tv_spend = np.random.gamma(5, 20, n_weeks)
    digital_spend = np.random.gamma(4, 15, n_weeks)
    search_spend = np.random.gamma(6, 10, n_weeks)
    social_spend = np.random.gamma(7, 8, n_weeks)

    # External factors
    promo_flag = np.random.binomial(1, 0.2, n_weeks)
    holiday_flag = np.random.binomial(1, 0.1, n_weeks)
    price_index = np.random.normal(1.0, 0.05, n_weeks)

    # Base demand
    base_sales = 30000 * seasonal_effect

    # Marketing impact (hidden truth)
    sales = (
        base_sales
        + 120 * tv_spend
        + 90 * digital_spend
        + 150 * search_spend
        + 180 * social_spend
        + promo_flag * 2000
        + holiday_flag * 1500
        - (price_index - 1) * 4000
        + np.random.normal(0, 1500, n_weeks)
    )

    df = pd.DataFrame({
        "date": dates,
        "weekofyear": weekofyear,
        "tv_spend": tv_spend,
        "digital_spend": digital_spend,
        "search_spend": search_spend,
        "social_spend": social_spend,
        "promo_flag": promo_flag,
        "holiday_flag": holiday_flag,
        "price_index": price_index,
        "sales": sales
    })

    return df


def save_data(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    df = generate_marketing_data()
    save_data(df, "data/raw/marketing_data.csv")
    print("✅ Marketing data generated and saved to data/raw/marketing_data.csv")