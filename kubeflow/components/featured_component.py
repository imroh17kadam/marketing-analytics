from kfp.dsl import component, Output, Dataset
import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.features.feature_builder import MediaFeatureBuilder
from src.common.constants import channel_params


# @component(
#     base_image="ml-base:latest"
# )
def build_features(
    input_data,
    output_data,
):
    df = pd.read_csv(input_data)
    df = df.dropna()

    builder = MediaFeatureBuilder(channel_params)
    df_mmm = builder.transform(df)

    output_path = Path(output_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(df_mmm)

    df_mmm.to_csv(output_path, index=False)



if __name__ == "__main__":
    input_data = PROJECT_ROOT / "artifacts" / "processed_data" / "processed_sales_data.csv"
    output_data = PROJECT_ROOT / "artifacts" / "featured_data" / "feature_engineered_sales_data.csv"
    

    build_features(input_data=str(input_data), output_data=output_data)