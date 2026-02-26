from kfp.dsl import component, Input, Output, Dataset
from pathlib import Path


@component(
    base_image="python:3.10",
    packages_to_install=["pandas"]
)
def build_features(
    input_path: Input[Dataset],
    output_path: Output[Dataset]
):
    """
    Build MMM features using adstock and saturation.
    """
    import pandas as pd
    from src.features.feature_builder import MediaFeatureBuilder
    from src.common.constants import channel_params

    df = pd.read_csv(input_path.path).dropna()

    builder = MediaFeatureBuilder(channel_params)

    df_mmm: pd.DataFrame = builder.transform(df)
    
    df_mmm.to_csv(output_path.path, index=False)

    print(f"✅ Data saved to {output_path.path}")