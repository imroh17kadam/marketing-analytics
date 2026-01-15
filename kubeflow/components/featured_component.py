from kfp.dsl import component, Input, Output, Dataset
from pathlib import Path


# @component(base_image="python:3.10")
def build_features(
    input_path: str | Path,
    output_path: str | Path
):
    """
    Build MMM features using adstock and saturation.
    """
    import pandas as pd
    from src.features.feature_builder import MediaFeatureBuilder
    from src.common.constants import channel_params

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path).dropna()

    builder = MediaFeatureBuilder(channel_params)

    df_mmm: pd.DataFrame = builder.transform(df)
    df_mmm.to_csv(output_path, index=False)

    print(f"✅ Data saved to {output_path}")