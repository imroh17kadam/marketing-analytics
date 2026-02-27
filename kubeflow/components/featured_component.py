
from kfp.dsl import Dataset, Input, Output, component


@component(base_image="python:3.10", packages_to_install=["pandas"])
def build_features(input_path: Input[Dataset], output_path: Output[Dataset]):
    """
    Build MMM features using adstock and saturation.
    """
    import pandas as pd

    from src.marketing_analytics.common.constants import channel_params
    from src.marketing_analytics.features.feature_builder import MediaFeatureBuilder

    df = pd.read_csv(input_path.path).dropna()

    builder = MediaFeatureBuilder(channel_params)

    df_mmm: pd.DataFrame = builder.transform(df)

    df_mmm.to_csv(output_path.path, index=False)

    print(f"✅ Data saved to {output_path.path}")
