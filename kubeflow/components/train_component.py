from kfp.dsl import Model
import pandas as pd

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from sklearn.linear_model import Ridge
from src.evaluation.metrics import RegressionMetrics
from src.common.constants import features_mmm
from src.utils.logger import get_logger


# @component(
#   base_image="ml-base:latest"
# )
def train_model(
    input_data,
    model_path,
    test_output,
    target: str = "sales",
    alpha: float = 1.0,
    test_size: float = 0.2
):
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split

    logger = get_logger(__name__)

    df_mmm = pd.read_csv(input_data)

    # Train-test split
    X = df_mmm[features_mmm]
    y = df_mmm[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # Train model
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    test_output_dir = Path(test_output)
    test_output_dir.mkdir(parents=True, exist_ok=True)

    X_test.to_csv(test_output_dir / "X_test.csv", index=False)
    y_test.to_csv(test_output_dir / "y_test.csv", index=False)    



if __name__ == "__main__":
    input_data = PROJECT_ROOT / "artifacts" / "featured_data" / "feature_engineered_sales_data.csv"
    model_path = PROJECT_ROOT / "artifacts" / "model" / "ridge_mmm_model.pkl"
    test_output = PROJECT_ROOT / "artifacts" / "evaluation_data"

    train_model(input_data=str(input_data), model_path=model_path, test_output=test_output, target="sales")