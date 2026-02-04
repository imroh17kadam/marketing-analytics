import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.evaluation.metrics import RegressionMetrics


class BaselineMMM:
    """
    Baseline (Naive) Marketing Mix Model using Linear Regression.

    - Media variables are treated as linear
    - No adstock or saturation is applied
    - Intended only as a benchmark model
    """

    def __init__(self, test_size: float = 0.2):
        self.test_size = test_size
        self.model = LinearRegression()
        self.coef_df = None
        self.logger = get_logger(self.__class__.__name__)

        # Stored after fit
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _validate_features(self, X: pd.DataFrame) -> None:
        if X is None or X.empty:
            raise ValueError("Feature matrix X cannot be empty.")

        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            non_numeric = [
                col for col in X.columns
                if not pd.api.types.is_numeric_dtype(X[col])
            ]
            raise TypeError(f"Non-numeric columns found in X: {non_numeric}")

        if X.isnull().any().any():
            raise ValueError("X contains NaN values. Please preprocess before training.")

    def _validate_target(self, y: pd.Series) -> None:
        if y is None or len(y) == 0:
            raise ValueError("Target y cannot be empty.")

        if not pd.api.types.is_numeric_dtype(y):
            raise TypeError("Target y must be numeric.")

        if y.isnull().any():
            raise ValueError("Target y contains NaN values.")

    def split_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Chronological train-test split (no shuffling for time series data)
        """
        self.logger.info(
            f"Performing chronological train-test split (test_size={self.test_size})"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            shuffle=False
        )

        self.logger.info(
            f"Train shape: X={X_train.shape}, y={y_train.shape} | "
            f"Test shape: X={X_test.shape}, y={y_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the baseline linear MMM
        """
        self.logger.info("Starting BaselineMMM training")

        # Validate inputs
        self._validate_features(X)
        self._validate_target(y)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(X, y)

        # Train model
        self.model.fit(self.X_train, self.y_train)

        self.logger.info("Linear Regression model trained successfully")

        # Store coefficients (with ranking + absolute importance)
        self.coef_df = (
            pd.DataFrame({
                "feature": X.columns,
                "coefficient": self.model.coef_,
                "abs_coefficient": np.abs(self.model.coef_),
            })
            .sort_values(by="abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )

        self.logger.info("Model coefficients stored")
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict sales using trained model
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() first.")

        self._validate_features(X)

        self.logger.info(f"Generating predictions for data with shape {X.shape}")

        return pd.Series(
            self.model.predict(X),
            index=X.index,
            name="predicted_sales"
        )

    def evaluate(self, X: pd.DataFrame = None, y: pd.Series = None):
        """
        Evaluate model using RMSE and R².
        Defaults to test set if X and y are not provided.
        """
        if X is None or y is None:
            self.logger.info("Evaluating on held-out test set")
            X = self.X_test
            y = self.y_test
        else:
            self.logger.info("Evaluating on provided dataset")

        self._validate_features(X)
        self._validate_target(y)

        y_pred = self.predict(X)

        metrics = RegressionMetrics.evaluate(y, y_pred)

        self.logger.info(f"Evaluation results: {metrics}")
        return metrics

    def get_coefficients(self) -> pd.DataFrame:
        """
        Return model coefficients sorted by absolute importance.
        """
        if self.coef_df is None:
            raise ValueError("Model must be fitted before accessing coefficients.")

        return self.coef_df.copy()