import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.evaluation.metrics import RegressionMetrics

class RegularizedMMM:
    """
    Regularized Marketing Mix Model using Ridge Regression
    """

    def __init__(
        self,
        alpha: float = 1.0,
        test_size: float = 0.2,
        shuffle: bool = False,
        random_state=None,
    ):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.logger = get_logger(self.__class__.__name__)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.coef_df = None

    def _validate_features(self, X: pd.DataFrame) -> None:
        if X is None or X.empty:
            raise ValueError("Feature matrix X cannot be empty.")

        non_numeric = [
            col for col in X.columns
            if not pd.api.types.is_numeric_dtype(X[col])
        ]
        if non_numeric:
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
        Train-test split (can be chronological if shuffle=False)
        """
        self.logger.info(
            f"Splitting data | test_size={self.test_size}, shuffle={self.shuffle}"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        self.logger.info(
            f"Train shape: X={X_train.shape}, y={y_train.shape} | "
            f"Test shape: X={X_test.shape}, y={y_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train Ridge Regression
        """
        self.logger.info(f"Starting RegularizedMMM training (alpha={self.alpha})")

        # Validate inputs
        self._validate_features(X)
        self._validate_target(y)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(X, y)

        # Train model
        self.model.fit(self.X_train, self.y_train)

        self.logger.info("Ridge Regression model trained successfully")

        # Store coefficients (sorted by importance)
        self.coef_df = (
            pd.DataFrame({
                "feature": self.X_train.columns,
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
        Predict using trained Ridge model
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call fit() first.")

        self._validate_features(X)

        self.logger.info(f"Generating predictions for data with shape {X.shape}")

        return pd.Series(
            self.model.predict(X),
            index=X.index,
            name="predicted_sales",
        )

    def evaluate(self, X: pd.DataFrame = None, y: pd.Series = None):
        """
        Evaluate model on RMSE and R².
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
        Return feature coefficients sorted by absolute importance.
        """
        if self.coef_df is None:
            raise ValueError("Model must be fitted before accessing coefficients.")

        return self.coef_df.copy()

    def channel_contribution(self, channel_cols: list) -> pd.DataFrame:
        """
        Calculate total contribution per channel
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before calculating contributions.")

        missing_cols = [c for c in channel_cols if c not in self.X_train.columns]
        if missing_cols:
            raise KeyError(f"Channels not found in training data: {missing_cols}")

        self.logger.info("Calculating channel contributions")

        contributions = {}
        for col in channel_cols:
            coef = self.model.coef_[self.X_train.columns.get_loc(col)]
            contributions[col] = coef * self.X_train[col].sum()

        contrib_df = (
            pd.DataFrame.from_dict(
                contributions,
                orient="index",
                columns=["total_contribution"],
            )
            .sort_values(by="total_contribution", ascending=False)
        )

        self.logger.info("Channel contributions computed successfully")
        return contrib_df