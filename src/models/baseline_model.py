import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from src.evaluation.metrics import RegressionMetrics


class BaselineMMM:
    """
    Baseline (Naive) Marketing Mix Model using Linear Regression.

    NOTE:
    - Media variables are treated as linear
    - No adstock or saturation is applied
    - Intended only as a benchmark model
    """

    def __init__(self, test_size: float = 0.2):
        self.test_size = test_size
        self.model = LinearRegression()
        self.coef_df = None

        # Stored after fit
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Chronological train-test split (no shuffling for time series data)
        """
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            shuffle=False
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the baseline linear MMM
        """
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(X, y)

        self.model.fit(self.X_train, self.y_train)

        # Store coefficients
        self.coef_df = (
            pd.DataFrame({
                "feature": X.columns,
                "coefficient": self.model.coef_
            })
            .sort_values(by="coefficient", ascending=False)
            .reset_index(drop=True)
        )

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict sales using trained model
        """
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
            X = self.X_test
            y = self.y_test

        y_pred = self.predict(X)
        return RegressionMetrics.evaluate(y, y_pred)

    def get_coefficients(self) -> pd.DataFrame:
        """
        Return model coefficients
        """
        if self.coef_df is None:
            raise ValueError("Model must be fitted before accessing coefficients.")

        return self.coef_df