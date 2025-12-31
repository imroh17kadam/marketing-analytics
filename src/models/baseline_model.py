import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.evaluation.metrics import RegressionMetrics
from sklearn.model_selection import train_test_split


class BaselineMMM:
    """
    Baseline Marketing Mix Model using Linear Regression
    """

    def __init__(self, test_size=0.2, shuffle=False, random_state=None):
        self.model = LinearRegression()
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.coef_df = None

    def split_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Chronological train/test split
        """
        return train_test_split(
            X, y,
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train Linear Regression model
        """
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.model.fit(X_train, y_train)

        # Save coefficients
        self.coef_df = pd.DataFrame({
            "feature": X.columns,
            "coefficient": self.model.coef_
        }).sort_values(by="coefficient", ascending=False)

        return self

    def predict(self, X: pd.DataFrame):
        """
        Predict using the trained model
        """
        return pd.Series(
            self.model.predict(X),
            index=X.index,
            name="predicted_sales"
        )

    def evaluate(self, X: pd.DataFrame = None, y: pd.Series = None):
        """
        Evaluate model using RMSE and R²
        If X and y are None, evaluate on test set
        """
        if X is None or y is None:
            X = self.X_test
            y = self.y_test

        y_pred = self.predict(X)
        metrics = RegressionMetrics.evaluate(y, y_pred)
        return metrics

    def get_coefficients(self):
        """
        Return feature coefficients
        """
        return self.coef_df