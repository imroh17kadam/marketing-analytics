from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List
import uuid

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

# Project-relative imports
from src.ingestion.ingestion import DataIngestion
from src.features.feature_builder import MediaFeatureBuilder
from src.preprocess.preprocess import Preprocessor   
from src.evaluation.metrics import RegressionMetrics
from src.utils.logger import get_logger
from src.common.snowflake_client import SnowflakeClient
from config_loader import ConfigLoader

# REUSING YOUR EXISTING MODEL COMPONENTS
from src.models.baseline_model import BaselineMMM
from src.models.mmm_model import RegularizedMMM


class TrainPipeline:
    """
    End-to-end pipeline for training MMM model with MLflow tracking & registry.
    Now:
    - Includes a preprocess step
    - Uses existing model components instead of re-writing training logic here
    """

    def __init__(
        self,
        query: str,
        channel_params: Dict[str, dict],
        features_mmm: List[str],
        target: str = "sales",
        alpha: float = 1.0,
        test_size: float = 0.2,
        experiment_name: str = "marketing-mmm",
        flag_columns: List[str] = None,
        log_transform_cols: List[str] = None
    ):
        self.query = query
        self.channel_params = channel_params
        self.features_mmm = features_mmm
        self.target = target
        self.alpha = alpha
        self.test_size = test_size
        self.experiment_name = experiment_name
        self.flag_columns = flag_columns or []
        self.log_transform_cols = log_transform_cols or []

        # Load environment-aware config
        self.config = ConfigLoader()
        self.logger = get_logger(self.__class__.__name__)

        # Resolve artifact path from config
        self.artifacts_dir = Path(self.config.get("paths")["artifacts"])
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _setup_mlflow(self) -> str:
        """
        Configure MLflow tracking based on environment.
        Handles cases where experiment was deleted.
        Returns a generated run_id.
        """

        tracking_uri = self.config.get("mlflow")["tracking_uri"]

        self.logger.info(f"Setting MLflow tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)

        # ---- FIX: Handle deleted / missing experiment properly ----
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

            if experiment is None:
                # Experiment does not exist → create new
                self.logger.info(f"Experiment '{self.experiment_name}' not found. Creating new one.")
                mlflow.create_experiment(self.experiment_name)

            elif experiment.lifecycle_stage == "deleted":
                # Experiment was deleted → restore it
                self.logger.warning(
                    f"Experiment '{self.experiment_name}' was deleted. Restoring it."
                )
                mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)

            # Now safely set experiment
            mlflow.set_experiment(self.experiment_name)

        except Exception as e:
            self.logger.error(f"MLflow experiment setup failed: {str(e)}")
            raise

        run_id = str(uuid.uuid4())
        return run_id

    def _load_data(self) -> pd.DataFrame:
        self.logger.info("Loading data from Snowflake")

        ingestion = DataIngestion(source="file")
        df = ingestion.load()

        if df.empty:
            raise ValueError("Loaded dataset is empty from Snowflake")

        self.logger.info(f"Loaded dataset with shape: {df.shape}")
        return df

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ CORRECT PREPROCESS STEP USING YOUR ACTUAL METHODS
        1) Fill missing values
        2) Encode flags
        3) Apply log transform
        """

        self.logger.info("Starting data preprocessing")

        # Step 1: Fill missing values
        df_clean = Preprocessor.fill_missing(df, strategy="mean")

        # Step 2: Encode flags (if provided)
        if self.flag_columns:
            df_clean = Preprocessor.encode_flags(df_clean, self.flag_columns)

        # Step 3: Log transform numeric columns (if provided)
        if self.log_transform_cols:
            df_clean = Preprocessor.log_transform(df_clean, self.log_transform_cols)

        self.logger.info(f"Preprocessed dataset shape: {df_clean.shape}")
        return df_clean

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Applying adstock + saturation transforms")

        builder = MediaFeatureBuilder(self.channel_params)
        df_mmm = builder.transform(df)

        missing_feats = [f for f in self.features_mmm if f not in df_mmm.columns]
        if missing_feats:
            raise ValueError(f"Missing required MMM features: {missing_feats}")

        return df_mmm

    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        ✅ USES YOUR EXISTING MMM MODEL COMPONENT
        Instead of re-writing Ridge training here
        """

        self.logger.info("Training MMM Model using src.models.mmm_model")

        mmm_model = RegularizedMMM(alpha=self.alpha)
        model = mmm_model.fit(X_train, y_train)

        return model

    def _save_local_model(self, model) -> Path:
        model_path = self.artifacts_dir / "ridge_mmm_model.pkl"
        joblib.dump(model, model_path)

        self.logger.info(f"Saved model locally at: {model_path}")
        return model_path

    def _save_coefficients(
        self, X: pd.DataFrame, model, run_id: str
    ) -> Path:
        """
        Save coefficients properly when using RegularizedMMM
        """

        # ---- IMPORTANT FIX: Handle RegularizedMMM properly ----
        if hasattr(model, "get_coefficients"):
            coef_df = model.get_coefficients()[["feature", "coefficient"]]
        else:
            # Fallback (if you ever switch back to sklearn model)
            coef_df = pd.DataFrame({
                "feature": X.columns,
                "coefficient": model.coef_,
            })

        coef_df = (
            coef_df
            .sort_values(by="coefficient", ascending=False)
            .reset_index(drop=True)
        )

        coef_df["model_name"] = "ridge_mmm_v1"
        coef_df["run_id"] = run_id
        coef_df["created_at"] = datetime.utcnow()

        coef_df = coef_df[
            ["model_name", "feature", "coefficient", "run_id", "created_at"]
        ]

        coef_path = self.artifacts_dir / "ridge_mmm_coefficients.csv"
        coef_df.to_csv(coef_path, index=False)

        self.logger.debug(f"Saved coefficients at: {coef_path}")
        return coef_path, coef_df

    def _write_coeffs_to_snowflake(self, coef_df: pd.DataFrame) -> None:
        self.logger.info("Writing coefficients to Snowflake: MMM_COEFFICIENTS")

        sf = SnowflakeClient()

        insert_query = """
            INSERT INTO MMM_COEFFICIENTS (
                model_name,
                feature,
                coefficient,
                run_id,
                created_at
            )
            VALUES (%s, %s, %s, %s, %s)
        """

        for _, row in coef_df.iterrows():
            params = (
                row["model_name"],
                row["feature"],
                float(row["coefficient"]),
                row["run_id"],
                row["created_at"].isoformat(),
            )
            sf.execute(insert_query, params)

        sf.close()
        self.logger.info("Successfully stored coefficients in Snowflake")

    def run(self) -> Tuple[object, Dict[str, float]]:
        self.logger.info("===== TRAINING PIPELINE STARTED =====")

        run_id = self._setup_mlflow()

        with mlflow.start_run(run_name=f"ridge-mmm-{run_id}") as run:

            # Log key params
            mlflow.log_param("alpha", self.alpha)
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("num_features", len(self.features_mmm))
            mlflow.set_tag("query", self.query)
            mlflow.set_tag("model_type", "Ridge-MMM")

            # 1️⃣ Load data
            df = self._load_data()

            # 2️⃣ PREPROCESS STEP (NEW)
            df_clean = self._preprocess_data(df)

            # 3️⃣ Feature engineering
            df_mmm = self._build_features(df_clean)

            # 4️⃣ Train-test split (chronological)
            X = df_mmm[self.features_mmm]
            y = df_mmm[self.target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=False
            )

            # 5️⃣ Train model using YOUR COMPONENT
            model = self._train_model(X_train, y_train)

            # 6️⃣ Evaluate
            y_pred = model.predict(X_test)
            metrics = RegressionMetrics.evaluate(y_test, y_pred)

            self.logger.info(f"Model evaluation: {metrics}")

            # Log metrics to MLflow
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            # 7️⃣ Save locally
            local_model_path = self._save_local_model(model)

            # 8️⃣ Log model to MLflow + Register
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="MMM-Ridge-Model",
            )

            # 9️⃣ Save & log coefficients
            coef_path, coef_df = self._save_coefficients(X, model, run.info.run_id)

            mlflow.log_artifact(str(coef_path), artifact_path="coefficients")

            # 🔟 Write coefficients to Snowflake
            # self._write_coeffs_to_snowflake(coef_df)

            self.logger.debug(f"MLflow Run ID: {run.info.run_id}")
            self.logger.info("===== TRAINING PIPELINE COMPLETED =====")

            return model, metrics