# ==============================================================================
# FINANCIAL FORECASTING PIPELINE
# Purpose:
# Forecast future Revenue and COGS using optimized Tree-based models.
#
# Main objectives:
# - Multi-target forecasting with target-specific optimized models
# - Apply log transformation for stable learning
# - Prevent overfitting
# - Generate prediction-ready submission files
# ==============================================================================

import copy
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Dict, Optional

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class FinancialForecaster:
    """
    End-to-end financial forecasting pipeline.

    Features:
    - Multi-target forecasting using custom tuned models
    - Log-transformed training
    - Leakage-safe prediction
    - Automatic submission export
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        targets: List[str] = ["Revenue_Log", "COGS_Log"],
        custom_models: Optional[Dict] = None
    ):
        """
        Initialize forecasting pipeline.

        Parameters
        ----------
        data : pd.DataFrame
            Engineered dataset.
        features : List[str]
            Selected model features.
        targets : List[str]
            Forecasting targets.
        custom_models : Dict
            Dictionary mapping clean target names to their specific optimized models.
            Example: {'Revenue': best_lgb, 'COGS': best_catboost}
        """
        self.data = data.copy()
        self.features = features
        self.targets = targets
        self.predictions: Dict = {}
        self.trained_models: Dict = {}
        
        self.custom_models = custom_models or {}
        
        if self.custom_models:
            logging.info("Initialized with custom optimized models from Main Pipeline.")
        else:
            base_model = lgb.LGBMRegressor(
                random_state=42, n_estimators=450, learning_rate=0.04,
                num_leaves=31, colsample_bytree=0.8, verbose=-1
            )
            # Duplicate the base model for all targets as fallback
            self.custom_models = {target.replace("_Log", ""): copy.deepcopy(base_model) for target in self.targets}
            logging.info("Initialized with default Base LightGBM models.")

    # ==========================================================================
    # STEP 1 — PREPARE DATA
    # ==========================================================================
    def prepare_data(self) -> "FinancialForecaster":
        """
        Prepare training and forecasting datasets.
        """
        logging.info("[1/3] Preparing forecasting datasets...")

        # Fill missing financial values safely
        self.data["Revenue"] = self.data["Revenue"].fillna(0)
        self.data["COGS"] = self.data["COGS"].fillna(0)

        # Apply log1p transformation to handle peaks and zero values
        self.data["Revenue_Log"] = np.log1p(self.data["Revenue"])
        self.data["COGS_Log"] = np.log1p(self.data["COGS"])

        cat_features = [
            c for c in ["day_of_week", "month", "is_weekend", "is_promo_active", "is_payday"]
            if c in self.features
        ]

        for col in cat_features:
            self.data[col] = self.data[col].astype(int).astype("category")

        # Ensure 'year' is treated as integer to capture growth trend
        if "year" in self.data.columns:
            self.data["year"] = self.data["year"].astype(int)

        # ----------------------------------------------------------------------
        # Split historical and future datasets
        # ----------------------------------------------------------------------
        self.df_past = self.data[self.data["is_test"] == 0].copy()
        self.df_future = self.data[self.data["is_test"] == 1].copy()

        # Create feature matrices
        self.X_train = self.df_past[self.features]
        self.X_test = self.df_future[self.features]

        logging.info(f"Training rows: {len(self.X_train)} | Forecast rows: {len(self.X_test)}")
        return self 

    # ==========================================================================
    # STEP 2 — TRAIN MODELS & GENERATE FORECASTS
    # ==========================================================================
    def train_and_predict(self) -> "FinancialForecaster":
        """
        Train the champion models and generate future predictions WITH EARLY STOPPING.
        """
        logging.info("[2/3] Training forecasting models with Early Stopping...")

        for target in self.targets:
            logging.info(f"Training target: {target}")
            
            # Retrieve the appropriate Champion model
            actual_target_name = target.replace('_Log', '')
            model = self.custom_models[actual_target_name]

            # Split Train/Val for Early Stopping (85/15 chronological split)
            split_idx = int(len(self.X_train) * 0.85)
            
            X_sub_train = self.X_train.iloc[:split_idx]
            y_sub_train = self.df_past[target].iloc[:split_idx]
            
            X_sub_val = self.X_train.iloc[split_idx:]
            y_sub_val = self.df_past[target].iloc[split_idx:]

            # Train with Early Stopping
            try:
                model.fit(
                    X_sub_train, y_sub_train,
                    eval_set=[(X_sub_val, y_sub_val)],
                    early_stopping_rounds=50,
                    verbose=False 
                )
            except TypeError:
                # Fallback for models or older sklearn versions that don't support early stopping natively
                logging.warning(f"Early stopping unsupported via standard fit for {target}. Falling back to default fit.")
                model.fit(self.X_train, self.df_past[target])

            # Generate log-scale predictions
            log_preds = model.predict(self.X_test)
            self.trained_models[target] = model

            # Reverse log1p transformation (e^x - 1) and prevent negative predictions
            self.predictions[actual_target_name] = np.maximum(0, np.expm1(log_preds))

        return self 
    
    # ==========================================================================
    # STEP 3 — EXPORT FORECAST REPORT
    # ==========================================================================
    def generate_report(self, export_path: str = "output/submission.csv") -> pd.DataFrame:
        """
        Generate final forecasting report and export submission CSV.
        """
        logging.info("[3/3] Generating submission report...")

        submission = pd.DataFrame({
            "Date": pd.to_datetime(self.df_future["Date"]).dt.strftime("%Y-%m-%d"),
            "Revenue": self.predictions.get("Revenue", 0),
            "COGS": self.predictions.get("COGS", 0)
        })

        submission.to_csv(export_path, index=False)
        logging.info(f"Submission file saved to: {export_path}")
        
        from IPython.display import display # Import locally to avoid issues in pure python scripts
        print("\n--- FINAL FORECAST PREVIEW ---")
        display(submission.head(10))

        return submission 