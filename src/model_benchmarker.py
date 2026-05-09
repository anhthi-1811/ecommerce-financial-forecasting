# ==============================================================================
# MODEL BENCHMARKING PIPELINE
# Purpose:
# Benchmark and compare multiple Gradient Boosting models
# for time-series forecasting tasks.
#
# Models:
# - LightGBM
# - XGBoost
# - CatBoost
#
# Main objectives:
# - Time-series validation
# - Hyperparameter optimization
# - Fair model comparison
# - Leakage-safe evaluation
# ==============================================================================

import pandas as pd
import numpy as np
import logging

import lightgbm as lgb
import xgboost as xgb

from catboost import CatBoostRegressor 

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import TimeSeriesSplit

import optuna

from optuna.samplers import TPESampler
from typing import List, Dict


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelBenchmarker:
    """
    Advanced benchmarking pipeline for forecasting models.

    Features:
    - Time-based train/validation split
    - Hyperparameter optimization using Optuna
    - TimeSeries cross-validation
    - Leakage-safe evaluation
    - Automatic leaderboard generation
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        target_col: str = "Revenue",
        val_start_date: str = "2022-07-01"
    ):
        """
        Initialize benchmark pipeline.

        Parameters
        ----------
        data : pd.DataFrame
            Engineered dataset.

        features : List[str]
            List of selected features.

        target_col : str
            Target column for forecasting.

        val_start_date : str
            Validation split date.
        """

        self.data = data.copy()

        self.features = features

        self.target_col = target_col

        self.val_start_date = val_start_date

        # --------------------------------------------------------------
        # Categorical Features
        # --------------------------------------------------------------

        self.cat_features = [
            "day_of_week",
            "month",
            "year",
            "is_weekend",
            "is_promo_active"
        ]

        self.cat_features = [
            c for c in self.cat_features
            if c in self.features
        ]

        # --------------------------------------------------------------
        # Model Registry
        # --------------------------------------------------------------

        self.models = {

            "LightGBM": lgb.LGBMRegressor(
                random_state=42,
                n_estimators=500,
                learning_rate=0.05,
                verbose=-1
            ),

            "XGBoost": xgb.XGBRegressor(
                random_state=42,
                n_estimators=500,
                learning_rate=0.05,
                enable_categorical=True
            ),

            "CatBoost": CatBoostRegressor(
                random_state=42,
                iterations=500,
                learning_rate=0.05,
                verbose=False
            )
        }

        self.results = []

    # ==========================================================================
    # STEP 1 — PREPARE DATA
    # ==========================================================================

    def prepare_data(self) -> "ModelBenchmarker":
        """
        Prepare train and validation datasets
        using time-based splitting.
        """

        logging.info("[1/4] Preparing time-series datasets...")

        # --------------------------------------------------------------
        # Use historical data only
        # --------------------------------------------------------------

        df_past = (
            self.data[self.data["is_test"] == 0]
            .copy()
        )

        # --------------------------------------------------------------
        # Convert categorical columns
        # --------------------------------------------------------------

        for col in self.cat_features:

            df_past[col] = (
                df_past[col]
                .astype(int)
                .astype("category")
            )

        # --------------------------------------------------------------
        # Time-based split
        # --------------------------------------------------------------

        train_df = (
            df_past[df_past["Date"] < self.val_start_date]
        )

        val_df = (
            df_past[df_past["Date"] >= self.val_start_date]
        )

        self.X_train = train_df[self.features]
        self.y_train = train_df[self.target_col]

        self.X_val = val_df[self.features]
        self.y_val = val_df[self.target_col]

        logging.info(
            f"Train size: {len(self.X_train)} rows"
        )

        logging.info(
            f"Validation size: {len(self.X_val)} rows"
        )

        return self

    # ==========================================================================
    # STEP 2 — HYPERPARAMETER OPTIMIZATION
    # ==========================================================================

    def tune_with_optuna(
        self,
        model_name: str = "XGBoost",
        n_trials: int = 20
    ) -> "ModelBenchmarker":
        """
        Optimize model hyperparameters using Optuna.
        """

        logging.info(
            f"[2/4] Starting Optuna tuning for {model_name}..."
        )

        # --------------------------------------------------------------
        # OBJECTIVE FUNCTION
        # --------------------------------------------------------------

        def objective(trial):

            # ----------------------------------------------------------
            # Model parameter search spaces
            # ----------------------------------------------------------

            if model_name == "XGBoost":

                params = {

                    "n_estimators": trial.suggest_int(
                        "n_estimators",
                        300,
                        1000
                    ),

                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        0.01,
                        0.1,
                        log=True
                    ),

                    "max_depth": trial.suggest_int(
                        "max_depth",
                        4,
                        8
                    ),

                    "subsample": trial.suggest_float(
                        "subsample",
                        0.7,
                        1.0
                    ),

                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree",
                        0.7,
                        1.0
                    ),

                    "min_child_weight": trial.suggest_int(
                        "min_child_weight",
                        1,
                        7
                    ),

                    "random_state": 42,

                    "enable_categorical": True
                }

                model = xgb.XGBRegressor(**params)

            # ----------------------------------------------------------

            elif model_name == "LightGBM":

                params = {

                    "n_estimators": trial.suggest_int(
                        "n_estimators",
                        300,
                        1000
                    ),

                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        0.01,
                        0.1,
                        log=True
                    ),

                    "num_leaves": trial.suggest_int(
                        "num_leaves",
                        20,
                        100
                    ),

                    "max_depth": trial.suggest_int(
                        "max_depth",
                        4,
                        10
                    ),

                    "subsample": trial.suggest_float(
                        "subsample",
                        0.7,
                        1.0
                    ),

                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree",
                        0.7,
                        1.0
                    ),

                    "random_state": 42,

                    "verbose": -1
                }

                model = lgb.LGBMRegressor(**params)

            # ----------------------------------------------------------

            elif model_name == "CatBoost":

                params = {

                    "iterations": trial.suggest_int(
                        "iterations",
                        300,
                        1000
                    ),

                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        0.01,
                        0.1,
                        log=True
                    ),

                    "depth": trial.suggest_int(
                        "depth",
                        4,
                        8
                    ),

                    "subsample": trial.suggest_float(
                        "subsample",
                        0.7,
                        1.0
                    ),

                    "random_state": 42,

                    "verbose": False
                }

                model = CatBoostRegressor(**params)

            # ----------------------------------------------------------
            # TimeSeries Cross Validation
            # ----------------------------------------------------------

            tscv = TimeSeriesSplit(n_splits=3)

            rmse_scores = []

            for train_idx, val_idx in tscv.split(self.X_train):

                X_tr = self.X_train.iloc[train_idx]
                y_tr = self.y_train.iloc[train_idx]

                X_va = self.X_train.iloc[val_idx]
                y_va = self.y_train.iloc[val_idx]

                # ------------------------------------------------------
                # CatBoost categorical handling
                # ------------------------------------------------------

                if model_name == "CatBoost":

                    X_tr_cat = X_tr.copy()
                    X_va_cat = X_va.copy()

                    for col in self.cat_features:

                        X_tr_cat[col] = (
                            X_tr_cat[col]
                            .astype(str)
                        )

                        X_va_cat[col] = (
                            X_va_cat[col]
                            .astype(str)
                        )

                    # Early stopping
                    model.fit(
                        X_tr_cat,
                        y_tr,

                        cat_features=self.cat_features,

                        eval_set=[(X_va_cat, y_va)],

                        early_stopping_rounds=50,

                        verbose=False
                    )

                    preds = model.predict(X_va_cat)

                else:

                    model.fit(X_tr, y_tr)

                    preds = model.predict(X_va)

                # ------------------------------------------------------
                # RMSE Evaluation
                # ------------------------------------------------------

                rmse = np.sqrt(
                    mean_squared_error(y_va, preds)
                )

                rmse_scores.append(rmse)

            return np.mean(rmse_scores)

        # --------------------------------------------------------------
        # Run Optuna Study
        # --------------------------------------------------------------

        optuna.logging.set_verbosity(
            optuna.logging.WARNING
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42)
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True
        )

        # --------------------------------------------------------------
        # Save best tuned model
        # --------------------------------------------------------------

        best_params = study.best_params

        best_params["random_state"] = 42

        if model_name == "XGBoost":

            best_params["enable_categorical"] = True

            self.models[
                f"{model_name}_Optuna_Tuned"
            ] = xgb.XGBRegressor(**best_params)

        elif model_name == "LightGBM":

            best_params["verbose"] = -1

            self.models[
                f"{model_name}_Optuna_Tuned"
            ] = lgb.LGBMRegressor(**best_params)

        elif model_name == "CatBoost":

            best_params["verbose"] = False

            self.models[
                f"{model_name}_Optuna_Tuned"
            ] = CatBoostRegressor(**best_params)

        logging.info(
            f"Best tuned model added: "
            f"{model_name}_Optuna_Tuned"
        )

        return self

    # ==========================================================================
    # STEP 3 — RUN EXPERIMENTS
    # ==========================================================================

    def run_experiments(self) -> "ModelBenchmarker":
        """
        Train and evaluate all models.
        """

        logging.info(
            f"[3/4] Benchmarking target: "
            f"{self.target_col}"
        )

        for name, model in self.models.items():

            logging.info(
                f"Training model: {name}"
            )

            # ----------------------------------------------------------
            # CatBoost handling
            # ----------------------------------------------------------

            if "CatBoost" in name:

                X_train_cat = self.X_train.copy()
                X_val_cat = self.X_val.copy()

                for col in self.cat_features:

                    X_train_cat[col] = (
                        X_train_cat[col]
                        .astype(str)
                    )

                    X_val_cat[col] = (
                        X_val_cat[col]
                        .astype(str)
                    )

                model.fit(
                    X_train_cat,
                    self.y_train,

                    cat_features=self.cat_features
                )

                preds = model.predict(X_val_cat)

            else:

                model.fit(
                    self.X_train,
                    self.y_train
                )

                preds = model.predict(
                    self.X_val
                )

            # ----------------------------------------------------------
            # Inverse log transformation 
            # ----------------------------------------------------------

            if "_Log" in self.target_col:

                preds_real = np.expm1(preds) 

                actuals_real = np.expm1(
                    self.y_val
                )

            else:

                preds_real = preds

                actuals_real = self.y_val

            # ----------------------------------------------------------
            # Evaluation metrics
            # ----------------------------------------------------------

            mae = mean_absolute_error(
                actuals_real,
                preds_real
            )

            rmse = np.sqrt(
                mean_squared_error(
                    actuals_real,
                    preds_real
                )
            )

            r2 = r2_score(
                actuals_real,
                preds_real
            )

            # ----------------------------------------------------------
            # Store results
            # ----------------------------------------------------------

            self.results.append({

                "Model": name,

                "MAE": round(mae, 2),

                "RMSE": round(rmse, 2),

                "R2_Score": round(r2, 4)
            })

        logging.info(
            "Benchmarking completed successfully."
        )

        return self

    # ==========================================================================
    # STEP 4 — LEADERBOARD
    # ==========================================================================

    def show_leaderboard(self) -> pd.DataFrame:
        """
        Display ranked model leaderboard.
        """

        logging.info("[4/4] Generating leaderboard...")

        leaderboard = (
            pd.DataFrame(self.results)
            .sort_values(by="RMSE")
            .reset_index(drop=True)
        )

        print("\nMODEL LEADERBOARD")
        print(leaderboard)

        return leaderboard 