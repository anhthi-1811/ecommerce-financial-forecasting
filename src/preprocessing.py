import pandas as pd
import numpy as np
import warnings
import logging
from typing import Optional

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class SalesDataPipeline: 
    """
    Sales Data Processing Pipeline

    Workflow:
    1. Load datasets
    2. Validate schema
    3. Inspect & repair time-series continuity
    4. Inspect & handle anomalies
    5. Generate master dataset
    6. Create time-series baseline features (Leakage-free)
    """

    def __init__(
        self,
        train_path: str,
        test_path: str,
        outlier_multiplier: int = 20,
        winsorize_quantile: float = 0.99
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.outlier_multiplier = outlier_multiplier
        self.winsorize_quantile = winsorize_quantile

        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.master_df: Optional[pd.DataFrame] = None

    # =========================================================
    # STEP 1 — LOAD DATA
    # =========================================================
    def load_data(self) -> "SalesDataPipeline":
        logging.info("[1/6] Loading datasets...")

        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)

        # Convert Date column to datetime
        self.train_df["Date"] = pd.to_datetime(self.train_df["Date"])
        self.test_df["Date"] = pd.to_datetime(self.test_df["Date"])

        logging.info("Datasets loaded successfully.")
        return self

    # =========================================================
    # STEP 2 — VALIDATE SCHEMA
    # =========================================================
    def validate_schema(self) -> "SalesDataPipeline":
        logging.info("[2/6] Validating dataset schema...")

        required_columns = ["Date", "Revenue", "COGS"]

        for col in required_columns:
            if col not in self.train_df.columns:
                raise ValueError(f"Missing required column in train set: {col}")

        if "Date" not in self.test_df.columns:
            raise ValueError("Missing required column in test set: Date")

        logging.info("Schema validation passed.")
        return self

    # =========================================================
    # STEP 3 — INSPECT & FIX TIME SERIES
    # =========================================================
    def inspect_and_fix_dates(self) -> "SalesDataPipeline":
        logging.info("[3/6] Inspecting time-series continuity...")

        df = self.train_df.copy()
        df = df.sort_values("Date").reset_index(drop=True)

        min_date = df["Date"].min()
        max_date = df["Date"].max()

        expected_days = (max_date - min_date).days + 1
        actual_days = df["Date"].nunique()
        missing_days = expected_days - actual_days

        if missing_days > 0:
            logging.warning(f"Detected {missing_days} missing day(s) in the timeline.")
            logging.info("Creating missing dates and filling financial values with zero.")

            full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")
            df = (
                df.set_index("Date")
                .reindex(full_date_range)
                .reset_index()
                .rename(columns={"index": "Date"})
            )

            df["Revenue"] = df["Revenue"].fillna(0)
            df["COGS"] = df["COGS"].fillna(0)
        else:
            logging.info("Time series is fully continuous.")

        df = df.sort_values("Date").reset_index(drop=True)
        self.train_df = df

        return self

    # =========================================================
    # STEP 4 — INSPECT & FIX ANOMALIES
    # =========================================================
    def inspect_and_fix_anomalies(self) -> "SalesDataPipeline":
        logging.info("[4/6] Inspecting anomalous data...")
        df = self.train_df.copy()

        # 1. Duplicate Detection
        duplicates = df.duplicated(subset=["Date"]).sum()
        if duplicates > 0:
            logging.warning(f"Detected {duplicates} duplicate date entries.")
            df = df.drop_duplicates(subset=["Date"], keep="last")
        
        # 2. Negative Values
        negative_revenue = (df["Revenue"] < 0).sum()
        negative_cogs = (df["COGS"] < 0).sum()
        if negative_revenue > 0 or negative_cogs > 0:
            logging.warning(f"Negative values detected (Revenue={negative_revenue}, COGS={negative_cogs}).")
            df["Revenue"] = df["Revenue"].clip(lower=0)
            df["COGS"] = df["COGS"].clip(lower=0)

        # 3. Outlier Detection (Winsorization)
        revenue_mean = df["Revenue"].mean()
        revenue_max = df["Revenue"].max()

        if revenue_max > revenue_mean * self.outlier_multiplier:
            logging.warning(f"Extreme outlier detected (Max={revenue_max:.2f}, Mean={revenue_mean:.2f}).")
            
            revenue_upper = df["Revenue"].quantile(self.winsorize_quantile)
            cogs_upper = df["COGS"].quantile(self.winsorize_quantile)

            df["Revenue"] = df["Revenue"].clip(upper=revenue_upper)
            df["COGS"] = df["COGS"].clip(upper=cogs_upper)

        self.train_df = df.sort_values("Date").reset_index(drop=True)
        return self

    # =========================================================
    # STEP 5 — PREPARE MASTER DATA
    # =========================================================
    def prepare_master_data(self) -> pd.DataFrame:
        logging.info("[5/6] Building master dataset...")

        self.train_df["is_test"] = 0
        test_dates = self.test_df[["Date"]].copy()
        test_dates["is_test"] = 1

        self.master_df = pd.concat([self.train_df, test_dates], axis=0, ignore_index=True)
        self.master_df = self.master_df.sort_values("Date").reset_index(drop=True)

        return self.master_df

    # =========================================================
    # STEP 6 — FEATURE ENGINEERING (LEAKAGE SAFE)
    # =========================================================
    def create_time_features(self) -> pd.DataFrame:
        logging.info("[6/6] Creating time-series baseline features...")
        df = self.master_df.copy()

        # Calendar Features
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["day_of_month"] = df["Date"].dt.day
        df["month"] = df["Date"].dt.month
        df["quarter"] = df["Date"].dt.quarter
        df["year"] = df["Date"].dt.year
        df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)

        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)

        # Lag & Rolling Features (CRITICAL: MUST SHIFT BEFORE ROLLING/PCT_CHANGE)
        if "Revenue" in df.columns:
            df["revenue_lag_1"] = df["Revenue"].shift(1)
            df["revenue_lag_7"] = df["Revenue"].shift(7)
            df["revenue_lag_30"] = df["Revenue"].shift(30)

            # Fix Leakage: Shift by 1 first, then apply rolling window
            df["rolling_mean_7"] = df["Revenue"].shift(1).rolling(window=7).mean()
            df["rolling_std_7"] = df["Revenue"].shift(1).rolling(window=7).std()

            # Fix Leakage: Shift by 1 first, then apply pct_change (calculates previous day's growth rate)
            df["revenue_growth_rate"] = df["Revenue"].shift(1).pct_change()

        # Convert categorical columns
        categorical_cols = ["day_of_week", "month", "quarter", "year"]
        for col in categorical_cols:
            df[col] = df[col].astype("category")

        self.master_df = df
        logging.info("Feature engineering completed successfully.")
        return self.master_df

# =============================================================
# PIPELINE EXECUTION
# =============================================================
if __name__ == "__main__":
    pipeline = (
        SalesDataPipeline(
            train_path="data/sales.csv", # Cập nhật đường dẫn file của bạn
            test_path="data/sample_submission.csv",
            outlier_multiplier=20,
            winsorize_quantile=0.99
        )
        .load_data()
        .validate_schema()
        .inspect_and_fix_dates()
        .inspect_and_fix_anomalies()
    )

    pipeline.prepare_master_data()
    final_df = pipeline.create_time_features()

    print("\nFinal Dataset Preview:")
    print(final_df.head())