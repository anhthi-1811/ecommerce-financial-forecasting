# ==============================================================================
# FEATURE ENGINEERING PIPELINE
# Purpose:
# Enrich the main sales dataset using external supporting datasets:
# - Promotions
# - Web traffic
# - Inventory
#
# Main focus:
# Prevent data leakage in time-series forecasting.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class FeatureEngineer:
    """
    Advanced Feature Engineering Pipeline for Time-Series Forecasting.

    This class receives a cleaned master dataframe from SalesDataPipeline
    and enriches it with external business signals and forecasting features.
    """

    def __init__(
        self,
        master_df: pd.DataFrame,
        data_dir: str = "data/"
    ):
        """
        Initialize feature engineering pipeline.

        Parameters
        ----------
        master_df : pd.DataFrame
            Cleaned master dataframe.

        data_dir : str
            Directory containing external datasets.
        """

        self.df = master_df.copy()

        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"])
            self.df = self.df.sort_values(by="Date").reset_index(drop=True)
            
        self.data_dir = data_dir 

    # ==========================================================================
    # STEP 1 — PROMOTION FEATURES
    # ==========================================================================

    def add_promotions(
        self,
        promo_filename: str = "promotions.csv"
    ) -> "FeatureEngineer":
        """
        Process promotion events.

        Problem:
        Promotion data is stored as date ranges.

        Solution:
        Expand date ranges into individual daily records.
        """

        logging.info("[1/8] Processing promotions dataset...")

        try:

            # --------------------------------------------------------------
            # Load promotion dataset
            # --------------------------------------------------------------

            promos = pd.read_csv(
                self.data_dir + promo_filename
            )

            promos["start_date"] = pd.to_datetime(
                promos["start_date"] 
            )

            promos["end_date"] = pd.to_datetime(
                promos["end_date"]
            )

            # --------------------------------------------------------------
            # Expand date ranges into daily rows
            # --------------------------------------------------------------

            promos["Date"] = promos.apply(
                lambda row: pd.date_range(
                    row["start_date"],
                    row["end_date"]
                ).tolist(),
                axis=1
            )

            promos_exploded = promos.explode("Date")

            # --------------------------------------------------------------
            # Aggregate daily promotion counts
            # --------------------------------------------------------------

            daily_promos = (
                promos_exploded
                .groupby("Date")
                .agg(
                    active_promos_count=(
                        "promo_id",
                        "count"
                    )
                )
                .reset_index()
            )

            # --------------------------------------------------------------
            # Merge with master dataframe
            # --------------------------------------------------------------

            self.df = pd.merge(
                self.df,
                daily_promos,
                on="Date",
                how="left"
            )

            # Fill missing promotion counts
            self.df["active_promos_count"] = (
                self.df["active_promos_count"]
                .fillna(0)
            )

            # Binary promotion flag
            self.df["is_promo_active"] = (
                self.df["active_promos_count"] > 0
            ).astype(int)

            logging.info(
                "Promotion features added successfully."
            )

        except FileNotFoundError:

            logging.warning(
                f"Promotion file not found: {promo_filename}"
            )

        return self

    # ==========================================================================
    # STEP 2 — WEB TRAFFIC FEATURES
    # ==========================================================================

    def add_web_traffic(
        self,
        traffic_filename: str = "web_traffic.csv"
    ) -> "FeatureEngineer":
        """
        Process website traffic data.

        IMPORTANT:
        Prevent data leakage by using lagged traffic only.

        We never use today's traffic to predict today's revenue.
        """

        logging.info("[2/8] Processing web traffic dataset...")

        try:

            traffic = pd.read_csv(
                self.data_dir + traffic_filename
            )

            # --------------------------------------------------------------
            # Standardize column names
            # --------------------------------------------------------------

            traffic.columns = (
                traffic.columns
                .str.strip()
                .str.lower()
            )

            if "date" in traffic.columns:

                traffic = traffic.rename(
                    columns={"date": "Date"}
                )

            traffic["Date"] = pd.to_datetime(
                traffic["Date"]
            )

            # --------------------------------------------------------------
            # Validate required column
            # --------------------------------------------------------------

            if "sessions" not in traffic.columns:

                raise ValueError(
                    "Missing required column: sessions"
                )

            traffic = traffic[["Date", "sessions"]]

            # --------------------------------------------------------------
            # Merge traffic into main dataframe
            # --------------------------------------------------------------

            self.df = pd.merge(
                self.df,
                traffic,
                on="Date",
                how="left"
            )

            # --------------------------------------------------------------
            # Create lagged traffic features
            # --------------------------------------------------------------

            self.df["sessions_lag_1"] = (
                self.df["sessions"]
                .shift(1)
            )

            self.df["sessions_lag_7"] = (
                self.df["sessions"]
                .shift(7)
            )

            # Rolling traffic trend
            self.df["sessions_rolling_mean_7"] = (
                self.df["sessions"]
                .shift(1)
                .rolling(7)
                .mean()
            )

            # Remove current-day sessions
            self.df = self.df.drop(
                columns=["sessions"]
            )

            logging.info(
                "Traffic features added successfully."
            )

        except FileNotFoundError:

            logging.warning(
                f"Traffic file not found: {traffic_filename}"
            )

        return self

    # ==========================================================================
    # STEP 3 — INVENTORY FEATURES
    # ==========================================================================

    def add_inventory(
        self,
        inventory_filename: str = "inventory.csv"
    ) -> "FeatureEngineer":
        """
        Process inventory snapshots.

        Uses forward-fill strategy to propagate
        monthly inventory values across daily records.
        """

        logging.info("[3/8] Processing inventory dataset...")

        try:

            inv = pd.read_csv(
                self.data_dir + inventory_filename
            )

            # --------------------------------------------------------------
            # Clean column names
            # --------------------------------------------------------------

            inv.columns = (
                inv.columns
                .str.strip()
                .str.lower()
            )

            inv["snapshot_date"] = pd.to_datetime(
                inv["snapshot_date"]
            )

            # --------------------------------------------------------------
            # Aggregate inventory metrics
            # --------------------------------------------------------------

            monthly_inv = (
                inv.groupby("snapshot_date")
                .agg(
                    total_stock_on_hand=(
                        "stock_on_hand",
                        "sum"
                    ),

                    total_stockouts=(
                        "stockout_flag",
                        "sum"
                    )
                )
                .reset_index()
            )

            monthly_inv = monthly_inv.rename(
                columns={"snapshot_date": "Date"}
            )

            # --------------------------------------------------------------
            # Merge inventory metrics
            # --------------------------------------------------------------

            self.df = pd.merge(
                self.df,
                monthly_inv,
                on="Date",
                how="left"
            )

            # --------------------------------------------------------------
            # Forward fill inventory values
            # --------------------------------------------------------------

            cols_to_fill = [
                "total_stock_on_hand",
                "total_stockouts"
            ]

            for col in cols_to_fill:

                self.df[col] = (
                    self.df[col]
                    .ffill()
                    .fillna(0)
                )

            logging.info(
                "Inventory features added successfully."
            )

        except FileNotFoundError:

            logging.warning(
                f"Inventory file not found: {inventory_filename}"
            )

        return self

    # ==========================================================================
    # STEP 4 — SALES LAG FEATURES
    # ==========================================================================

    def add_sales_lags(self) -> "FeatureEngineer":
        """
        Create historical revenue features.

        These are among the most powerful features
        for time-series forecasting models.
        """

        logging.info("[4/8] Creating lag and rolling features...")

        # --------------------------------------------------------------
        # Point lag features
        # --------------------------------------------------------------

        for lag in [1, 7, 14, 30]:

            self.df[f"revenue_lag_{lag}"] = (
                self.df["Revenue"]
                .shift(lag)
            )

        # --------------------------------------------------------------
        # Rolling mean features
        # --------------------------------------------------------------

        self.df["revenue_rolling_mean_7"] = (
            self.df["Revenue"]
            .shift(1)
            .rolling(window=7)
            .mean()
        )

        self.df["revenue_rolling_mean_30"] = (
            self.df["Revenue"]
            .shift(1)
            .rolling(window=30)
            .mean()
        )

        # --------------------------------------------------------------
        # Rolling standard deviation
        # --------------------------------------------------------------

        self.df["revenue_rolling_std_7"] = (
            self.df["Revenue"]
            .shift(1)
            .rolling(window=7)
            .std()
        )

        logging.info(
            "Lag and rolling features created successfully."
        )

        return self

    # ==========================================================================
    # STEP 5 — CALENDAR EVENT FEATURES
    # ==========================================================================

    def add_calendar_events(self) -> "FeatureEngineer":
        """
        Add calendar-based business event features.
        """

        logging.info("[5/8] Creating calendar event features...")

        # Payday indicators
        paydays = [1, 5, 15, 25]

        self.df["is_payday"] = (
            self.df["day_of_month"]
            .isin(paydays)
            .astype(int)
        )

        # Double-day sales events (1/1, 2/2, ...)
        self.df["is_double_day"] = (
            self.df["day_of_month"]
            == self.df["month"]
        ).astype(int)

        # Christmas season
        is_christmas = (
            (self.df["month"] == 12)
            &
            (
                self.df["day_of_month"]
                .between(20, 25)
            )
        )

        self.df["is_christmas"] = (
            is_christmas.astype(int)
        )

        logging.info(
            "Calendar event features created successfully."
        )

        return self

    # ==========================================================================
    # STEP 6 — PROXIMITY FEATURES
    # ==========================================================================

    def add_proximity_features(self) -> "FeatureEngineer":
        """
        Create distance-to-event features.
        """

        logging.info("[6/8] Creating proximity features...")

        # --------------------------------------------------------------
        # Distance to nearest payday
        # --------------------------------------------------------------

        def days_to_nearest_payday(day):

            paydays = [1, 5, 15, 25, 31]

            distances = [
                p - day
                for p in paydays
                if p >= day
            ]

            return min(distances) if distances else 1

        self.df["days_to_payday"] = (
            self.df["day_of_month"]
            .apply(days_to_nearest_payday)
        )

        # --------------------------------------------------------------
        # Distance to next double-day event
        # --------------------------------------------------------------

        def days_to_double_day(row):

            d = row["day_of_month"]
            m = row["month"]

            if d <= m:
                return m - d

            return (30 - d) + (m + 1)

        self.df["days_to_double_day"] = (
            self.df.apply(
                days_to_double_day,
                axis=1
            )
        )

        logging.info(
            "Proximity features created successfully."
        )

        return self

    # ==========================================================================
    # STEP 7 — MOMENTUM FEATURES
    # ==========================================================================

    def add_momentum_features(self) -> "FeatureEngineer":
        """
        Create momentum and acceleration features.
        """

        logging.info("[7/8] Creating momentum features...")

        self.df["revenue_lag_2"] = (
            self.df["Revenue"]
            .shift(2)
        )

        self.df["revenue_lag_3"] = (
            self.df["Revenue"]
            .shift(3)
        )

        lag_cols = [
            "revenue_lag_2",
            "revenue_lag_3"
        ]

        self.df[lag_cols] = (
            self.df[lag_cols]
            .fillna(0)
        )

        # Momentum
        self.df["revenue_diff_1"] = (
            self.df["revenue_lag_1"]
            -
            self.df["revenue_lag_2"]
        )

        self.df["revenue_diff_7"] = (
            self.df["revenue_lag_1"]
            -
            self.df["revenue_lag_7"]
        )

        # Acceleration
        self.df["revenue_acceleration"] = (
            self.df["revenue_diff_1"]
            -
            (
                self.df["revenue_lag_2"]
                -
                self.df["revenue_lag_3"]
            )
        )

        logging.info(
            "Momentum features created successfully."
        )

        return self

    # ==========================================================================
    # STEP 8 — FEATURE INTERACTIONS
    # ==========================================================================

    def add_feature_interactions(self) -> "FeatureEngineer":
        """
        Create interaction features between important signals.
        """

        logging.info("[8/8] Creating interaction features...")

        # Weekend + payday interaction
        self.df["weekend_x_payday"] = (
            self.df["is_weekend"]
            *
            self.df["is_payday"]
        )

        # Promotion + weekend interaction
        self.df["promo_x_weekend"] = (
            self.df["is_promo_active"]
            *
            self.df["is_weekend"]
        )

        # Traffic normalized by weekday
        self.df["sessions_per_dayofweek"] = (
            self.df["sessions_lag_1"]
            /
            (
                self.df["day_of_week"]
                .astype(int)
                + 1
            )
        )

        logging.info(
            "Interaction features created successfully."
        )

        return self

    # ==========================================================================
    # FINAL OUTPUT
    # ==========================================================================

    def get_data(self) -> pd.DataFrame:
        """
        Return final engineered dataframe.
        """

        logging.info(
            "Feature engineering pipeline completed successfully."
        )

        return self.df