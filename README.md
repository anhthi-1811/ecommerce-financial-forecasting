![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

# E-commerce Financial Forecasting System 

An end-to-end Machine Learning pipeline designed to forecast daily Revenue and Cost of Goods Sold (COGS) for an e-commerce platform.

## Table of Contents
- [E-commerce Financial Forecasting System](#e-commerce-financial-forecasting-system)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Tech Stack](#tech-stack)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Configuration file:](#configuration-file)
  - [Evaluation Metrics](#evaluation-metrics)
    - [Arena 1: Revenue Forecasting](#arena-1-revenue-forecasting)
    - [Arena 2: COGS Forecasting](#arena-2-cogs-forecasting)
  - [Business Insights (XAI)](#business-insights-xai)
    - [1. Revenue Drivers (XGBoost Insights):](#1-revenue-drivers-xgboost-insights)
    - [2. COGS Dynamics (LightGBM Insights):](#2-cogs-dynamics-lightgbm-insights)
  - [LICENSE](#license)

## Overview
- **Domain:** Retail & E-commerce
- **Objective:** This project aims to predict future financial metrics (Revenue & COGS) using advanced Machine Learning techniques based on historical sales data.
- **Methodology:** The system heavily emphasizes robust **Feature Engineering** (lags, rolling windows, momentum, cross-interactions) and strict **Data Leakage prevention**. It utilizes Log1p transformations to handle highly skewed sales spikes and evaluates powerful tree-based models like **LightGBM** and **CatBoost**.

## Key Features
- **Data Pipeline:** Cleans and preprocesses raw time-series data seamlessly.
- **Advanced Feature Engineering:** Generates temporal and mathematical features while strictly maintaining forward-shifting to prevent data leakage.
- **Hyperparameter Optimization:** Fully automated tuning process utilizing Optuna.
- **Target Transformation:** Implementation of Log1p to normalize highly skewed target variables.
- **Modular Architecture:** Clean Object-Oriented Programming (OOP) design, making it easy to scale or plug in new algorithms.

## Tech Stack
- **Language:** Python 3.9+
- **Data Manipulation:** NumPy, Pandas
- **Machine Learning:** Scikit-learn, LightGBM, CatBoost
- **Hyperparameter Tuning:** Optuna
- **Visualization:** Matplotlib, Seaborn (for Feature Importance and Actual vs Predicted analysis)

## Project Structure
```bash
ecommerce-financial-forecasting/
├── assets/                   # Images and plots (e.g., Feature Importance)
├── data/                     # Raw dataset folder (Ignored by Git for security)
├── sample_data/              # Snippets of data to test the pipeline safely
├── src/                      # Core modules
│   ├── data_pipeline.py      # Data loading, cleaning, and merging
│   ├── feature_engineer.py   # Time-series feature extraction
│   ├── model_benchmarker.py  # Optuna optimization scripts
│   └── financial_forecaster.py # Model training and evaluation
├── .gitignore                # Files and directories to be ignored by Git
├── LICENSE                   # MIT License
├── main.ipynb                # Main execution notebook
├── README.md                 # Project documentation
└── requirements.txt          # Python library dependencies
```
## Installation 

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/anhthi-1811/ecommerce-financial-forecasting.git](https://github.com/anhthi-1811/ecommerce-financial-forecasting.git)
   cd ecommerce-financial-forecasting
    ```    
2. **Create and activate a virtual environment (Recommended):** 
   - cmd-Windows / macOS terminal
    ```bash
    python -m venv venv 
    ```
    - Windows powershell
    ```bash
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    python -m venv venv
    ```
    - Activate virtual environment  
        - Windows - sử dụng cmd
        ```bash
        venv\Scripts\activate 
        ```
        - Linux/Mac
        ```bash
        source venv/bin/activate
        ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
4. Execute the Pipeline (Time-Series Forecasting)
  Open the `main.ipynb` notebook and execute all cells sequentially (`Run All`). The notebook orchestrates the following end-to-end workflow:

  * **Configuration Parsing:** Dynamically loads directory paths, tuning trials, and validation configurations from `config.ini`.
  * **Chronological Splitting:** Divides the dataset into Training and Validation sets strictly based on the `split_date` to prevent time-series data leakage.
  * **Hyperparameter Tuning:** Executes automated Optuna trials to optimize Tree-based architectures.
  * **Champion Selection:** Evaluates and selects the ultimate best-performing model for each specific target (Revenue and COGS) based on the **RMSE** metric.
  * **Inference & Export:** Utilizes the champion models to forecast future values, generating the final `submission.csv`.
  * **Explainable AI (XAI):** Generates and saves SHAP Summary and Feature Importance plots to visually interpret the model's decision-making process.
  
## Configuration file: 
- The entire project is controlled via the `config.ini` file, enabling seamless deployment across different environments without modifying the source code. 
```bash  
[PATHS]
DATA_DIR = dataset
ASSETS_DIR = assets
TRAIN_FILE = dataset/sales.csv
TEST_FILE = dataset/sample_submission.csv
OUT_FILE = submission.csv

[VALID]
split_date = 2022-10-01

[SETTINGS]
N_TRIALS = 15
N_JOBS = -1
TARGET_COLS = Revenue, COGS 
```

## Evaluation Metrics
- The following insights were derived during the Optuna Hyperparameter Tuning phase, utilizing Tree-based ensemble architectures with 50-round Early Stopping.
  ### Arena 1: Revenue Forecasting
  - Champion Model: `XGBoost` 

  - **Analysis**: Revenue data exhibits high volatility driven by promotional spikes and seasonal surges. XGBoost's advanced regularization mechanisms effectively handled these sudden variance spikes (outliers), resulting in the lowest RMSE among all benchmarking models.
  ### Arena 2: COGS Forecasting
  - Champion Model: `LightGBM` 
  - **Analysis**: LightGBM's leaf-wise tree growth strategy demonstrated exceptional sensitivity to inventory depletion dynamics and cost fluctuations. The model successfully captured subtle operational micro-patterns, outperforming both XGBoost and CatBoost for COGS forecasting.

## Business Insights (XAI)
- By applying SHAP (SHapley Additive exPlanations), the models uncovered several critical strategic business drivers: 
  ### 1. Revenue Drivers (XGBoost Insights): 
  - **Autoregressive Dominance**:The `revenue_lag_1` feature contributes more than 60% of the global feature importance. This indicates strong temporal inertia in customer spending behavior; a successful sales day creates momentum that propagates into the following day. 
  
  - **Promotional Elasticity**: Consumer purchasing behavior is highly elastic during `is_double_day` and `is_payday`. SHAP values indicate a massive conversion uplift on these specific days. Recommendation: Concentrate Performance Marketing budgets heavily around these temporal windows.  
  
  ### 2. COGS Dynamics (LightGBM Insights):
  - Traffic as a Leading Indicator: Variables such as `sessions_per_dayofweek` and `sessions_lag_1` dominate the top tier. Web traffic strictly correlates with inventory depletion (COGS), acting as a precursor even before final checkout. 
  - Supply Chain Stabilization: The model prioritizes `revenue_rolling_mean_7` as a baseline to forecast costs, preventing the system from overreacting to short-term anomalies. Recommendation: Utilize this rolling metric as an Early Warning Signal (EWS) for warehouse and inventory allocation.  

## LICENSE
[MIT License](LICENSE) 