![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

# E-commerce Financial Forecasting System 

An end-to-end Machine Learning pipeline designed to forecast daily Revenue and Cost of Goods Sold (COGS) for an e-commerce platform.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Results](#key-results)
- [License](#license)

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

## Cài đặt

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
   
4. Trực quan hoá dữ liệu.
    - Run các cells trong file notebook `summary.ipynb` để xem trend, phân phối dữ liệu... của tập dữ liệu `sales.csv`.
5. Pipe line chính - Timeseries Forcasting.
    - Run các cells trong file note book `main_pipeline.ipynb`.
    - Pipeline bao gồm:
        + Đọc dữ liệu từ file config.ini
        + Chia tập train/valid theo `split_date`
        + Hiệu chỉnh siêu tham số của các models.
        + Chọn model dựa trên metric `RMSE` cho từng biến target.
        + Sử dụng các best models trên từng biến target để dự đoán.
        + Vẽ các biểu đồ SHAP, Feature Importance, PDP để giải thích mô hình

## Configuration file:
    - Lưu các directory của `dataset`, `output`...
    - Lưu mốc thời gian để chia tập dữ liệu train/val.
    - Lưu các settings của optuna, và target columns.
```bash 
