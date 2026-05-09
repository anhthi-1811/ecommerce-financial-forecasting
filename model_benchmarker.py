# ==============================================================================
# KHỐI 3: MODEL BENCHMARKER
# Mục tiêu: So sánh hiệu năng của 3 thuật toán Gradient Boosting.
# ==============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

import optuna
from optuna.samplers import TPESampler

class ModelBenchmarker: 
    """
    Tự động chia tập Train/Validation, huấn luyện và đánh giá trên 3 chỉ số.
    """
    def __init__(self, data, features, target_col='Revenue', val_start_date='2022-07-01'):
        self.data = data.copy()
        self.features = features
        self.target_col = target_col
        self.val_start_date = val_start_date
        
        # 1. DANH SÁCH CÁC CỘT PHÂN LOẠI (Categorical)
        self.cat_features = ['day_of_week', 'month', 'year', 'is_weekend', 'is_promo_active']
        # Lọc lại chỉ giữ những cột có mặt trong SAFE_FEATURES 
        self.cat_features = [c for c in self.cat_features if c in self.features]
        
        # 2. KHỞI TẠO BỘ 3 THUẬT TOÁN 
        # Khai báo cùng mức n_estimators (số lượng cây) và learning_rate (tốc độ học) 
        self.models = {
            'LightGBM': lgb.LGBMRegressor(
                random_state=42, n_estimators=500, learning_rate=0.05, verbose=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                random_state=42, n_estimators=500, learning_rate=0.05, enable_categorical=True
            ),
            'CatBoost': CatBoostRegressor(
                random_state=42, iterations=500, learning_rate=0.05, verbose=False
            )
        }
        self.results = []
        
    def prepare_data(self):
        """Bước 1: Chia tách dữ liệu theo chiều thời gian (Time-based splitting)""" 
        print(f"Đang cắt dữ liệu tại mốc: {self.val_start_date}...")
        
        # Chỉ lấy dữ liệu quá khứ 
        df_past = self.data[self.data['is_test'] == 0].copy()
        
        # Ép kiểu dữ liệu về 'category' để XGBoost và LightGBM có thể hiểu được
        for col in self.cat_features:
            df_past[col] = df_past[col].astype('int').astype('category')
                
        # CHIA TẬP TRAIN VÀ VALIDATION
        # Huấn luyện trên dữ liệu trước mốc thời gian, thi thử trên dữ liệu sau mốc thời gian 
        train_df = df_past[df_past['Date'] < self.val_start_date]
        val_df = df_past[df_past['Date'] >= self.val_start_date]
        
        self.X_train = train_df[self.features]
        self.y_train = train_df[self.target_col]
        
        self.X_val = val_df[self.features]
        self.y_val = val_df[self.target_col]
        
        print(f"   -> Tập Train (Học):     {len(self.X_train)} ngày")
        print(f"   -> Tập Validation (Thi): {len(self.X_val)} ngày")
        return self

    def tune_with_optuna(self, model_name='XGBoost', n_trials=20):
        print(f"\nBẮT ĐẦU TỐI ƯU HÓA BẰNG OPTUNA [{model_name}]...")
        
        def objective(trial):
            # ==========================================
            # ĐOẠN IF SỐ 1: KHAI BÁO THÔNG SỐ (GIỮ NGUYÊN)
            # ==========================================
            if model_name == 'XGBoost':
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'max_depth': trial.suggest_int('max_depth', 4, 8),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'random_state': 42,
                    'enable_categorical': True
                }
                model = xgb.XGBRegressor(**param)

            elif model_name == 'LightGBM':
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100), 
                    'max_depth': trial.suggest_int('max_depth', 4, 10),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMRegressor(**param)

            elif model_name == 'CatBoost':
                param = {
                    'iterations': trial.suggest_int('iterations', 300, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'depth': trial.suggest_int('depth', 4, 8),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'random_state': 42,
                    'verbose': False
                }
                model = CatBoostRegressor(**param)

            # ==========================================
            # VÒNG LẶP HUẤN LUYỆN
            # ==========================================
            tscv = TimeSeriesSplit(n_splits=3)
            rmse_scores = []
            
            for train_idx, val_idx in tscv.split(self.X_train):
                X_tr, y_tr = self.X_train.iloc[train_idx], self.y_train.iloc[train_idx]
                X_va, y_va = self.X_train.iloc[val_idx], self.y_train.iloc[val_idx]
                
                # ==========================================
                # ÉP KIỂU VÀ FIT (ĐÃ THÊM EARLY STOPPING)
                # ==========================================
                if model_name == 'CatBoost':
                    X_tr_cat, X_va_cat = X_tr.copy(), X_va.copy()
                    for col in self.cat_features:
                        X_tr_cat[col] = X_tr_cat[col].astype(str)
                        X_va_cat[col] = X_va_cat[col].astype(str)
                    
                    # Mô hình bắt đầu học có giám sát (Early Stopping)
                    model.fit(
                        X_tr_cat, y_tr, 
                        cat_features=self.cat_features,
                        eval_set=[(X_va_cat, y_va)], 
                        early_stopping_rounds=50,    
                        verbose=False
                    )
                    preds = model.predict(X_va_cat)
                
                else:
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_va)
                
                rmse = np.sqrt(mean_squared_error(y_va, preds))
                rmse_scores.append(rmse)
                
            return np.mean(rmse_scores)

        # Khởi động Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Lưu Quán quân
        best_params = study.best_params
        best_params['random_state'] = 42
        
        if model_name == 'XGBoost':
            best_params['enable_categorical'] = True
            self.models[f"{model_name}_Optuna_Tuned"] = xgb.XGBRegressor(**best_params)
        elif model_name == 'LightGBM':
            best_params['verbose'] = -1
            self.models[f"{model_name}_Optuna_Tuned"] = lgb.LGBMRegressor(**best_params)
        elif model_name == 'CatBoost':
            best_params['verbose'] = False
            self.models[f"{model_name}_Optuna_Tuned"] = CatBoostRegressor(**best_params)
            
        print(f"Đã thêm [{model_name}_Optuna_Tuned]!")
        return self

        # 3. KÍCH HOẠT OPTUNA
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        
        if model_name == 'XGBoost':
            best_params['enable_categorical'] = True
            self.models[f"{model_name}_Optuna_Tuned"] = xgb.XGBRegressor(**best_params)
            
        elif model_name == 'LightGBM':
            best_params['verbose'] = -1
            self.models[f"{model_name}_Optuna_Tuned"] = lgb.LGBMRegressor(**best_params)
            
        elif model_name == 'CatBoost':
            best_params['verbose'] = False
            self.models[f"{model_name}_Optuna_Tuned"] = CatBoostRegressor(**best_params)
            
        print(f"Đã thêm [{model_name}_Optuna_Tuned]!")
        return self
    
    def run_experiments(self):
        """Bước 2: Cho các mô hình chạy trên tập Validation và tính toán các chỉ số MAE, RMSE, R2 Score"""
        print(f"\nBẮT ĐẦU DỰ BÁO: {self.target_col.upper()}...")
        
        for name, model in self.models.items():
            print(f"Đang huấn luyện [{name}]...")
            
            if 'CatBoost' in name:
                X_train_cat = self.X_train.copy()
                X_val_cat = self.X_val.copy()
                
                # Ép kiểu category của Pandas về String để CatBoost 
                for col in self.cat_features:
                    X_train_cat[col] = X_train_cat[col].astype(str)
                    X_val_cat[col] = X_val_cat[col].astype(str)
                
                # BẮT BUỘC phải truyền cat_features
                model.fit(X_train_cat, self.y_train, cat_features=self.cat_features)
                preds = model.predict(X_val_cat)
                
            else:
                # Dành cho LightGBM và XGBoost 
                model.fit(self.X_train, self.y_train)
                preds = model.predict(self.X_val)
            
            # ==========================================
            # GIẢI MÃ LOGARIT (INVERSE TRANSFORM)
            # ==========================================
            if self.target_col == 'Revenue_Log':
                preds_real = np.expm1(preds) 
                actuals_real = np.expm1(self.y_val)
            else:
                preds_real = preds
                actuals_real = self.y_val
                
            preds_real = np.clip(preds_real, a_min=0, a_max=None)
            
            mae = mean_absolute_error(actuals_real, preds_real)
            rmse = np.sqrt(mean_squared_error(actuals_real, preds_real))
            r2 = r2_score(actuals_real, preds_real)
            
            self.results.append({
                'Model': name,
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'R2_Score': round(r2, 4)
            })
            
        return self

    def show_leaderboard(self):
        """Bước 3: In ra Bảng xếp hạng, sắp xếp theo độ chính xác"""
        print("\nBẢNG XẾP HẠNG MÔ HÌNH")
        
        # Sắp xếp theo RMSE thấp nhất (Lỗi càng nhỏ càng xếp trên)
        leaderboard = pd.DataFrame(self.results).sort_values(by='RMSE')
        
        display(leaderboard) 
        return leaderboard