import lightgbm as lgb
import numpy as np
import pandas as pd
import copy

class FinancialForecaster:
    def __init__(self, data, features, targets=['Revenue_Log', 'COGS_Log']):
        self.data = data.copy()
        self.features = features
        self.targets = targets
        self.models = {}
        self.predictions = {}
        
        # MÔ HÌNH LIGHTGBM VỚI THÔNG SỐ CHỐNG OVERFITTING TỐT NHẤT
        self.base_model = lgb.LGBMRegressor(
            random_state=42, 
            n_estimators=450,     # Giảm số cây xuống một chút để tránh học vẹt
            learning_rate=0.04,   # Tốc độ học cân bằng
            num_leaves=31,
            colsample_bytree=0.8, # Thêm tính năng này: Chỉ dùng 80% cột dữ liệu mỗi lần xây cây -> Chống nhiễu từ các biến mới
            verbose=-1
        )

    def prepare_data(self):
        print("[1/3] Đang chuẩn bị dữ liệu (Dùng phép biến đổi Log1p kinh điển)...")
        self.data['Revenue'] = self.data['Revenue'].fillna(0)
        self.data['COGS'] = self.data['COGS'].fillna(0)
        
        # SỬ DỤNG LẠI LOG1P BẢO TOÀN ĐỈNH DOANH THU
        self.data['Revenue_Log'] = np.log1p(self.data['Revenue'])
        self.data['COGS_Log'] = np.log1p(self.data['COGS'])
        
        self.df_past = self.data[self.data['is_test'] == 0].copy()
        self.df_future = self.data[self.data['is_test'] == 1].copy()
        
        # GIỮ NGUYÊN CATEGORY
        cat_features = [c for c in ['day_of_week', 'month', 'year', 'is_weekend', 'is_promo_active', 'is_payday'] if c in self.features]
        for col in cat_features:
            self.df_past[col] = self.df_past[col].astype('int').astype('category')
            self.df_future[col] = self.df_future[col].astype('int').astype('category')
            
        self.X_train = self.df_past[self.features]
        self.X_test = self.df_future[self.features]
        return self

    def train_and_predict(self):
        print("[2/3] LightGBM đang phân tích và dự báo...")
        for target in self.targets:
            clean_name = target.replace('_Log', '')
            print(f"   -> Đang huấn luyện cho: {clean_name}")
            
            model = copy.deepcopy(self.base_model)
            model.fit(self.X_train, self.df_past[target])
            self.models[target] = model
            
            # Dự báo và giải mã Log1p (Bằng expm1)
            log_preds = model.predict(self.X_test)
            real_preds = np.expm1(log_preds)
                
            self.predictions[clean_name] = np.clip(real_preds, a_min=0, a_max=None)
            
        return self

    def generate_report(self, export_path='submission_hybrid.csv'):
        print("[3/3] Đang xuất Báo cáo nộp bài...")
        submission = pd.DataFrame({
            'Date': pd.to_datetime(self.df_future['Date']).dt.strftime('%Y-%m-%d'),
            'Revenue': self.predictions.get('Revenue', 0),
            'COGS': self.predictions.get('COGS', 0)
        })
        submission.to_csv(export_path, index=False)
        print(f"File nộp bài lưu tại: {export_path}")
        display(submission.head(10))
        return submission