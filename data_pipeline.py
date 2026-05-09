import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class SalesDataPipeline:
    """
    Module 1: Pipeline xử lý dữ liệu gốc (sales.csv)
    Bao gồm các bước Kiểm tra (Inspect) -> Ra quyết định -> Xử lý (Fix)
    """
    
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.master_df = None
        
    def load_data(self):
        """Bước 1: Nạp dữ liệu vào bộ nhớ"""
        print("[1/4] Đang nạp dữ liệu...")
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        
        self.train_df['Date'] = pd.to_datetime(self.train_df['Date'])
        self.test_df['Date'] = pd.to_datetime(self.test_df['Date'])
        return self

    def inspect_and_fix_dates(self):
        """Bước 2: KIỂM TRA VÀ XỬ LÝ CHUỖI THỜI GIAN"""
        print("\n[2/4] Đang chẩn đoán Trục thời gian...")
        df = self.train_df
        
        # 1. KKiểm tra: Có bị thiếu ngày nào không? 
        min_date, max_date = df['Date'].min(), df['Date'].max()
        expected_days = (max_date - min_date).days + 1
        actual_days = df['Date'].nunique()
        missing_days = expected_days - actual_days
        
        print(f"   -> Giai đoạn: {min_date.date()} đến {max_date.date()}")
        print(f"   -> Số ngày thực tế / Kỳ vọng: {actual_days} / {expected_days}")
        
        # 2. Ra quyết định & Xử lý
        if missing_days > 0:
            print(f"   PHÁT HIỆN LỖI: Thiếu {missing_days} ngày trong lịch sử!")
            print("   Quyết định: Tạo các ngày bị thiếu và điền Doanh thu/Giá vốn = 0.") 
            
            full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
            df = df.set_index('Date').reindex(full_date_range).reset_index()
            df = df.rename(columns={'index': 'Date'})
            df['Revenue'] = df['Revenue'].fillna(0)
            df['COGS'] = df['COGS'].fillna(0)
        else:
            print("Chuỗi thời gian hoàn toàn liền mạch.")
            
        self.train_df = df
        return self

    def inspect_and_fix_anomalies(self):
        """Bước 3: KIỂM TRA VÀ XỬ LÝ DỮ LIỆU DỊ THƯỜNG (TRÙNG/ÂM/NGOẠI LAI)"""
        print("\n[3/4] Đang chẩn đoán Dữ liệu dị thường (Anomalies)...")
        df = self.train_df
        
        # 1. Kiểm tra Trùng lặp
        duplicates = df.duplicated(subset=['Date']).sum()
        if duplicates > 0:
            print(f"PHÁT HIỆN LỖI: Có {duplicates} ngày bị nhập trùng lặp (Double entry)!")
            print("Quyết định: Xóa dòng trùng, chỉ giữ lại 1 dòng duy nhất.")
            df = df.drop_duplicates(subset=['Date'], keep='last')
        else:
            print("Không có dữ liệu trùng lặp.")
            
        # 2. Kiểm tra Giá trị Âm (Doanh thu/Giá vốn không thể âm)
        negative_rev = (df['Revenue'] < 0).sum()
        negative_cogs = (df['COGS'] < 0).sum()
        if negative_rev > 0 or negative_cogs > 0:
            print(f"PHÁT HIỆN LỖI: Có số liệu ÂM ({negative_rev} dòng Revenue, {negative_cogs} dòng COGS)!")
            print("Quyết định: Ép các số âm này về 0 (Clipping lower bounds).")
            df['Revenue'] = df['Revenue'].clip(lower=0)
            df['COGS'] = df['COGS'].clip(lower=0)
        else:
            print("Không có giá trị tài chính âm.")

        # 3. Kiểm tra Ngoại lai (Outliers)
        # Kiểm tra xem giá trị Max có lớn bất thường so với mức trung bình (mean) hay không
        rev_mean = df['Revenue'].mean()
        rev_max = df['Revenue'].max()
        if rev_max > rev_mean * 20: # Nếu có ngày bán gấp 20 lần trung bình
            print(f"CẢNH BÁO: Phát hiện ngày có doanh thu khổng lồ (Max = {rev_max:.2f}, Mean = {rev_mean:.2f}).")
            print("Quyết định: Cắt ngọn (Winsorization) ở mốc 99th percentile để tránh mô hình bị nhiễu.")
            upper_limit = df['Revenue'].quantile(0.99)
            df['Revenue'] = df['Revenue'].clip(upper=upper_limit)
            df['COGS'] = df['COGS'].clip(upper=df['COGS'].quantile(0.99))
            print(f"      Đã giới hạn trần Doanh thu ở mức: {upper_limit:.2f}")
        else:
            print("Không phát hiện ngoại lai gây nhiễu nghiêm trọng.")
            
        self.train_df = df
        return self

    def prepare_master_data(self):
        """Bước 4: GHÉP NỐI VÀ TẠO ĐẶC TRƯNG NỀN TẢNG"""
        print("\n[4/4] Đang tổng hợp Master Data và trích xuất đặc trưng...")
        
        # 1. Gắn cờ và gộp Train + Test
        self.train_df['is_test'] = 0
        test_dates = self.test_df[['Date']].copy()
        test_dates['is_test'] = 1
        
        self.master_df = pd.concat([self.train_df, test_dates], axis=0, ignore_index=True)
        
        # 2. Bóc tách thời gian
        df = self.master_df
        df['day_of_week'] = df['Date'].dt.dayofweek 
        df['day_of_month'] = df['Date'].dt.day
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Ép kiểu Category cho LightGBM
        for col in ['day_of_week', 'month', 'year']:
            df[col] = df[col].astype('category')
            
        print("HOÀN TẤT GIAI ĐOẠN 1: Dữ liệu đã sẵn sàng để Feature Engineering!")
        return self.master_df
