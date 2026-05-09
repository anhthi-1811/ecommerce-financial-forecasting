# ==============================================================================
# KHỐI 2: FEATURE ENGINEERING 
# Mục tiêu: Kéo dữ liệu từ các file vệ tinh (Khuyến mãi, Traffic, Tồn kho)
#           đắp vào bảng Sales chính. Trọng tâm là CHỐNG RÒ RỈ DỮ LIỆU.
# ==============================================================================

import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Class này nhận Dataframe gốc (đã được làm sạch) từ SalesDataPipeline.
    """
    
    def __init__(self, master_df, data_dir='data/'):
        # Dùng .copy() để không làm biến đổi nhầm bản gốc
        self.df = master_df.copy() 
        self.data_dir = data_dir
        
    def add_promotions(self, promo_filename='promotions.csv'):
        """
        BƯỚC 1: XỬ LÝ BẢNG KHUYẾN MÃI
        Vấn đề: File khuyến mãi ghi theo kiểu "Từ ngày A đến ngày B".
        Giải pháp: Kéo giãn (Explode) khoảng thời gian đó ra thành từng ngày lẻ.
        """
        print("[1/4] Đang xử lý bảng Promotions...")
        try:
            # 1. Đọc file
            promos = pd.read_csv(self.data_dir + promo_filename)
            promos['start_date'] = pd.to_datetime(promos['start_date'])
            promos['end_date'] = pd.to_datetime(promos['end_date'])
            
            # 2. TẠO LIST NGÀY: Với mỗi dòng, tạo một danh sách các ngày nằm giữa start và end
            # Ví dụ: start 01/01, end 03/01 -> tạo ra list [01/01, 02/01, 03/01]
            promos['Date'] = promos.apply( 
                lambda row: pd.date_range(row['start_date'], row['end_date']).tolist(), axis=1
            )
            
            # 3. KÉO GIÃN (EXPLODE): Lệnh này biến 1 dòng chứa list 3 ngày thành 3 dòng riêng biệt
            promos_exploded = promos.explode('Date')
            
            # 4. GOM NHÓM (GROUPBY): Đếm xem trong 1 ngày cụ thể có bao nhiêu chương trình chạy song song
            daily_promos = promos_exploded.groupby('Date').agg(
                active_promos_count=('promo_id', 'count') # Đếm số lượng promo_id
            ).reset_index()
            
            # 5. GHÉP NỐI (LEFT JOIN) vào self.df 
            self.df = pd.merge(self.df, daily_promos, on='Date', how='left')
            
            # 6. DỌN DẸP SAU KHI NỐI: Ngày nào không có khuyến mãi sẽ bị NaN, ta điền 0
            self.df['active_promos_count'] = self.df['active_promos_count'].fillna(0)
            
            # Tạo thêm 1 biến Cờ (Flag): 1 là có KM, 0 là không có KM 
            self.df['is_promo_active'] = (self.df['active_promos_count'] > 0).astype(int)
            
        except FileNotFoundError:
            print(f"Bỏ qua: Không tìm thấy file {promo_filename}.")
            
        return self

    def add_web_traffic(self, traffic_filename='web_traffic.csv'):
        """
        BƯỚC 2: XỬ LÝ BẢNG WEB TRAFFIC (KỸ THUẬT CHỐNG LEAKAGE)
        Nguyên tắc: Không được dùng traffic hôm nay để đoán doanh thu hôm nay.
        Giải pháp: Nối vào xong, phải đẩy (shift) dữ liệu lùi xuống 1 ngày.
        """
        print("[2/4] Đang xử lý bảng Web Traffic (Chỉ lấy Lag 1)...")
        try:
            traffic = pd.read_csv(self.data_dir + traffic_filename)
            
            # 1. Làm sạch tên cột
            traffic.columns = traffic.columns.str.strip().str.lower()
            if 'date' in traffic.columns:
                traffic = traffic.rename(columns={'date': 'Date'})
            traffic['Date'] = pd.to_datetime(traffic['Date'])
            
            # Chỉ lọc lấy đúng cột Date và sessions. 
            # ==========================================
            if 'sessions' in traffic.columns:
                traffic = traffic[['Date', 'sessions']]
            else:
                print("LỖI: Không tìm thấy cột 'sessions', vui lòng kiểm tra lại file traffic.")
                return self
            
            # 2. Nối vào bảng chính 
            self.df = pd.merge(self.df, traffic, on='Date', how='left')
            
            # 3. Kéo lùi dữ liệu xuống 1 ngày (Lag 1)
            self.df['sessions_lag_1'] = self.df['sessions'].shift(1)
            
            # 4. Xóa cột sessions của ngày hiện tại
            self.df = self.df.drop(columns=['sessions'])
            
        except FileNotFoundError:
            print(f"Bỏ qua: Không tìm thấy file {traffic_filename}.") 
            
        return self

    def add_inventory(self, inventory_filename='inventory.csv'):
        """
        BƯỚC 3: XỬ LÝ BẢNG TỒN KHO 
        Dùng kỹ thuật Forward Fill để kéo dài số liệu cuối tháng cho cả tháng.
        """
        print("[3/4] Đang xử lý bảng Inventory (Gom nhóm và Forward Fill)...")
        try:
            inv = pd.read_csv(self.data_dir + inventory_filename)
            
            # Xóa khoảng trắng thừa ở tên cột 
            inv.columns = inv.columns.str.strip().str.lower()
            
            inv['snapshot_date'] = pd.to_datetime(inv['snapshot_date'])
            
            # GOM NHÓM VÀ TÍNH TỔNG (Trích xuất đặc trưng) 
            # Không chỉ tính tổng hàng tồn, ta đếm luôn có bao nhiêu mã bị đứt hàng (stockout)
            monthly_inv = inv.groupby('snapshot_date').agg(
                total_stock_on_hand=('stock_on_hand', 'sum'),
                total_stockouts=('stockout_flag', 'sum') 
            ).reset_index()
            
            # Đổi tên cột ngày để JOIN
            monthly_inv = monthly_inv.rename(columns={'snapshot_date': 'Date'})
            
            # Ghép vào bảng chính
            self.df = pd.merge(self.df, monthly_inv, on='Date', how='left')
            
            # FORWARD FILL: Lấp đầy dữ liệu từ cuối tháng trước cho các ngày tháng này
            cols_to_fill = ['total_stock_on_hand', 'total_stockouts']
            for col in cols_to_fill:
                self.df[col] = self.df[col].ffill()
                self.df[col] = self.df[col].fillna(0) # Điền 0 cho những ngày đầu năm 2012
            
        except FileNotFoundError: 
            print(f"Bỏ qua: Không tìm thấy file {inventory_filename}.")
            
        return self

    def add_sales_lags(self):
        """
        BƯỚC 4: TÍNH TOÁN LỊCH SỬ DOANH THU (LAG & ROLLING)
        Đây là các biến mang sức mạnh dự báo lớn nhất cho Time-Series.
        """
        print("[4/4] Đang tạo các biến Trễ Doanh thu (Lags & Rolling)...")
        
        # 1. Các biến trễ Đơn điểm (Hôm qua, Tuần trước, Tháng trước)
        for lag in [1, 7, 30]:
            self.df[f'revenue_lag_{lag}'] = self.df['Revenue'].shift(lag)
            
        # TÍNH ĐÀ TĂNG TRƯỞNG (ROLLING MEAN):
        # 1. shift(1): Lùi về hôm qua (tránh rò rỉ dữ liệu hôm nay)
        # 2. rolling(7): Khoanh vùng 7 ngày liên tiếp từ hôm qua trở về trước
        # 3. mean(): Tính trung bình của 7 ngày đó

        # 2. Các biến trượt Trung bình (Rolling Mean) - Thể hiện đà tăng trưởng
        # Đẩy về hôm qua (shift 1) rồi mới tính trung bình (rolling)
        self.df['revenue_rolling_mean_7'] = self.df['Revenue'].shift(1).rolling(window=7).mean()
        self.df['revenue_rolling_mean_30'] = self.df['Revenue'].shift(1).rolling(window=30).mean()
        
        # 3. Biến Độ lệch chuẩn (Rolling STD) - Thể hiện rủi ro / Sự bất ổn định
        # Nếu std cao nghĩa là doanh thu dạo này trồi sụt thất thường
        self.df['revenue_rolling_std_7'] = self.df['Revenue'].shift(1).rolling(window=7).std()
        
        return self

    def add_calendar_events(self):
        """
        BƯỚC 5: CALENDAR EVENTS
        """
        print("[5/5] Đang đánh dấu các (Siêu Sale, Nhận lương)...")
        
        # 1. NGÀY NHẬN LƯƠNG (Payday)
        paydays = [1, 5, 15, 25] 
        self.df['is_payday'] = self.df['day_of_month'].isin(paydays).astype(int)
        
        # 2. NGÀY SIÊU SALE SỐ ĐÔI (Double Days)
        # Các ngày 1/1, 2/2, 3/3, ..., 11/11, 12/12 (Ngày == Tháng)
        self.df['is_double_day'] = (self.df['day_of_month'] == self.df['month']).astype(int)
        
        # 3. MÙA GIÁNG SINH
        # Giáng sinh (20 - 25/12)
        is_xmas = (self.df['month'] == 12) & (self.df['day_of_month'].between(20, 25))
        self.df['is_christmas'] = is_xmas.astype(int)
        
        return self

    def add_proximity_features(self):
        print("[6] Đang tính toán...")
        def days_to_nearest_payday(day):
            paydays = [1, 5, 15, 25, 31] 
            distances = [p - day for p in paydays if p >= day]
            return min(distances) if distances else 1
        self.df['days_to_payday'] = self.df['day_of_month'].apply(days_to_nearest_payday)
        
        def days_to_double_day(row):
            d, m = row['day_of_month'], row['month']
            if d <= m: return m - d
            return (30 - d) + (m + 1) 
        self.df['days_to_double_day'] = self.df.apply(days_to_double_day, axis=1)
        return self

    def add_momentum_features(self):
        print("[7] Đang tính toán Động lượng và Gia tốc...")
        self.df['revenue_lag_2'] = self.df['Revenue'].shift(2)
        self.df['revenue_lag_3'] = self.df['Revenue'].shift(3)
        self.df[['revenue_lag_2', 'revenue_lag_3']] = self.df[['revenue_lag_2', 'revenue_lag_3']].fillna(0)
        
        self.df['revenue_diff_1'] = self.df['revenue_lag_1'] - self.df['revenue_lag_2']
        self.df['revenue_diff_7'] = self.df['revenue_lag_1'] - self.df['revenue_lag_7']
        self.df['revenue_acceleration'] = self.df['revenue_diff_1'] - (self.df['revenue_lag_2'] - self.df['revenue_lag_3'])
        return self

    def add_feature_interactions(self):
        print("[8] Đang kết hợp các Siêu Tương tác chéo...")
        self.df['weekend_x_payday'] = self.df['is_weekend'] * self.df['is_payday']
        self.df['promo_x_weekend'] = self.df['is_promo_active'] * self.df['is_weekend']
        
        self.df['sessions_per_dayofweek'] = self.df['sessions_lag_1'] / (self.df['day_of_week'].astype(int) + 1)
        
        return self

    def get_data(self):
        """Trả về sản phẩm cuối cùng sau khi đã lắp ráp xong"""
        print("HOÀN TẤT QUÁ TRÌNH FEATURE ENGINEERING!")
        return self.df