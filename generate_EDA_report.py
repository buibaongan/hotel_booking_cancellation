import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging
import datetime

# --- CẤU HÌNH ---
DATA_PATH = "data/raw/hotel_bookings.csv"
REPORT_DIR = "reports"
IMG_DIR = os.path.join(REPORT_DIR, "images")
LOG_FILE = os.path.join(REPORT_DIR, "eda_activity.log")

# Tạo thư mục nếu chưa có
os.makedirs(IMG_DIR, exist_ok=True)

# --- CẤU HÌNH LOGGING (Bỏ icon, format đơn giản) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class EDAHotelBooking:
    def __init__(self, df):
        logging.info("Bat dau khoi tao lop EDAHotelBooking...")
        self.df = df.copy()
        self.img_count = 0
        
        try:
            # 1. Feature Engineering nhẹ để vẽ biểu đồ
            self.df['total_nights'] = self.df['stays_in_weekend_nights'] + self.df['stays_in_week_nights']
            self.df['total_guests'] = self.df['adults'] + self.df['children'].fillna(0) + self.df['babies']
            self.df['has_requests'] = self.df['total_of_special_requests'] > 0
            
            # 2. Danh sách cột phân loại cần phân tích
            self.target_cat_cols = [
                'hotel', 'is_canceled', 'arrival_date_year', 'arrival_date_month', 'meal', 
                'country', 'market_segment', 'distribution_channel', 'is_repeated_guest', 
                'reserved_room_type', 'assigned_room_type', 'deposit_type', 'agent', 
                'company', 'customer_type', 'reservation_status', 
                'name', 'email', 'phone-number', 'credit_card'
            ]
            
            # Ép kiểu dữ liệu phân loại
            for col in self.target_cat_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].apply(lambda x: str(x) if pd.notnull(x) else np.nan)
            
            logging.info(f" Đã xử lý sơ bộ dữ liệu. Kích thước: {self.df.shape}")
           
        except Exception as e:
            logging.error(f"Lỗi trong quá trình khởi tạo: {str(e)}")
            raise e

    def save_plot(self, title_vn, filename_slug):
        """Lưu biểu đồ vào folder images"""
        self.img_count += 1
        filename = f"{self.img_count:02d}_{filename_slug}.png"
        filepath = os.path.join(IMG_DIR, filename)
        
        try:
            plt.savefig(filepath, bbox_inches='tight', dpi=100)
            plt.close() 
            logging.info(f"Đã lưu biểu đồ {self.img_count}: {title_vn}")
            return filename, title_vn
        except Exception as e:
            logging.error(f"Lỗi khi lưu biểu đồ {filename}: {str(e)}")
            return None, title_vn

    def generate_info_missing_table(self):
        """Tạo bảng Info """
        
        # --- CẤU HÌNH MÀU SẮC ---
        main_color = "#3498db"    # Xanh dương đậm (Viền)
        header_bg = "#e3f2fd"     # Xanh dương nhạt (Nền Header & Cột 1 lúc bình thường)
        warn_bg = "#ffe6e6"       # Đỏ nhạt (Cảnh báo lỗi)
        text_black = "#333333"    # Màu chữ đen
        
        html = f"""
        <h3 style="color: {main_color};">Thông tin chi tiết các cột</h3>
        <div style="width: 100%;">
        <table style="width: 100%; border-collapse: collapse; border: 2px solid {main_color}; font-family: sans-serif;">
            <thead>
                <tr style="background-color: {header_bg}; color: {main_color}; font-weight: bold;">
                    <th style="padding: 10px; border: 2px solid {main_color};">Column</th>
                    <th style="padding: 10px; border: 1px solid {main_color};">Dtype</th>
                    <th style="padding: 10px; border: 1px solid {main_color};">Missing Count</th>
                    <th style="padding: 10px; border: 1px solid {main_color};">NUnique</th>
                </tr>
            </thead>
            <tbody>
        """
        for col in self.df.columns:
            missing = self.df[col].isnull().sum()
            dtype = str(self.df[col].dtype)
            nunique = self.df[col].nunique()
            
            if missing > 0:
                row_bg = warn_bg
                col1_bg = warn_bg 
                text_style = "color: #c0392b; font-weight: bold;"
            else:
                row_bg = "" 
                col1_bg = header_bg 
                text_style = ""
            
            html += f"""
            <tr style="border-bottom: 1px solid #ddd; background-color: {row_bg};">
                <td style="padding: 8px; border: 2px solid {main_color}; background-color: {col1_bg}; font-weight: bold; color: {text_black};">{col}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{dtype}</td>
                <td style="padding: 8px; border: 1px solid #ddd; {text_style}">{missing:,}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{nunique:,}</td>
            </tr>
            """
        html += "</tbody></table></div>"
        return html

    # --- PHẦN 1: TỔNG QUAN ---
    def generate_overview_section(self):
        n_rows, n_cols = self.df.shape
        obj_cols = len(self.df.select_dtypes(include=['object']).columns)
        int_cols = len(self.df.select_dtypes(include=['int64', 'int32']).columns)
        float_cols = len(self.df.select_dtypes(include=['float64']).columns)
        
        missing_series = self.df.isnull().sum()
        missing_cols = missing_series[missing_series > 0].sort_values(ascending=False)
        
        missing_list_html = ""
        if missing_cols.empty:
            missing_list_html = "<li><b>Sạch sẽ!</b> Bộ dữ liệu này không có bất kỳ giá trị thiếu nào.</li>"
        else:
            for col_name, count in missing_cols.items():
                pct = (count / n_rows) * 100
                comment_note = ""
                if pct > 50: comment_note = " <span style='color: #c0392b;'>(Thiếu quá nhiều)</span>"
                missing_list_html += f"<li>Cột <b>{col_name}</b>: {count:,} giá trị thiếu ({pct:.2f}%).{comment_note}</li>"

        # CẤU HÌNH MÀU SẮC
        main_color = "#3498db"
        bg_color = "#e3f2fd"

        html = f"""
        <h2 style="color: {main_color}; border-left: 5px solid {main_color};">1. TỔNG QUAN DỮ LIỆU</h2>
        <div class="comment-box" style="background-color: #e8f4f8; border: 2px solid #3498db; margin-top: 15px;">
            <ul>
                <li><b>Kích thước:</b> {n_rows:,} dòng, {n_cols} cột.</li>
                <ul>
                    <li><b>Số lượng bản ghi:</b> Bộ dữ liệu bao gồm {n_rows:,} bản ghi.</li>
                    <li><b>Số lượng cột:</b> Bộ dữ liệu có {n_cols} cột, tương ứng với nhiều thuộc tính khác nhau liên quan đến đặt phòng khách sạn.</li>
                </ul>
            </ul>
        </div>

        {self.generate_info_missing_table()} 
        
        <div class="comment-box" style="background-color: #e8f4f8; border: 2px solid #3498db; margin-top: 15px;">
            <h3> NHẬN XÉT:</h3>
            <ul>
                <li><b>Cấu trúc dữ liệu:</b>
                    <ul>
                        <li>Phần lớn các cột ({obj_cols} cột) thuộc kiểu object (thường là chuỗi hoặc dữ liệu phân loại).</li>
                        <li>{int_cols} cột thuộc kiểu int64, biểu diễn các giá trị số nguyên.</li>
                        <li>{float_cols} cột thuộc kiểu float64, thường biểu diễn các giá trị thập phân.</li>
                    </ul>
                </li>
                <li><b>Phân tích Giá trị thiếu (Missing Values):</b>
                    <ul>
                        {missing_list_html}
                    </ul>
                    <p><i>Các giá trị thiếu này cần được xử lý tùy theo mục tiêu phân tích hoặc mô hình cần xây dựng..</i></p>
                </li>
            </ul>
            <p><b>Phân loại biến:</b> Dựa trên kiểu dữ liệu, xác định được có {len(self.target_cat_cols)} biến phân loại mục tiêu. Những đặc trưng này cần có kiểu dữ liệu dạng chuỗi (object) để đảm bảo quá trình phân tích và diễn giải trong các bước tiếp theo được thực hiện chính xác.</p>
        </div>
        """
        return html

    # --- PHẦN 2: THỐNG KÊ BIẾN SỐ ----
    def generate_numeric_stats(self):
        num_cols = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
                    'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
                    'babies', 'previous_cancellations', 'previous_bookings_not_canceled',
                    'booking_changes', 'days_in_waiting_list', 'adr', 
                    'required_car_parking_spaces', 'total_of_special_requests']
        valid_cols = [c for c in num_cols if c in self.df.columns]
        desc = self.df[valid_cols].describe().T
        
        main_color = "#3498db"
        header_bg = "#e3f2fd"
        
        table_html = f"""
        <div style="width: 100%;">
        <table class="table numeric-table" style="background-color: transparent; border: 2px solid {main_color}; width: 100%;">
            <thead>
                <tr style="background-color: {header_bg}; color: {main_color}; font-weight: bold;">
                    <th style="border: 1px solid {main_color}; padding: 10px;"></th>
                    <th style="border: 1px solid {main_color}; padding: 10px;">count</th>
                    <th style="border: 1px solid {main_color}; padding: 10px;">mean</th>
                    <th style="border: 1px solid {main_color}; padding: 10px;">std</th>
                    <th style="border: 1px solid {main_color}; padding: 10px;">min</th>
                    <th style="border: 1px solid {main_color}; padding: 10px;">25%</th>
                    <th style="border: 1px solid {main_color}; padding: 10px;">50%</th>
                    <th style="border: 1px solid {main_color}; padding: 10px;">75%</th>
                    <th style="border: 1px solid {main_color}; padding: 10px;">max</th>
                </tr>
            </thead>
            <tbody>
        """
        for idx, row in desc.iterrows():
            table_html += f"""
                <tr style="border: 1px solid #ddd;">
                    <td style="font-weight: bold; background-color: {header_bg}; border: 1px solid {main_color}; padding: 8px;">{idx}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{row['count']:,.2f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{row['mean']:.2f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{row['std']:.2f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{row['min']:.2f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{row['25%']:.2f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{row['50%']:.2f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{row['75%']:.2f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{row['max']:.2f}</td>
                </tr>
            """
        table_html += "</tbody></table></div>"
        
        # Thống kê chi tiết cho từng biến
        main_color = "#3498db"
        numeric_insights = f"""
        <div class="comment-box" style="background-color: #e8f4f8; border: 2px solid {main_color}; margin-top: 20px;">
            <h3 style="color: #3498db;">📈 NHẬN XÉT THỐNG KÊ MÔ TẢ BIẾN SỐ</h3>
            <ul>
                <li><b>lead_time:</b> Trung bình 104 ngày, dao động 0–737.</li>
                <li><b>arrival_date_week_number:</b> Trung bình tuần 27.17, từ 1–53.</li>
                <li><b>arrival_date_day_of_month:</b> Trung bình ngày 15.80, từ 1–31.</li>
                <li><b>stays_in_weekend_nights:</b> Trung bình 0.93 đêm cuối tuần, tối đa 19.</li>
                <li><b>stays_in_week_nights:</b> Trung bình 2.5 đêm trong tuần, từ 0–50.</li>
                <li><b>adults:</b> Trung bình 1.86 người lớn, từ 0–55.</li>
                <li><b>children:</b> Trung bình 0.1 trẻ em, tối đa 10.</li>
                <li><b>babies:</b> Trung bình 0.008 trẻ sơ sinh, tối đa 10.</li>
                <li><b>previous_cancellations:</b> Trung bình 0.09 lần hủy, tối đa 26.</li>
                <li><b>previous_bookings_not_canceled:</b> Trung bình 0.14, tối đa 72.</li>
                <li><b>booking_changes:</b> Trung bình 0.22 lần thay đổi, tối đa 21.</li>
                <li><b>days_in_waiting_list:</b> Trung bình 2.32 ngày, tối đa 391.</li>
                <li><b>adr:</b> Trung bình 101.83, dao động -6.38–5400.</li>
                <li><b>required_car_parking_spaces:</b> Trung bình 0.06, tối đa 8.</li>
                <li><b>total_of_special_requests:</b> Trung bình 0.57, tối đa 5.</li>
            </ul>
            <h4 style="color: #e74c3c;">⚠️ NOISY DATA FEATURE</h4>
            <ul>
                <li><b>adr:</b> Giá âm (-6.38) không hợp lý → có thể lỗi dữ liệu.</li>
                <li><b>adults:</b> Giá trị tối thiểu 0 → có thể lỗi nếu không có người lớn.</li>
                <li><b>children, babies:</b> Giá trị tối đa 10 → có thể ngoại lệ hoặc lỗi.</li>
            </ul>
        </div>
        """
        
        html = f"""
        <h2 style="color: {main_color}; border-left: 5px solid {main_color};">2. THỐNG KÊ MÔ TẢ BIẾN SỐ</h2>
        
        <p>Chi tiết các chỉ số thống kê (Mean, Std, Min, Max, Quartiles):</p>
        {table_html}
        {numeric_insights} 
        """
        return html
    
    # --- PHẦN 3: THỐNG KÊ BIẾN PHÂN LOẠI (Count, Unique, Top, Freq) ---
    def generate_categorical_stats(self):
        valid_cat_cols = [c for c in self.target_cat_cols if c in self.df.columns]
        desc_cat = self.df[valid_cat_cols].describe(include=['object']).T
        
        main_color = "#3498db"
        header_bg = "#e3f2fd"
        
        table_html = f"""
        <div style="width: 100%;">
        <table class="table cat-table" style="background-color: transparent; border: 2px solid {main_color}; width: 100%;">
            <thead>
                <tr style="background-color: {header_bg}; color: {main_color}; font-weight: bold;">
                    <th style="border: 1px solid {main_color}; padding: 10px;"></th>
                    <th style="border: 1px solid {main_color}; padding: 10px;">count</th>
                    <th style="border: 1px solid {main_color}; padding: 10px;">unique</th>
                    <th style="border: 1px solid {main_color}; padding: 10px;">top</th>
                    <th style="border: 1px solid {main_color}; padding: 10px;">freq</th>
                </tr>
            </thead>
            <tbody>
        """
        for idx, row in desc_cat.iterrows():
            freq_formatted = f"{row['freq']:,}"
            table_html += f"""
                <tr style="border: 1px solid #ddd;">
                    <td style="font-weight: bold; background-color: {header_bg}; border: 1px solid {main_color}; padding: 8px;">{idx}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{row['count']:,}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{row['unique']:,}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{row['top']}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{freq_formatted}</td>
                </tr>
            """
        table_html += "</tbody></table></div>"
        
         # Nhận xét
        categorical_insights = f"""        
        <div class="comment-box" style="background-color: #e8f4f8; border: 2px solid {main_color}; margin-top: 20px;">
            <h3 style="color: {main_color};">📊 NHẬN XÉT THỐNG KÊ BIẾN PHÂN LOẠI</h3>
            <ul>
                <li>Dữ liệu cho thấy sự phân hóa rõ rệt ở các biến như <b>hotel</b>, <b>market_segment</b>.</li>
                <li>Biến <b>country</b> có số lượng unique lớn, cần cẩn trọng khi One-hot Encoding.</li>
            </ul>
        </div>
        """
        
        html = f"""
        <h2 style="color: {main_color}; border-left: 5px solid {main_color};">3. THỐNG KÊ MÔ TẢ BIẾN PHÂN LOẠI</h2>
        {table_html}
        {categorical_insights}
        """
        return html

    def run_all(self):
        plots = []
        logging.info("Đang vẽ biểu đồ và phân tích dữ liệu...")

        try:
            # --- 0. BOXPLOT ---
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            cols_box = ['lead_time', 'adr', 'total_nights']
            
            for i, col in enumerate(cols_box):
                limit = self.df[col].quantile(0.99)
                data_sub = self.df[self.df[col] < limit]
                sns.boxplot(data=data_sub, x='is_canceled', y=col, ax=axes[i], palette="Set2")
                axes[i].set_title(f"Boxplot: {col}")
                
            plt.tight_layout()
            fname, title = self.save_plot("Biểu đồ Hộp (Boxplot) phát hiện Outlier", "boxplot_outliers")
            
            med_lead_1 = self.df[self.df['is_canceled']=='1']['lead_time'].median()
            med_lead_0 = self.df[self.df['is_canceled']=='0']['lead_time'].median()
            comment = f"""
            <b>Phân tích Outlier và Phân phối:</b>
            <ul>
                <li><b>Lead Time:</b> Trung vị nhóm Hủy ({med_lead_1:.0f} ngày) cao hơn nhóm Không hủy ({med_lead_0:.0f} ngày).
                    <br><i>=> Khách đặt càng sớm, nguy cơ hủy càng cao.</i>
                    <br><i>=> <b>lead_time</b> là đặc trưng trọng số cao.</i>
                </li>
                <br>
                <li><b>ADR:</b> Giá trung bình đồng đều, nhưng nhiều outliers.
                    <br><i>=> <b>adr</b> đơn lẻ không phải yếu tố phân loại mạnh.</i>
                </li>
                <br>
                <li><b>Total Nights:</b> Hai nhóm tương đồng (2-4 đêm).
                    <br><i>=> <b>total_nights</b> ít ảnh hưởng đến quyết định hủy.</i>
                </li>
            </ul>
            """
            plots.append((fname, title, comment))

            # --- 1. HISTOGRAM ---
            plt.figure(figsize=(14, 8))
            cols = ['lead_time', 'adr', 'total_nights', 'total_guests']
            for i, col in enumerate(cols, 1):
                plt.subplot(2, 2, i)
                data_plot = self.df[self.df[col] < self.df[col].quantile(0.99)]
                sns.histplot(data=data_plot, x=col, kde=True, bins=30, color="#3498db")
                plt.title(f"Phân bố {col}")
            plt.tight_layout()
            fname, title = self.save_plot("Phân phối các biến số (Histogram)", "histogram_vars")
            
            avg_lead = self.df['lead_time'].mean()
            avg_price = self.df['adr'].mean()
            
            comment = f"""
            <b>Phân tích Phân phối và Outlier:</b>
            <ul>
                <li><b>lead_time:</b> Thời gian đặt phòng trước trung bình là <b>{avg_lead:.0f} ngày</b>, phân phối lệch phải (Right-skewed).
                    <br><i>=> Khách đặt càng sớm, nguy cơ hủy càng cao.</i>
                </li>
                <br>
                <li><b>adr:</b> Giá trung bình 2 nhóm khá đồng đều ~<b>{avg_price:.1f} USD</b>, phân phối  có dạng gần chuẩn (Bell-shaped) nhưng hơi lệch phải nhẹ..
                    <br><i>=> Cần kiểm tra kỹ nhóm <b>adr = 0</b> (lỗi dữ liệu, phòng miễn phí hay booking bị hủy nên giá về 0?) trước khi đưa vào mô hình.</i>
                </li>
                <br>
                <li><b>total_nights:</b> Đỉnh cao nhất ở 1-4 đêm. Đỉnh nhỏ ở 7 đêm.
                    <br><i>=> Chủ yếu là khách ngắn hạn. Đỉnh 7 đêm gợi ý nhóm khách du lịch theo tuần.</i>
                </li>
                <br>
                <li><b>total_guests:</b> Đa số là 1-2 khách. Số lượng khách từ 3 trở lên giảm dần đều.
                    <br><i>=> Khách sạn này chủ yếu phục vụ các cá nhân hoặc cặp đôi, ít nhóm lớn.</i>
                </li>
            </ul>
            """    
            plots.append((fname, title, comment))
            
            # --- 2. COUNTPLOT ---
            plt.figure(figsize=(16, 10))
            cats = ['hotel', 'arrival_date_month', 'market_segment', 'customer_type']
            for i, col in enumerate(cats, 1):
                plt.subplot(2, 2, i)
                sns.countplot(data=self.df, y=col, order=self.df[col].value_counts().index, palette="viridis")
                plt.title(f"Phân bố {col}")
            plt.tight_layout()
            fname, title = self.save_plot("Thống kê các nhóm phân loại", "countplot_cats")
            
            top_hotel = self.df['hotel'].mode()[0]
            top_segment = self.df['market_segment'].mode()[0]
        
            comment = f"""
            <b>Thống kê các nhóm phân loại chính:</b>
            <ul>
                <li><b>hotel:</b> Số lượng đặt phòng ở City Hotel đôi so với Resort Hotel (khoảng 80.000 so với 40.000)).</li>
                <li><b>month:</b> Cao điểm mùa Hè/Thu (tháng 7, 8, 9, 10). Thấp điểm mùa Đông (tháng 1, 11, 12).</li>
                <li><b>market_segment:</b>
                    <ul>
                        <li><b>Online TA</b> chiếm đa số (~60.000 lượt) -> Khách chủ yếu đặt qua mạng.</li>
                        <li><b>Offline TA/TO</b> đứng thứ 2. Kênh Direct và Corporate chiếm tỷ trọng nhỏ.</li>
                    </ul>
                </li>
                <li><b>customer_type:</b> Mất cân bằng nghiêm trọng. Nhóm <b>Transient</b> (khách lẻ) chiếm đa số áp đảo (~90.000).</li>
            </ul>
            """
            plots.append((fname, title, comment))
            # --- 3. TOP 10 QUỐC GIA ---
            plt.figure(figsize=(12, 6))
            top_countries = self.df['country'].value_counts().head(10)
            sns.barplot(x=top_countries.values, y=top_countries.index, palette="Blues_r")
            plt.title("Top 10 Quốc gia có lượng khách lớn nhất")
            fname, title = self.save_plot("Nguồn khách theo Quốc gia", "top_10_countries")
            
            top_country = top_countries.index[0]
            val = top_countries.values[0]
            comment = f"Khách hàng đến từ <b>{top_country}</b> chiếm áp đảo ({val:,} lượt). Đây là điều dễ hiểu vì các khách sạn này ở Đào Nha (Portugal).  \
                        => Cần lưu ý feature này có thể gây Overfitting nếu mô hình học quá kỹ vào quốc gia này."
            plots.append((fname, title, comment))

            # --- 4. TỶ LỆ HỦY THEO PHÂN KHÚC ---
            plt.figure(figsize=(10, 6))
            # Convert lại sang int để tính mean
            self.df['is_canceled_int'] = pd.to_numeric(self.df['is_canceled'], errors='coerce')
            segment_cancel = self.df.groupby('market_segment')['is_canceled_int'].mean().sort_values(ascending=False)
            sns.barplot(x=segment_cancel.values, y=segment_cancel.index, palette="Reds_r")
            plt.title("Tỷ lệ hủy theo Phân khúc thị trường")
            fname, title = self.save_plot("Rủi ro hủy theo từng phân khúc", "segment_cancel")
            
            highest_seg = segment_cancel.index[0]
            second_highest_seg = segment_cancel.index[1]
            third_highest_seg = segment_cancel.index[2]
            lowest_seg = segment_cancel.index[-1]
            # Lấy các giá trị cần thiết
            val_seg = segment_cancel.values[0]   # Tỷ lệ nhóm cao nhất
            val_seg_1 = segment_cancel.values[0] # Tỷ lệ nhóm 1
            val_seg_2 = segment_cancel.values[1] # Tỷ lệ nhóm 2
            val_seg_3 = segment_cancel.values[2] # Tỷ lệ nhóm 3
            val_seg_last = segment_cancel.values[-1] # Tỷ lệ nhóm thấp nhất

            comment = (
                f"Phân khúc <b>{highest_seg}</b> có tỷ lệ hủy cao nhất ({val_seg:.1%}).<br>"
                f"Phân khúc <b>{second_highest_seg}</b> và <b>{third_highest_seg}</b> có mức rủi ro trung bình cao "
                f"({val_seg_3:.1%} - {val_seg_2:.1%}).<br>"
                f"Các phân khúc khác rủi ro thấp hơn (dưới 20%), trong đó <b>{lowest_seg}</b> thấp nhất ({val_seg_last:.1%}).<br>"
                f"<i>=> <b>market_segment</b> là biến quan trọng dự báo khả năng hủy phòng.</i>"
            )
            plots.append((fname, title, comment))

            # --- 5. LEAD TIME KDE ---
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=self.df, x='lead_time', hue='is_canceled', fill=True, common_norm=False, palette="husl")
            plt.xlim(0, 400)
            fname, title = self.save_plot("Phân phối Lead Time: Hủy vs Không hủy", "lead_time_kde")
            
            comment = "Đường màu cam (Hủy) nhô cao và kéo dài về phía bên phải hơn đường màu xanh. Khách đặt phòng từ rất sớm (Lead Time > 100) có xác suất hủy cao hơn hẳn."
            plots.append((fname, title, comment))
            
            # --- 6. DEPOSIT TYPE VS CANCEL ---
            plt.figure(figsize=(8, 6))
            sns.countplot(data=self.df, x='deposit_type', hue='is_canceled', palette="Set2")
            plt.title("So sánh Hủy phòng theo Loại đặt cọc")
            fname, title = self.save_plot("Mối quan hệ: deposit_type và is_canceled", "deposit_cancel_countplot")
            
            # Tính toán số liệu
            non_refund_cancel = self.df[self.df['deposit_type']=='Non Refund']['is_canceled_int'].mean()
            
            comment_dep = f"""
            <b>Nhận xét & Insight:</b>
            <ul>
                <li><b>Phổ biến nhất:</b> Đa số khách hàng chọn hình thức <b>No Deposit</b> (Không đặt cọc).</li>
                <li><b>Nghịch lý Non Refund:</b> Nhóm "Không hoàn tiền" có tỷ lệ hủy cực cao (<b>{non_refund_cancel:.1%}</b>), trái ngược với logic thông thường.</li>
                <li><b>Nguyên nhân:</b> Trong bộ dữ liệu này, nhóm Non Refund chủ yếu là các booking giữ chỗ số lượng lớn từ <b>Khách đoàn (Groups)</b> hoặc <b>Đại lý (Agents)</b> tại Bồ Đào Nha, sau đó bị hủy hàng loạt hoặc do lỗi thanh toán.</li>
            </ul>
            """
            plots.append((fname, title, comment_dep))

            # --- 7. TOTAL_OF_SPECIAL_REQUESTS VS IS_CANCEL ---
            plt.figure(figsize=(8, 6))
            sns.countplot(data=self.df, x='total_of_special_requests', hue='is_canceled', palette="Paired")
            plt.title("So sánh hủy phòng theo yêu cầu đặc biệt")
            fname, title = self.save_plot("Mối quan hệ: total_of_special_requests vs is_canceled", "requests_cancel_countplot")
            
            comment_req = """
            <b>Nhận xét:</b><br>
            - Khách hàng không có yêu cầu đặc biệt nào (0 requests) có tỷ lệ hủy cao nhất.<br>
            - Khi số lượng yêu cầu tăng lên (1, 2, 3...), tỷ lệ hủy giảm đi rõ rệt.<br>
            <b>=> Điều này cho thấy những khách hàng có trao đổi, tương tác thêm thường là những khách hàng nghiêm túc với chuyến đi.</b>
            """
            plots.append((fname, title, comment_req))
            
            # --- 8. CORRELATION HEATMAP ---
            plt.figure(figsize=(10, 8))
            # Chỉ lấy các cột số quan trọng
            numeric_cols_heat = ['is_canceled', 'lead_time', 'arrival_date_year', 'arrival_date_week_number', 
                            'adults', 'children', 'babies', 'is_repeated_guest', 
                            'previous_cancellations', 'booking_changes', 'days_in_waiting_list', 'adr', 
                            'required_car_parking_spaces', 'total_of_special_requests']
            valid_cols_heat = [c for c in numeric_cols_heat if c in self.df.columns]
            numeric_df = self.df[valid_cols_heat]
            # Thêm lại cột target dạng số
            numeric_df['is_canceled'] = self.df['is_canceled_int']
            
            mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
            sns.heatmap(numeric_df.corr(), mask=mask, cmap="coolwarm", center=0, vmax=1, vmin=-1, linewidths=0.5)
            plt.title("Ma trận tương quan")
            fname, title = self.save_plot("Ma trận tương quan (Correlation Matrix)", "heatmap_corr")
            
            comment = "Biến <b>lead_time</b> và <b>previous_cancellations</b> có tương quan dương với việc hủy. Ngược lại, <b>total_of_special_requests</b> và <b>parking_spaces</b> có tương quan âm (càng nhiều request càng ít hủy)."
            plots.append((fname, title, comment))


            # --- TONG HOP HTML ---
            self.create_html_report(plots)
            
        except Exception as e:
            logging.error(f"Lỗi trong quá trình vẽ biểu đồ: {str(e)}")
            raise e

    def create_html_report(self, plots):
        logging.info("Đang tạo báo cáo HTML...")
        main_color = "#3498db"
        bg_color = "#e3f2fd"

        html = f"""
        <!DOCTYPE html>
        <html><head><meta charset="utf-8"><title>Full EDA Report</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #eee; padding-bottom: 20px; }}
            h2 {{ margin-top: 40px; padding-left: 10px; }}
            .summary-box {{ background: {bg_color}; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .chart-section {{ margin-bottom: 60px; page-break-inside: avoid; }}
            .comment-box {{ padding: 15px; margin-top: 15px; border-radius: 4px; }}
            img {{ width: 100%; border-radius: 5px; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            hr {{ border: 0; border-top: 1px solid #eee; margin: 40px 0; }}
            ul {{ line-height: 1.6; }}
        </style>
        </head><body>
        <div class="container">
            <h1>BÁO CÁO PHÂN TÍCH DỮ LIỆU TOÀN DIỆN</h1>
            {self.generate_overview_section()}
            <h2 style="color: {main_color}; border-left: 5px solid {main_color};">2. THỐNG KÊ MÔ TẢ BIẾN SỐ</h2>
            {self.generate_numeric_stats().split('</h2>')[1]} 
            {self.generate_categorical_stats()}
            <hr>
            <h2 style="color: {main_color}; border-left: 5px solid {main_color};">4. BIỂU ĐỒ TRỰC QUAN HÓA & INSIGHTS</h2>
            {self.generate_toc(plots)}
            <br>
        """
        for i, (filename, title, comment) in enumerate(plots, 1):
            html += f"""
            <div class="chart-section" id="chart_{i}">
                <h3>{i}. {title}</h3>
                <img src="images/{filename}" alt="{title}">
                <div class="comment-box" style="background-color: {bg_color}; border-left: 5px solid {main_color};">
                    {comment}
                </div>
            </div>
            """
        html += """<div style="text-align:center; margin-top:50px; color:#888;">Bao cao duoc tao tu dong.</div></div></body></html>"""
        
        report_path = os.path.join(REPORT_DIR, "FULL_EDA_REPORT.html")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html)
            logging.info(f"Đã xuất báo cáo đầy đủ tại: {report_path}")
            print(f"\nĐã xuất báo cáo đầy đủ tại: {report_path}")
        except Exception as e:
            logging.error(f"Lỗi khi ghi file báo cáo: {str(e)}")

    def generate_toc(self, plots):
        """Tạo Mục lục"""
        html = "<h3>MỤC LỤC BIỂU ĐỒ</h3><ul>"
        for i, (_, title, _) in enumerate(plots, 1):
            html += f"<li><a href='#chart_{i}' style='text-decoration:none; color:#2980b9;'>{i}. {title}</a></li>"
        html += "</ul>"
        return html

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    logging.info(f"Script bắt đầu lúc: {datetime.datetime.now()}")
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            eda = EDAHotelBooking(df)
            eda.run_all()
        except Exception as e:
            logging.critical(f"LỖI: {str(e)}")
            print(f"Có lỗi xảy ra: {e}")
    else:
        msg = f"Lỗi: Không tìm thấy file dữ liệu tại {DATA_PATH}"
        logging.error(msg)
        print(msg)
    logging.info(f"Script kết thúc lúc: {datetime.datetime.now()}")
