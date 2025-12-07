import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- CẤU HÌNH ---
DATA_PATH = "data/raw/hotel_bookings.csv"
REPORT_DIR = "reports"
IMG_DIR = os.path.join(REPORT_DIR, "images")

# Tạo thư mục nếu chưa có
os.makedirs(IMG_DIR, exist_ok=True)

class EDAHotelBooking:
    def __init__(self, df):
        self.df = df.copy()
        self.img_count = 0
        
        # 1. Feature Engineering nhẹ để vẽ biểu đồ
        self.df['total_nights'] = self.df['stays_in_weekend_nights'] + self.df['stays_in_week_nights']
        self.df['total_guests'] = self.df['adults'] + self.df['children'].fillna(0) + self.df['babies']
        self.df['has_requests'] = self.df['total_of_special_requests'] > 0

        # 2. Danh sách các cột phân loại cần phân tích
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
                self.df[col] = self.df[col].astype(str)

    def save_plot(self, title_vn, filename_slug):
        """Lưu biểu đồ vào folder images"""
        self.img_count += 1
        filename = f"{self.img_count:02d}_{filename_slug}.png"
        filepath = os.path.join(IMG_DIR, filename)
        
        # Lưu file ảnh
        plt.savefig(filepath, bbox_inches='tight', dpi=100)
        plt.close() 
        return filename, title_vn

    # --- PHẦN 1: TỔNG QUAN ---
    def generate_overview_section(self):
        n_rows, n_cols = self.df.shape
        
        # Đếm kiểu dữ liệu
        obj_cols = len(self.df.select_dtypes(include=['object']).columns)
        int_cols = len(self.df.select_dtypes(include=['int64', 'int32']).columns)
        float_cols = len(self.df.select_dtypes(include=['float64']).columns)
        
        cols_to_check = ['children', 'country', 'agent', 'company']
        missing_html = "<ul>"
        for col in cols_to_check:
            if col in self.df.columns:
                miss_val = self.df[col].isnull().sum()
                if miss_val > 0:
                    missing_html += f"<li>Cột <b>{col}</b>: {miss_val:,} giá trị thiếu ({miss_val/n_rows:.1%}).</li>"
        missing_html += "</ul>"
        
        html = f"""
        <h2>1.  TỔNG QUAN DỮ LIỆU</h2>
        <div class="comment-box" style="background-color: #e8f4f8; border-color: #3498db; margin-top: 15px;">
            <h3> NHẬN XÉT:</h3>
            <ul>
                <li><b>Số lượng bản ghi:</b> Bộ dữ liệu bao gồm {n_rows:,} bản ghi.</li>
                <li><b>Số lượng cột:</b> Bộ dữ liệu có {n_cols} cột, tương ứng với nhiều thuộc tính khác nhau liên quan đến đặt phòng khách sạn.</li>
                <li><b>Kiểu dữ liệu:</b>
                    <ul>
                        <li>Phần lớn các cột ({obj_cols} cột) thuộc kiểu object (thường là chuỗi hoặc dữ liệu phân loại).</li>
                        <li>{int_cols} cột thuộc kiểu int64, biểu diễn các giá trị số nguyên.</li>
                        <li>{float_cols} cột thuộc kiểu float64, thường biểu diễn các giá trị thập phân.</li>
                    </ul>
                </li>
                <li><b>Giá trị thiếu:</b>
                    <ul>
                        <li>Cột children có {self.df['children'].isnull().sum():,} giá trị bị thiếu.</li>
                        <li>Cột country có {self.df['country'].isnull().sum():,} giá trị bị thiếu.</li>
                        <li>Cột agent có {self.df['agent'].isnull().sum():,} giá trị bị thiếu.</li>
                        <li>Cột company có số lượng giá trị thiếu rất lớn, lên đến {self.df['company'].isnull().sum():,}.</li>
                    </ul>
                    <p><i>Những giá trị thiếu này có thể cần được xử lý tùy theo mục tiêu phân tích hoặc mô hình cần xây dựng.</i></p>
                </li>
            </ul>
            <p><b>Phân loại biến:</b> Dựa trên kiểu dữ liệu và phần mô tả đặc trưng đã trình bày trước đó, xác định được rằng có {len(self.target_cat_cols)} cột là các biến phân loại theo của chúng. Những đặc trưng này cần có kiểu dữ liệu dạng chuỗi (object) để đảm bảo quá trình phân tích và diễn giải trong các bước tiếp theo được thực hiện chính xác.</p>
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
        
        # Describe mặc định đã có đủ: count, mean, std, min, 25%, 50%, 75%, max
        desc = self.df[valid_cols].describe().T
        
        
        table_html = """
        <table class="table numeric-table" style="background-color: #ffffff; border: 2px solid #3498db;">
            <thead>
                <tr style="background-color: #e3f2fd; color: #1565c0; font-weight: bold;">
                    <th style="border: 1px solid #3498db; padding: 10px;"></th>
                    <th style="border: 1px solid #3498db; padding: 10px;">count</th>
                    <th style="border: 1px solid #3498db; padding: 10px;">mean</th>
                    <th style="border: 1px solid #3498db; padding: 10px;">std</th>
                    <th style="border: 1px solid #3498db; padding: 10px;">min</th>
                    <th style="border: 1px solid #3498db; padding: 10px;">25%</th>
                    <th style="border: 1px solid #3498db; padding: 10px;">50%</th>
                    <th style="border: 1px solid #3498db; padding: 10px;">75%</th>
                    <th style="border: 1px solid #3498db; padding: 10px;">max</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Thêm từng hàng dữ liệu
        for idx, row in desc.iterrows():
            table_html += f"""
                <tr style="border: 1px solid #ddd;">
                    <td style="font-weight: bold; background-color: #f0f8ff; border: 1px solid #3498db; padding: 8px;">{idx}</td>
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
        
        table_html += """
            </tbody>
        </table>
        """
        
        # Thống kê chi tiết cho từng biến
        numeric_insights = """
        <div class="comment-box" style="background-color: #e8f4f8; border-color: #3498db; margin-top: 20px;">
            <h3>📈 NHẬN XÉT THỐNG KÊ MÔ TẢ BIẾN SỐ</h3>
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
        <h2>2. THỐNG KÊ MÔ TẢ BIẾN SỐ</h2>
        <p>Chi tiết các chỉ số thống kê (Mean, Std, Min, Max, Quartiles):</p>
        <div style="overflow-x:auto;">
            {table_html}
        </div>
        {numeric_insights}
        """
        return html

    # --- PHẦN 3: THỐNG KÊ BIẾN PHÂN LOẠI (Count, Unique, Top, Freq) ---
    def generate_categorical_stats(self):
        valid_cat_cols = [c for c in self.target_cat_cols if c in self.df.columns]
        
        # Describe object để lấy count, unique, top, freq
        desc_cat = self.df[valid_cat_cols].describe(include=['object']).T
        
        # Chỉ lấy 4 cột bạn yêu cầu
        display_cols = ['count', 'unique', 'top', 'freq']
        
        # Tạo bảng HTML với màu xanh lá nhạt (giống trong hình)
        table_html = """
        <table class="table cat-table" style="background-color: #ffffff; border: 2px solid #2ecc71;">
            <thead>
                <tr style="background-color: #d4efdf; color: #27ae60; font-weight: bold;">
                    <th style="border: 1px solid #2ecc71; padding: 10px;"></th>
                    <th style="border: 1px solid #2ecc71; padding: 10px;">count</th>
                    <th style="border: 1px solid #2ecc71; padding: 10px;">unique</th>
                    <th style="border: 1px solid #2ecc71; padding: 10px;">top</th>
                    <th style="border: 1px solid #2ecc71; padding: 10px;">freq</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Thêm từng hàng dữ liệu
        for idx, row in desc_cat.iterrows():
            freq_formatted = f"{row['freq']:,}"
            table_html += f"""
                <tr style="border: 1px solid #ddd;">
                    <td style="font-weight: bold; background-color: #eafaf1; border: 1px solid #2ecc71; padding: 8px;">{idx}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{row['count']:,}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{row['unique']:,}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{row['top']}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{freq_formatted}</td>
                </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        """
        
        # Thống kê chi tiết cho từng biến phân loại
        categorical_insights = """
        <div class="comment-box" style="background-color: #e8f4f8; border-color: #3498db; margin-top: 20px;">
            <h3>📊 NHẬN XÉT THỐNG KÊ MÔ TẢ BIẾN PHÂN LOẠI</h3>
            <ul>
                <li><b>hotel:</b> Có hai loại khách sạn, trong đó "City Hotel" là loại xuất hiện nhiều nhất với 79,330 lần trên tổng số 119,390 bản ghi.</li>
                <li><b>is_canceled:</b> Có hai giá trị (0 = không hủy, 1 = hủy). Giá trị phổ biến nhất là "0" (không hủy), xuất hiện 75,166 lần.</li>
                <li><b>arrival_date_year:</b> Có ba năm khác nhau trong dữ liệu, trong đó năm 2016 là phổ biến nhất với 56,707 lượt đến.</li>
                <li><b>arrival_date_month:</b> Có 12 tháng khác nhau. Tháng có lượt đến nhiều nhất là tháng 8 với 13,877 lượt.</li>
                <li><b>meal:</b> Có 5 loại bữa ăn được đặt. Loại phổ biến nhất là "BB", xuất hiện 92,310 lần.</li>
                <li><b>country:</b> Có 178 quốc gia xuất hiện trong dữ liệu. Quốc gia phổ biến nhất là "PRT" (Bồ Đào Nha), với 48,590 lượt.</li>
                <li><b>market_segment:</b> Có 8 phân khúc thị trường, trong đó phổ biến nhất là "Online TA" với 56,477 lượt.</li>
                <li><b>distribution_channel:</b> Có 5 kênh phân phối đặt phòng, với "TA/TO" là phổ biến nhất, xuất hiện 97,870 lần.</li>
                <li><b>is_repeated_guest:</b> Có hai giá trị (0 = khách mới, 1 = khách quay lại). Giá trị phổ biến nhất là "0".</li>
                <li><b>reserved_room_type và assigned_room_type:</b> Có nhiều loại phòng khác nhau, một số loại xuất hiện thường xuyên hơn các loại khác.</li>
                <li><b>deposit_type:</b> Có 3 loại đặt cọc, trong đó "No Deposit" là phổ biến nhất với 104,641 lượt.</li>
                <li><b>agent:</b> Có 334 đại lý, trong đó mã '9.0' là phổ biến nhất với 31,961 lượt.</li>
                <li><b>company:</b> Có 353 công ty, nhưng giá trị 'nan' (thiếu dữ liệu) chiếm nhiều nhất với 112,593 lượt, cho thấy biến này có tỷ lệ thiếu rất cao.</li>
                <li><b>customer_type:</b> Có 4 loại khách hàng, phổ biến nhất là "Transient" với 89,613 lượt.</li>
                <li><b>reservation_status:</b> Có 3 trạng thái đặt phòng, trong đó "Check-Out" là phổ biến nhất với 75,166 lượt.</li>
                <li><b>reservation_status_date:</b> Có 926 ngày khác nhau, trong đó ngày phổ biến nhất là '2015-10-21' với 1,461 lượt.</li>
                <li><b>name:</b> Có 81,503 tên khác nhau, với 'Michael Johnson' là tên xuất hiện nhiều nhất (48 lần).</li>
                <li><b>email:</b> Có 115,889 email khác nhau, với 'Michael.C@gmail.com' là phổ biến nhất (6 lần).</li>
                <li><b>phone-number:</b> Có 119,390 số điện thoại khác nhau, nghĩa là gần như mỗi khách đều có số riêng.</li>
                <li><b>credit_card:</b> Có 9,000 số thẻ tín dụng, trong đó '***4923' là phổ biến nhất (28 lần).</li>
            </ul>
        </div>
        """
        
        html = f"""
        <h2>3. THỐNG KÊ MÔ TẢ BIẾN PHÂN LOẠI</h2>
        <div style="overflow-x:auto;">
            {table_html}
        </div>
        {categorical_insights}
        """
        return html

    # --- PHẦN 4: VẼ BIỂU ĐỒ (Thêm Boxplot) ---
    def run_all(self):
        plots = []
        print("⏳ Đang vẽ biểu đồ và phân tích dữ liệu (Full Option)...")

        # --- 0. BOXPLOT (MỚI THÊM) ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        cols_box = ['lead_time', 'adr', 'total_nights']
        
        for i, col in enumerate(cols_box):
            # Lọc bớt nhiễu 1% để biểu đồ dễ nhìn hơn
            limit = self.df[col].quantile(0.99)
            data_sub = self.df[self.df[col] < limit]
            
            sns.boxplot(data=data_sub, x='is_canceled', y=col, ax=axes[i], palette="Set2")
            axes[i].set_title(f"Boxplot: {col}")
        
        plt.tight_layout()
        fname, title = self.save_plot("Biểu đồ Hộp (Boxplot) phát hiện Outlier", "boxplot_outliers")
        
        # Nhận xét Boxplot
        med_lead_1 = self.df[self.df['is_canceled']=='1']['lead_time'].median()
        med_lead_0 = self.df[self.df['is_canceled']=='0']['lead_time'].median()
        comment = f"""
        <b>Phân tích Outlier và Phân phối:</b><br>
        - <b>Lead Time:</b> Trung vị của nhóm Hủy ({med_lead_1:.0f} ngày) cao hơn đáng kể so với nhóm Không hủy ({med_lead_0:.0f} ngày). Hộp IQR của nhóm hủy cũng dài hơn, cho thấy sự biến động lớn.<br>
        - <b>ADR:</b> Giá phòng có nhiều điểm ngoại lai (Outliers) ở phía trên. Điều này gợi ý nên sử dụng RobustScaler thay vì StandardScaler khi tiền xử lý.
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
        comment = f"Thời gian đặt trước trung bình là <b>{avg_lead:.0f} ngày</b>, nhưng phân phối bị lệch phải (Right-skewed). Giá phòng trung bình (ADR) khoảng <b>{avg_price:.1f} USD</b>."
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
        comment = f"Dữ liệu bị mất cân bằng ở Customer Type (Transient chiếm đa số). Loại hình <b>{top_hotel}</b> và phân khúc <b>{top_segment}</b> là phổ biến nhất."
        plots.append((fname, title, comment))

        # --- 3. TOP 10 QUỐC GIA ---
        plt.figure(figsize=(12, 6))
        top_countries = self.df['country'].value_counts().head(10)
        sns.barplot(x=top_countries.values, y=top_countries.index, palette="Blues_r")
        plt.title("Top 10 Quốc gia có lượng khách lớn nhất")
        fname, title = self.save_plot("Nguồn khách theo Quốc gia", "top_10_countries")
        
        top_country = top_countries.index[0]
        val = top_countries.values[0]
        comment = f"Khách hàng đến từ <b>{top_country}</b> chiếm áp đảo ({val:,} lượt). Cần lưu ý feature này có thể gây Overfitting nếu mô hình học quá kỹ vào quốc gia này."
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
        val_seg = segment_cancel.values[0]
        comment = f"Phân khúc <b>{highest_seg}</b> có tỷ lệ hủy cao nhất ({val_seg:.1%}). Nhóm khách này rủi ro cao, cần features riêng để bắt tín hiệu hủy."
        plots.append((fname, title, comment))

        # --- 5. CORRELATION HEATMAP ---
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

        # --- 6. LEAD TIME KDE ---
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=self.df, x='lead_time', hue='is_canceled', fill=True, common_norm=False, palette="husl")
        plt.xlim(0, 400)
        fname, title = self.save_plot("Phân phối Lead Time: Hủy vs Không hủy", "lead_time_kde")
        
        comment = "Đường màu cam (Hủy) nhô cao và kéo dài về phía bên phải hơn đường màu xanh. Khách đặt phòng từ rất sớm (Lead Time > 100) có xác suất hủy cao hơn hẳn."
        plots.append((fname, title, comment))

        # --- 7. SPECIAL REQUESTS ---
        plt.figure(figsize=(8, 6))
        sns.countplot(data=self.df, x='total_of_special_requests', hue='is_canceled', palette="Paired")
        plt.title("Số lượng yêu cầu đặc biệt vs Trạng thái hủy")
        fname, title = self.save_plot("Tác động của Special Requests", "special_requests")
        
        comment = "Rõ ràng: Khách hàng không có yêu cầu đặc biệt nào (0 requests) có tỷ lệ hủy cực cao. Khách có yêu cầu thường là khách 'thật' và cam kết ở lại."
        plots.append((fname, title, comment))

        # --- TỔNG HỢP HTML ---
        self.create_html_report(plots)

    def create_html_report(self, plots):
        html = f"""
        <!DOCTYPE html>
        <html><head><meta charset="utf-8"><title>Full EDA Report</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #eee; padding-bottom: 20px; }}
            h2 {{ color: #3498db; margin-top: 40px; border-left: 5px solid #3498db; padding-left: 10px; }}
            .summary-box {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .chart-section {{ margin-bottom: 60px; page-break-inside: avoid; }}
            .comment-box {{ background-color: #fff3cd; border: 1px solid #ffeeba; border-left: 5px solid #ffc107; padding: 15px; margin-top: 15px; border-radius: 4px; }}
            img {{ width: 100%; border-radius: 5px; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
            
            /* CSS cho bảng */
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            
            .table th {{ background-color: #f8f9fa; }}
            .table tr:nth-child(even) {{ background-color: #f8f9fa; }}
            .table tr:hover {{ background-color: #e9ecef; }}
            
            hr {{ border: 0; border-top: 1px solid #eee; margin: 40px 0; }}
            ul {{ line-height: 1.6; }}
        </style>
        </head><body>
        <div class="container">
            <h1>📊 BÁO CÁO PHÂN TÍCH DỮ LIỆU TOÀN DIỆN</h1>
            
            {self.generate_overview_section()}
            
            {self.generate_numeric_stats()}
            
            {self.generate_categorical_stats()}
            
            <hr>
            
            <h2>4. BIỂU ĐỒ TRỰC QUAN HÓA & INSIGHTS</h2>
            {self.generate_toc(plots)}
            <br>
        """
        
        for i, (filename, title, comment) in enumerate(plots, 1):
            html += f"""
            <div class="chart-section" id="chart_{i}">
                <h3>{i}. {title}</h3>
                <img src="images/{filename}" alt="{title}">
                <div class="comment-box">
                    <strong>💡 Insight:</strong> {comment}
                </div>
            </div>
            """
            
        html += """<div style="text-align:center; margin-top:50px; color:#888;">Report generated by Python Auto-EDA v2.0</div></div></body></html>"""
        
        report_path = os.path.join(REPORT_DIR, "FULL_EDA_REPORT.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n✅ XONG! Đã xuất báo cáo đầy đủ tại: {report_path}")

    def generate_toc(self, plots):
        """Tạo Mục lục"""
        html = "<h3>MỤC LỤC BIỂU ĐỒ</h3><ul>"
        for i, (_, title, _) in enumerate(plots, 1):
            html += f"<li><a href='#chart_{i}' style='text-decoration:none; color:#2980b9;'>{i}. {title}</a></li>"
        html += "</ul>"
        return html

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            eda = EDAHotelBooking(df)
            eda.run_all()
        except Exception as e:
            print(f"❌ Có lỗi xảy ra: {e}")
    else:
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu tại {DATA_PATH}")