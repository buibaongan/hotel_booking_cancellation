import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging
import datetime

# --- CONFIGURATION ---
DATA_PATH = "data/raw/hotel_bookings.csv"
REPORT_DIR = "reports"
IMG_DIR = os.path.join(REPORT_DIR, "images")
LOG_FILE = os.path.join(REPORT_DIR, "eda_activity.log")

# Create the directory if it does not exist.
os.makedirs(IMG_DIR, exist_ok=True)

# --- LOGGING CONFIGURATION ---
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
        logging.info("Starting EDAHotelBooking initialization...")
        self.df = df.copy()
        self.img_count = 0

        try:
            # 1. Light feature engineering for plotting.
            self.df['total_nights'] = self.df['stays_in_weekend_nights'] + self.df['stays_in_week_nights']
            self.df['total_guests'] = self.df['adults'] + self.df['children'].fillna(0) + self.df['babies']
            self.df['has_requests'] = self.df['total_of_special_requests'] > 0

            # 2. Categorical columns to analyze.
            self.target_cat_cols = [
                'hotel', 'is_canceled', 'arrival_date_year', 'arrival_date_month', 'meal',
                'country', 'market_segment', 'distribution_channel', 'is_repeated_guest',
                'reserved_room_type', 'assigned_room_type', 'deposit_type', 'agent',
                'company', 'customer_type', 'reservation_status',
                'name', 'email', 'phone-number', 'credit_card'
            ]

            # Cast categorical data to string.
            for col in self.target_cat_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].apply(lambda x: str(x) if pd.notnull(x) else np.nan)

            logging.info(f"Preprocessed data. Shape: {self.df.shape}")

        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            raise e

    def save_plot(self, title_vn, filename_slug):
        """Save the chart to the images folder."""
        self.img_count += 1
        filename = f"{self.img_count:02d}_{filename_slug}.png"
        filepath = os.path.join(IMG_DIR, filename)

        try:
            plt.savefig(filepath, bbox_inches='tight', dpi=100)
            plt.close()
            logging.info(f"Saved chart {self.img_count}: {title_vn}")
            return filename, title_vn
        except Exception as e:
            logging.error(f"Error while saving chart {filename}: {str(e)}")
            return None, title_vn

    def generate_info_missing_table(self):
        """Create the information table."""

        # --- COLOR CONFIGURATION ---
        main_color = "#3498db"    # Dark blue border.
        header_bg = "#e3f2fd"     # Light blue header and first-column background.
        warn_bg = "#ffe6e6"       # Light red warning background.
        text_black = "#333333"    # Black text color.

        html = f"""
        <h3 style="color: {main_color};">Column Details</h3>
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

    # --- SECTION 1: OVERVIEW ---
    def generate_overview_section(self):
        n_rows, n_cols = self.df.shape
        obj_cols = len(self.df.select_dtypes(include=['object']).columns)
        int_cols = len(self.df.select_dtypes(include=['int64', 'int32']).columns)
        float_cols = len(self.df.select_dtypes(include=['float64']).columns)

        missing_series = self.df.isnull().sum()
        missing_cols = missing_series[missing_series > 0].sort_values(ascending=False)

        missing_list_html = ""
        if missing_cols.empty:
            missing_list_html = "<li><b>Clean!</b> This dataset does not contain any missing values.</li>"
        else:
            for col_name, count in missing_cols.items():
                pct = (count / n_rows) * 100
                comment_note = ""
                if pct > 50: comment_note = " <span style='color: #c0392b;'>(Too many missing values)</span>"
                missing_list_html += f"<li>Column <b>{col_name}</b>: {count:,} missing values ({pct:.2f}%).{comment_note}</li>"

        # COLOR CONFIGURATION
        main_color = "#3498db"
        bg_color = "#e3f2fd"

        html = f"""
        <h2 style="color: {main_color}; border-left: 5px solid {main_color};">1. DATA OVERVIEW</h2>
        <div class="comment-box" style="background-color: #e8f4f8; border: 2px solid #3498db; margin-top: 15px;">
            <ul>
                <li><b>Shape:</b> {n_rows:,} rows, {n_cols} columns.</li>
                <ul>
                    <li><b>Record count:</b> The dataset contains {n_rows:,} records.</li>
                    <li><b>Column count:</b> The dataset has {n_cols} columns, covering different hotel booking attributes.</li>
                </ul>
            </ul>
        </div>

        {self.generate_info_missing_table()}

        <div class="comment-box" style="background-color: #e8f4f8; border: 2px solid #3498db; margin-top: 15px;">
            <h3>COMMENTS:</h3>
            <ul>
                <li><b>Data structure:</b>
                    <ul>
                        <li>Most columns ({obj_cols} columns) are object type, usually strings or categorical values.</li>
                        <li>{int_cols} columns are int64 and represent integer values.</li>
                        <li>{float_cols} columns are float64 and usually represent decimal values.</li>
                    </ul>
                </li>
                <li><b>Missing Value Analysis:</b>
                    <ul>
                        {missing_list_html}
                    </ul>
                    <p><i>These missing values should be handled based on the analysis goals or the model being built.</i></p>
                </li>
            </ul>
            <p><b>Variable classification:</b> Based on data types, {len(self.target_cat_cols)} target categorical variables were identified. These features should remain string/object values so later analysis and interpretation are accurate.</p>
        </div>
        """
        return html

    # --- SECTION 2: NUMERIC STATISTICS ---
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

        # Detailed statistics for each variable.
        main_color = "#3498db"
        numeric_insights = f"""
        <div class="comment-box" style="background-color: #e8f4f8; border: 2px solid {main_color}; margin-top: 20px;">
            <h3 style="color: #3498db;">NUMERIC DESCRIPTIVE STATISTICS COMMENTS</h3>
            <ul>
                <li><b>lead_time:</b> Average 104 days, ranging from 0 to 737.</li>
                <li><b>arrival_date_week_number:</b> Average week 27.17, ranging from 1 to 53.</li>
                <li><b>arrival_date_day_of_month:</b> Average day 15.80, ranging from 1 to 31.</li>
                <li><b>stays_in_weekend_nights:</b> Average 0.93 weekend nights, maximum 19.</li>
                <li><b>stays_in_week_nights:</b> Average 2.5 week nights, ranging from 0 to 50.</li>
                <li><b>adults:</b> Average 1.86 adults, ranging from 0 to 55.</li>
                <li><b>children:</b> Average 0.1 children, maximum 10.</li>
                <li><b>babies:</b> Average 0.008 babies, maximum 10.</li>
                <li><b>previous_cancellations:</b> Average 0.09 previous cancellations, maximum 26.</li>
                <li><b>previous_bookings_not_canceled:</b> Average 0.14, maximum 72.</li>
                <li><b>booking_changes:</b> Average 0.22 booking changes, maximum 21.</li>
                <li><b>days_in_waiting_list:</b> Average 2.32 days, maximum 391.</li>
                <li><b>adr:</b> Average 101.83, ranging from -6.38 to 5400.</li>
                <li><b>required_car_parking_spaces:</b> Average 0.06, maximum 8.</li>
                <li><b>total_of_special_requests:</b> Average 0.57, maximum 5.</li>
            </ul>
            <h4 style="color: #e74c3c;">NOISY DATA FEATURE</h4>
            <ul>
                <li><b>adr:</b> A negative price (-6.38) is not reasonable and may indicate a data error.</li>
                <li><b>adults:</b> A minimum value of 0 may be an error if no adults are present.</li>
                <li><b>children, babies:</b> A maximum value of 10 may be an outlier or an error.</li>
            </ul>
        </div>
        """

        html = f"""
        <h2 style="color: {main_color}; border-left: 5px solid {main_color};">2. NUMERIC DESCRIPTIVE STATISTICS</h2>

        <p>Detailed statistics (Mean, Std, Min, Max, Quartiles):</p>
        {table_html}
        {numeric_insights}
        """
        return html

    # --- SECTION 3: CATEGORICAL STATISTICS (Count, Unique, Top, Freq) ---
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

         # Comments.
        categorical_insights = f"""
        <div class="comment-box" style="background-color: #e8f4f8; border: 2px solid {main_color}; margin-top: 20px;">
            <h3 style="color: {main_color};">Categorical Descriptive Statistics Comments</h3>
            <ul>
                <li>The data shows clear segmentation in variables such as <b>hotel</b> and <b>market_segment</b>.</li>
                <li>The <b>country</b> variable has many unique values, so one-hot encoding should be used carefully.</li>
            </ul>
        </div>
        """

        html = f"""
        <h2 style="color: {main_color}; border-left: 5px solid {main_color};">3. CATEGORICAL DESCRIPTIVE STATISTICS</h2>
        {table_html}
        {categorical_insights}
        """
        return html

    def run_all(self):
        plots = []
        logging.info("Drawing charts and analyzing data...")

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
            fname, title = self.save_plot("Boxplot for Outlier Detection", "boxplot_outliers")

            med_lead_1 = self.df[self.df['is_canceled']=='1']['lead_time'].median()
            med_lead_0 = self.df[self.df['is_canceled']=='0']['lead_time'].median()
            comment = f"""
            <b>Outlier and Distribution Analysis:</b>
            <ul>
                <li><b>Lead Time:</b> The median for the canceled group ({med_lead_1:.0f} days) is higher than the non-canceled group ({med_lead_0:.0f} days).
                    <br><i>=> The earlier guests book, the higher the cancellation risk.</i>
                    <br><i>=> <b>lead_time</b> is a highly weighted feature.</i>
                </li>
                <br>
                <li><b>ADR:</b> Average prices are similar, but there are many outliers.
                    <br><i>=> <b>adr</b> alone is not a strong classification factor.</i>
                </li>
                <br>
                <li><b>Total Nights:</b> The two groups are similar (2-4 nights).
                    <br><i>=> <b>total_nights</b> has limited influence on cancellation decisions.</i>
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
                plt.title(f"Distribution of {col}")
            plt.tight_layout()
            fname, title = self.save_plot("Numeric Variable Distributions (Histogram)", "histogram_vars")

            avg_lead = self.df['lead_time'].mean()
            avg_price = self.df['adr'].mean()

            comment = f"""
            <b>Distribution and Outlier Analysis:</b>
            <ul>
                <li><b>lead_time:</b> The average booking lead time is <b>{avg_lead:.0f} days</b>, with a right-skewed distribution.
                    <br><i>=> The earlier guests book, the higher the cancellation risk.</i>
                </li>
                <br>
                <li><b>adr:</b> Average prices are fairly similar between the two groups at ~<b>{avg_price:.1f} USD</b>. The distribution is close to bell-shaped but slightly right-skewed.
                    <br><i>=> The <b>adr = 0</b> group should be checked carefully before modeling to determine whether it is a data error, free room, or canceled booking priced at 0.</i>
                </li>
                <br>
                <li><b>total_nights:</b> The highest peak is at 1-4 nights, with a smaller peak at 7 nights.
                    <br><i>=> Most guests are short-stay guests. The 7-night peak suggests weekly travel patterns.</i>
                </li>
                <br>
                <li><b>total_guests:</b> Most bookings have 1-2 guests. Counts gradually decrease for 3 or more guests.
                    <br><i>=> These hotels mainly serve individuals or couples, with fewer large groups.</i>
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
                plt.title(f"Distribution of {col}")
            plt.tight_layout()
            fname, title = self.save_plot("Categorical Group Counts", "countplot_cats")

            top_hotel = self.df['hotel'].mode()[0]
            top_segment = self.df['market_segment'].mode()[0]

            comment = f"""
            <b>Main categorical group counts:</b>
            <ul>
                <li><b>hotel:</b> City Hotel has about twice as many bookings as Resort Hotel (about 80,000 versus 40,000).</li>
                <li><b>month:</b> Peak season is summer/fall (July, August, September, October). Low season is winter (January, November, December).</li>
                <li><b>market_segment:</b>
                    <ul>
                        <li><b>Online TA</b> is the majority group (~60,000 bookings), meaning customers mainly book online.</li>
                        <li><b>Offline TA/TO</b> is second. Direct and Corporate channels have smaller shares.</li>
                    </ul>
                </li>
                <li><b>customer_type:</b> The distribution is highly imbalanced. The <b>Transient</b> group dominates (~90,000 bookings).</li>
            </ul>
            """
            plots.append((fname, title, comment))
            # --- 3. TOP 10 COUNTRIES ---
            plt.figure(figsize=(12, 6))
            top_countries = self.df['country'].value_counts().head(10)
            sns.barplot(x=top_countries.values, y=top_countries.index, palette="Blues_r")
            plt.title("Top 10 Countries by Number of Guests")
            fname, title = self.save_plot("Guest Sources by Country", "top_10_countries")

            top_country = top_countries.index[0]
            val = top_countries.values[0]
            comment = f"Customers from <b>{top_country}</b> dominate ({val:,} bookings). This is understandable because these hotels are in Portugal.  \
                        => This feature may cause overfitting if the model learns too strongly from this country."
            plots.append((fname, title, comment))

            # --- 4. CANCELLATION RATE BY SEGMENT ---
            plt.figure(figsize=(10, 6))
            # Convert back to int to calculate the mean.
            self.df['is_canceled_int'] = pd.to_numeric(self.df['is_canceled'], errors='coerce')
            segment_cancel = self.df.groupby('market_segment')['is_canceled_int'].mean().sort_values(ascending=False)
            sns.barplot(x=segment_cancel.values, y=segment_cancel.index, palette="Reds_r")
            plt.title("Cancellation Rate by Market Segment")
            fname, title = self.save_plot("Cancellation Risk by Segment", "segment_cancel")

            highest_seg = segment_cancel.index[0]
            second_highest_seg = segment_cancel.index[1]
            third_highest_seg = segment_cancel.index[2]
            lowest_seg = segment_cancel.index[-1]
            # Get the required values.
            val_seg = segment_cancel.values[0]   # Highest group rate.
            val_seg_1 = segment_cancel.values[0] # Group 1 rate.
            val_seg_2 = segment_cancel.values[1] # Group 2 rate.
            val_seg_3 = segment_cancel.values[2] # Group 3 rate.
            val_seg_last = segment_cancel.values[-1] # Lowest group rate.

            comment = (
                f"Segment <b>{highest_seg}</b> has the highest cancellation rate ({val_seg:.1%}).<br>"
                f"Segments <b>{second_highest_seg}</b> and <b>{third_highest_seg}</b> have high average risk "
                f"({val_seg_3:.1%} - {val_seg_2:.1%}).<br>"
                f"Other segments have lower risk (below 20%), with <b>{lowest_seg}</b> the lowest ({val_seg_last:.1%}).<br>"
                f"<i>=> <b>market_segment</b> is an important variable for predicting cancellation probability.</i>"
            )
            plots.append((fname, title, comment))

            # --- 5. LEAD TIME KDE ---
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=self.df, x='lead_time', hue='is_canceled', fill=True, common_norm=False, palette="husl")
            plt.xlim(0, 400)
            fname, title = self.save_plot("Lead Time Distribution: Canceled vs Not Canceled", "lead_time_kde")

            comment = "The orange curve (Canceled) is higher and extends farther to the right than the blue curve. Guests who book very early (Lead Time > 100) have a much higher cancellation probability."
            plots.append((fname, title, comment))

            # --- 6. DEPOSIT TYPE VS CANCEL ---
            plt.figure(figsize=(8, 6))
            sns.countplot(data=self.df, x='deposit_type', hue='is_canceled', palette="Set2")
            plt.title("Booking Cancellation by Deposit Type")
            fname, title = self.save_plot("Relationship: deposit_type and is_canceled", "deposit_cancel_countplot")

            # Calculate metrics.
            non_refund_cancel = self.df[self.df['deposit_type']=='Non Refund']['is_canceled_int'].mean()

            comment_dep = f"""
            <b>Comments and Insights:</b>
            <ul>
                <li><b>Most common:</b> Most customers choose <b>No Deposit</b>.</li>
                <li><b>Non Refund paradox:</b> The "Non Refund" group has an extremely high cancellation rate (<b>{non_refund_cancel:.1%}</b>), which goes against normal expectations.</li>
                <li><b>Reason:</b> In this dataset, the Non Refund group mainly contains bulk reservations from <b>Groups</b> or <b>Agents</b> in Portugal, which were later canceled in batches or due to payment issues.</li>
            </ul>
            """
            plots.append((fname, title, comment_dep))

            # --- 7. TOTAL_OF_SPECIAL_REQUESTS VS IS_CANCEL ---
            plt.figure(figsize=(8, 6))
            sns.countplot(data=self.df, x='total_of_special_requests', hue='is_canceled', palette="Paired")
            plt.title("Booking Cancellation by Special Requests")
            fname, title = self.save_plot("Relationship: total_of_special_requests vs is_canceled", "requests_cancel_countplot")

            comment_req = """
            <b>Comments:</b><br>
            - Customers with no special requests (0 requests) have the highest cancellation rate.<br>
            - As the number of requests increases (1, 2, 3, ...), the cancellation rate decreases noticeably.<br>
            <b>=> This suggests that customers who communicate or interact more are usually more serious about the trip.</b>
            """
            plots.append((fname, title, comment_req))

            # --- 8. CORRELATION HEATMAP ---
            plt.figure(figsize=(10, 8))
            # Keep only important numeric columns.
            numeric_cols_heat = ['is_canceled', 'lead_time', 'arrival_date_year', 'arrival_date_week_number',
                            'adults', 'children', 'babies', 'is_repeated_guest',
                            'previous_cancellations', 'booking_changes', 'days_in_waiting_list', 'adr',
                            'required_car_parking_spaces', 'total_of_special_requests']
            valid_cols_heat = [c for c in numeric_cols_heat if c in self.df.columns]
            numeric_df = self.df[valid_cols_heat]
            # Add the numeric target column back.
            numeric_df['is_canceled'] = self.df['is_canceled_int']

            mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
            sns.heatmap(numeric_df.corr(), mask=mask, cmap="coolwarm", center=0, vmax=1, vmin=-1, linewidths=0.5)
            plt.title("Correlation Matrix")
            fname, title = self.save_plot("Correlation Matrix (Correlation Matrix)", "heatmap_corr")

            comment = "Variables <b>lead_time</b> and <b>previous_cancellations</b> have positive correlation with cancellations. In contrast, <b>total_of_special_requests</b> and <b>parking_spaces</b> have negative correlation: more requests generally mean fewer cancellations."
            plots.append((fname, title, comment))


            # --- COMBINE HTML ---
            self.create_html_report(plots)

        except Exception as e:
            logging.error(f"Error while drawing charts: {str(e)}")
            raise e

    def create_html_report(self, plots):
        logging.info("Creating HTML report...")
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
            <h1>COMPREHENSIVE DATA ANALYSIS REPORT</h1>
            {self.generate_overview_section()}
            <h2 style="color: {main_color}; border-left: 5px solid {main_color};">2. NUMERIC DESCRIPTIVE STATISTICS</h2>
            {self.generate_numeric_stats().split('</h2>')[1]}
            {self.generate_categorical_stats()}
            <hr>
            <h2 style="color: {main_color}; border-left: 5px solid {main_color};">4. VISUALIZATION CHARTS AND INSIGHTS</h2>
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
        html += """<div style="text-align:center; margin-top:50px; color:#888;">Report generated automatically.</div></div></body></html>"""

        report_path = os.path.join(REPORT_DIR, "FULL_EDA_REPORT.html")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html)
            logging.info(f"Full report exported to: {report_path}")
            print(f"\nFull report exported to: {report_path}")
        except Exception as e:
            logging.error(f"Error while writing report file: {str(e)}")

    def generate_toc(self, plots):
        """Create the table of contents."""
        html = "<h3>CHART TABLE OF CONTENTS</h3><ul>"
        for i, (_, title, _) in enumerate(plots, 1):
            html += f"<li><a href='#chart_{i}' style='text-decoration:none; color:#2980b9;'>{i}. {title}</a></li>"
        html += "</ul>"
        return html

# --- RUN PROGRAM ---
if __name__ == "__main__":
    logging.info(f"Script started at: {datetime.datetime.now()}")
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            eda = EDAHotelBooking(df)
            eda.run_all()
        except Exception as e:
            logging.critical(f"ERROR: {str(e)}")
            print(f"An error occurred: {e}")
    else:
        msg = f"Error: Data file not found at {DATA_PATH}"
        logging.error(msg)
        print(msg)
    logging.info(f"Script finished at: {datetime.datetime.now()}")
