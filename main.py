import os
import pandas as pd
import numpy as np
import time
import logging
import matplotlib.pyplot as plt

# Import các thư viện mô hình Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 

# Import module tự viết
from src.preprocessing import DataPreprocessor
from src.model import ModelTrainer 

# =========================================================
# CẤU HÌNH LOGGING & MÔI TRƯỜNG
# =========================================================
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True) 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reports/training.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True 
)
logger = logging.getLogger(__name__)
plt.switch_backend('Agg') # Backend không GUI

# =========================================================
# HÀM FEATURE ENGINEERING
# =========================================================
def hotel_feature_engineering(df):
    logger.info("🛠️ Đang thực hiện Feature Engineering đặc thù...")
    df = df.copy()

    # 1. Tổng số đêm
    stays_cols = ['stays_in_weekend_nights', 'stays_in_week_nights']
    if set(stays_cols).issubset(df.columns):
        df['total_nights'] = df[stays_cols].sum(axis=1)
        df.drop(columns=stays_cols, inplace=True)

    # 2. Thông tin khách
    guest_cols = ['adults', 'children', 'babies']
    if set(guest_cols).issubset(df.columns):
        df[guest_cols] = df[guest_cols].fillna(0)
        df['total_guests'] = df[guest_cols].sum(axis=1)
        df['has_children'] = (df[['children', 'babies']].sum(axis=1) > 0).astype(int)
        df.drop(columns=guest_cols, inplace=True)

    # 3. Khách nội địa
    if 'country' in df.columns:
        df['is_domestic'] = (df['country'] == 'PRT').astype(int)
        df.drop(columns=['country'], inplace=True)

    # 4. Đổi phòng
    if {'reserved_room_type', 'assigned_room_type'}.issubset(df.columns):
        df['is_room_changed'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)
    
    return df

# =========================================================
# MAIN PROGRAM
# =========================================================
def main():
    INPUT_PATH = "data/raw/hotel_bookings.csv"
    TARGET_COL = 'is_canceled'
    
    if not os.path.exists(INPUT_PATH):
        logger.error(f"❌ Không tìm thấy file: {INPUT_PATH}")
        return
    
    temp_prep = DataPreprocessor(target_col=TARGET_COL)
    # ---------------------------------------------------------
    # 1. LOAD & CLEAN
    # ---------------------------------------------------------
    logger.info("Load dữ liệu và loại bỏ cột rác...")
    df = pd.read_csv(INPUT_PATH, dtype={'agent': 'object', 'company': 'object'})
    
        
    logger.info(" Loại bỏ dòng trùng lặp...")
    df = temp_prep.remove_duplicates(df)
    logger.info(f"   -> Shape sau khi loại bỏ trùng lặp: {df.shape}")
    # ---------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # ---------------------------------------------------------
    logger.info(" Feature Engineering (Custom)...")
    df = hotel_feature_engineering(df)
    temp_prep._detect_column_types(df)
    df = temp_prep.fill_missing(df, num_strategy='median', cat_strategy='mode', mode='fit_transform')
    logger.info(f"   -> Shape sau khi xử lý sơ bộ: {df.shape}")
    
    # Drop cột không cần thiết
    logger.info("3️⃣ Loại bỏ cột không cần thiết...")
    cols_drop = ['reservation_status', 'reservation_status_date', 'assigned_room_type', 'arrival_date_year', 'agent', 'company']
    existing = [c for c in cols_drop if c in df.columns]
    df.drop(columns=existing, inplace=True)
    logger.info(f"   -> Đã xóa {len(existing)} cột.")
    # ---------------------------------------------------------
    # 3. SPLIT DATA
    # ---------------------------------------------------------
    logger.info("3️⃣ Chia dữ liệu Train / Test ...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # 4. PREPROCESSING (OUTLIER, ENCODE, SCALE)
    # ---------------------------------------------------------
    logger.info("4️⃣ Chạy Pipeline Tiền xử lý (Outlier, Encode, Scale)...")
    preprocessor = DataPreprocessor(target_col=TARGET_COL)
    
    logger.info("   -> Đang xử lý tập Train...")
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train_raw, y_train_raw)
    
    logger.info("   -> Đang xử lý tập Test...")
    X_test_processed = preprocessor.transform(X_test_raw)
    
    # 5. Lưu
    logger.info("   -> Lưu dữ liệu đã xử lý vào thư mục data/processed ...")
    preprocessor.save_processed_data(X_train_processed, filename="train_processed.csv", y=y_train_processed)
    preprocessor.save_processed_data(X_test_processed, filename="test_processed.csv", y=y_test_raw)
if __name__ == "__main__":
    main()

