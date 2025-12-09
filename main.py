import os
import pandas as pd
import numpy as np
import logging
import argparse
import configparser

# CẤU HÌNH LOGGING & MÔI TRƯỜNG
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True) 
os.makedirs("data/processed", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # Ghi vào file
        logging.FileHandler("activity.log", mode='w', encoding='utf-8'),
        # Ghi ra màn hình console
        logging.StreamHandler()
    ],
    force=True 
)
logger = logging.getLogger(__name__)
print(f"Log file đang được ghi tại: {os.path.abspath('activity.log')}")

# Import module tự viết
from src.preprocessing import DataPreprocessor
from src.model import ModelTrainer 

# HÀM FEATURE ENGINEERING ĐẶC THÙ CHO DỮ LIỆU KHÁCH SẠN
def hotel_feature_engineering(df):
    logger.info("*** Đang thực hiện Feature Engineering đặc thù...")
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

# MAIN PROGRAM 
def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing Pipeline")
    parser.add_argument('--config', type=str, default='config.ini', help='Path to configuration file')
    args = parser.parse_args()

    # Đọc config
    config = configparser.ConfigParser()
    config.read(args.config, encoding='utf-8')

    try:
        INPUT_PATH = config['PREPROCESSING']['inputpath']
        TARGET_COL = config['DATA']['target']
    except KeyError as e:
        logger.error(f"Lỗi Config: Thiếu key {e}. Kiểm tra lại file config.ini")
        return

    # Kiểm tra file input
    if not os.path.exists(INPUT_PATH):
        logger.error(f"*** Không tìm thấy file: {INPUT_PATH}")
        return
    
    temp_prep = DataPreprocessor(target_col=TARGET_COL)

    # ------ 1. LOAD & CLEAN ------
    logger.info("Load dữ liệu và loại bỏ cột rác...")
    df = pd.read_csv(INPUT_PATH, dtype={'agent': 'object', 'company': 'object'})
    
    # Chuẩn hóa tên cột
    df = temp_prep.clean_column_names(df) 
    logger.info("Đã chuẩn hóa tên cột (Clean Column Names).")
    
    # Loại bỏ dòng trùng lặp
    logger.info("Loại bỏ dòng trùng lặp...")
    df = temp_prep.remove_duplicates(df)
    logger.info(f"   -> Shape sau khi loại bỏ trùng lặp: {df.shape}")

    # Drop cột không cần thiết
    logger.info("Loại bỏ cột không cần thiết...")
    cols_drop = ['reservation_status', 'reservation_status_date', 'assigned_room_type', 'arrival_date_year', 'agent', 'company']
    existing = [c for c in cols_drop if c in df.columns]
    df.drop(columns=existing, inplace=True)
    logger.info(f"   -> Đã xóa {len(existing)} cột.")
    
    # ------ 2. FEATURE ENGINEERING AND FILL MISSING VALUES ------
    # Create new features
    logger.info(" Feature Engineering (Custom)...")
    df = hotel_feature_engineering(df)
    temp_prep._detect_column_types(df)
    # Fill missing values
    logger.info("Điền giá trị thiếu (Fill Missing Values)...")
    df = temp_prep.fill_missing(df, num_strategy='median', cat_strategy='mode', mode='fit_transform')
    logger.info(f"   -> Shape sau khi xử lý sơ bộ: {df.shape}")
    
    # ------ 3. SPLIT DATA ------
    logger.info("3. Chia dữ liệu Train / Test ...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    trainer_tool = ModelTrainer(config_path=args.config)
    
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = trainer_tool.split_data(
        X, y, test_size=0.2, stratify=y
    )
    logger.info(f"   -> Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")

    # ------ 4. PREPROCESSING (OUTLIER, ENCODE, SCALE) ------
    # model
    logger.info("4. Chạy Pipeline Tiền xử lý (Outlier, Encode, Scale)...")
    preprocessor = DataPreprocessor(target_col=TARGET_COL)
    
    logger.info("   -> Đang xử lý tập Train...")
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train_raw, y_train_raw)
    
    logger.info("   -> Đang xử lý tập Test...")
    X_test_processed = preprocessor.transform(X_test_raw)
    
    # Lưu dữ liệu đã xử lý
    logger.info("   -> Lưu dữ liệu đã xử lý vào thư mục data/processed ...")
    preprocessor.save_processed_data(X_train_processed, filename="train_processed.csv", y=y_train_processed)
    preprocessor.save_processed_data(X_test_processed, filename="test_processed.csv", y=y_test_raw)
    
    # Lưu preprocessor để dùng lại
    preprocessor.save_preprocessor("reports/preprocessor.joblib")
    logger.info("HOÀN TẤT PREPROCESSING.")
        
    # PAUSE & CONFIRMATION STEP
    print("\n" + "="*70)
    print("GÕ 'model' ĐỂ TIẾP TỤC CHẠY MODEL (Gõ 'exit' để thoát)")
    print("="*70)
    
    try:
        # ------ 1. Vòng lặp chọn 'model' hoặc 'exit' ------
        while True:
            user_input = input(">> ").strip().lower() 
            
            if user_input == 'exit':
                print("Đã thoát chương trình.")
                return # Dừng hàm main
            
            if user_input == 'model':
                break # Thoát vòng lặp để xuống phần chọn thuật toán
            
            print("Nhập sai! Hãy gõ 'model' để tiếp tục hoặc 'exit' để thoát.")

        # ------ 2. Chọn thuật toán ------
        print("\n" + "-"*50)
        print("CHỌN THUẬT TOÁN: Auto, CatBoost, XGBoost, LightGBM, RandomForest")
        
        # Nhập và xử lý chuỗi
        raw_model_input = input(">> Nhập tên model (Mặc định 'Auto' nếu bỏ trống): ").strip().lower()
        
        # Dictionary ánh xạ từ input người dùng sang tên chuẩn
        model_map = {
            'auto': 'Auto',
            'catboost': 'CatBoost',
            'cat': 'CatBoost',          # Hỗ trợ viết tắt
            'xgboost': 'XGBoost',
            'xgb': 'XGBoost',           # Hỗ trợ viết tắt
            'lightgbm': 'LightGBM',
            'lgbm': 'LightGBM',         # Hỗ trợ viết tắt
            'randomforest': 'RandomForest',
            'rf': 'RandomForest'        # Hỗ trợ viết tắt
        }
        
        # Lấy tên chuẩn, nếu không tìm thấy hoặc để trống thì mặc định là 'Auto'
        model_name = model_map.get(raw_model_input, 'Auto')
        
        print(f"-> Thuật toán được chọn: {model_name}")
        print("-"*50 + "\n")
        
        logger.info(f"*** BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN: {model_name}...")
        
        # ------ 5. MODEL TRAINING & EVALUATION ------
        # Khởi tạo trainer và load data đã xử lý
        trainer = ModelTrainer(config_path=args.config)
        trainer.load_data()
        
        # Logic chạy model
        if model_name == 'Auto':
            trainer.auto_select_model()
            trainer.plot_evaluation_results()
        else:
            trainer.optimize_params(model_name=model_name)
            trainer.train_predict()
            try:
                trainer.get_feature_importance(model_name)
            except: pass
            trainer.save_model()
            
        print("\n*** ĐÃ HOÀN TẤT HUẤN LUYỆN MODEL!")
            
    except KeyboardInterrupt:
        print("\nĐã thoát chương trình.")
    
if __name__ == "__main__":
    main()
