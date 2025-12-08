import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import joblib
import os

class MyStandardScaler:
    """
    Bộ chuẩn hóa dữ liệu theo phương pháp Standardization (Z-score).
    Công thức: z = (x - mean) / std.
    Lưu thủ công giá trị mean và std cho từng cột để đảm bảo
    quá trình transform trên tập test sử dụng lại đúng tham số đã học.
    """
    def __init__(self):
        """
        Khởi tạo scaler với dict lưu mean và std cho từng cột.
        """
        self.means = {}
        self.stds = {}

    def fit(self, df, columns):
        """
        Học mean và std cho từng cột số.

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu huấn luyện.
            columns (list): Danh sách tên cột cần chuẩn hóa.
        Returns:
            self: Trả về chính object scaler sau khi đã học tham số.
        """
        for col in columns:
            self.means[col] = df[col].mean()
            self.stds[col] = df[col].std() if df[col].std() != 0 else 1
        return self

    def transform(self, df, columns):
        """
        Chuẩn hóa dữ liệu dựa trên mean và std đã học từ hàm fit().

        Args:
            df (pd.DataFrame): DataFrame cần chuẩn hóa.
            columns (list): Danh sách cột cần chuẩn hóa.
        Returns:
            pd.DataFrame: DataFrame chứa các cột đã được chuẩn hóa.
        """
        df_transformed = df.copy()
        for col in columns:
            if col in self.means:
                df_transformed[col] = (df[col] - self.means[col]) / self.stds[col]
        return df_transformed

    def fit_transform(self, df, columns):
        """
        Thực hiện fit() sau đó transform() dữ liệu.

        Args:
            df (pd.DataFrame): DataFrame đầu vào.
            columns (list): Danh sách cột cần chuẩn hóa.
        Returns:
            pd.DataFrame: DataFrame đã được chuẩn hóa.
        """
        self.fit(df, columns)
        return self.transform(df, columns)

class MyMinMaxScaler:
    """
    Bộ chuẩn hóa Min-Max:
    (x - min) / (max - min)
    Tự lưu giá trị min/max cho từng cột để dùng lại cho tập test.
    """
    def __init__(self):
        """
        Khởi tạo scaler với dict lưu min và max cho mỗi cột.
        """
        self.mins = {}
        self.maxs = {}

    def fit(self, df, columns):
        """
        Học min và max cho từng cột.

        Args:
            df (pd.DataFrame): DataFrame đầu vào.
            columns (list): Danh sách cột cần scale.
        Returns:
            self: Trả về chính object scaler sau khi đã học tham số.
        """
        for col in columns:
            self.mins[col] = df[col].min()
            val_range = df[col].max() - df[col].min()
            self.maxs[col] = df[col].max() if val_range != 0 else df[col].min() + 1
        return self

    def transform(self, df, columns):
        """
        Chuẩn hóa dữ liệu bằng min/max đã học.
        Args:
            df (pd.DataFrame): DataFrame cần chuẩn hóa.
            columns (list): Danh sách cột.
        Returns:
            pd.DataFrame: DataFrame đã chuẩn hóa về khoảng [0, 1].
        """
        df_transformed = df.copy()
        for col in columns:
            if col in self.mins:
                denominator = self.maxs[col] - self.mins[col]
                if denominator == 0: denominator = 1
                df_transformed[col] = (df[col] - self.mins[col]) / denominator
        return df_transformed

    def fit_transform(self, df, columns):
        """
        Thực hiện fit() sau đó transform() dữ liệu.

        Args:
            df (pd.DataFrame): DataFrame đầu vào.
            columns (list): Danh sách cột cần scale.

        Returns:
            pd.DataFrame: DataFrame đã được chuẩn hóa.
        """
        self.fit(df, columns)
        return self.transform(df, columns)

class DataPreprocessor:
    """
    Class quản lý toàn bộ quy trình tiền xử lý dữ liệu (End-to-End Pipeline).
    
    Quy trình bao gồm:
    1. Làm sạch tên cột.
    2. Xử lý dữ liệu thời gian (Datetime).
    3. Xử lý giá trị thiếu (Missing values).
    4. Xử lý dữ liệu mất cân bằng (Imbalance categorical).
    5. Loại bỏ cột tương quan cao (High Correlation).
    6. Mã hóa dữ liệu phân loại (Encoding: Label, Frequency, One-Hot).
    7. Chuẩn hóa dữ liệu số (Scaling).
    8. Loại bỏ ngoại lai (Outliers).
    9. Làm sạch cuối cùng (Đảm bảo không còn NaN hoặc giá trị vô cực (inf))
    10. Lưu dữ liệu và lưu preprocessor

    Phục vụ cho chuẩn hóa dữ liệu trước huấn luyện mô hình ML.
    """
    def __init__(self, target_col=None):
        self.target_col = target_col
        self.df = None
        
        # Các thuộc tính lưu trữ metadata của dữ liệu
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        
        # Các thuộc tính lưu trữ tham số đã học (để dùng cho tập test)
        self.impute_values = {}
        self.imbalance_map = {}
        self.cols_to_drop_corr = []
        self.encoders = {}
        self.onehot_columns = None
        self.freq_maps = {}
        self.scaler = None
        self.scale_method = 'standard'

    def clean_column_names(self, df):
        """
        Chuẩn hóa tên cột: Chuyển về dạng snake_case, loại bỏ ký tự đặc biệt.
        Ví dụ: "Customer Type" -> "customer_type", "Transient-Party" -> "Transient_Party".
        
        Args:
            df (pd.DataFrame): DataFrame đầu vào.
        Returns:
            pd.DataFrame: DataFrame với tên cột đã được làm sạch.
        """
        if df is None: return None
        # Regex: Chỉ giữ lại chữ cái, số và dấu gạch dưới. 
        # Các ký tự khác (khoảng trắng, dấu -, /, ...) sẽ bị thay thế bằng '_'
        clean_func = lambda name: re.sub(r'[^A-Za-z0-9_]+', '_', name)
        df = df.rename(columns=clean_func)
        return df

    def _detect_column_types(self, df):
        """
        Hàm nội bộ để phân loại các cột thành numeric, categorical hoặc datetime.
        Kết quả được lưu vào các thuộc tính self.numeric_cols, self.categorical_cols.

        Args:
            df (pd.DataFrame): DataFrame cần phân tích.
        """
        self.numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.target_col in self.numeric_cols: 
            self.numeric_cols.remove(self.target_col)
        
        # Thử parse các cột object sang datetime nếu có thể
        if not self.datetime_cols:
            for col in df.columns:
                if col in self.numeric_cols or col == self.target_col: continue
                if pd.api.types.is_numeric_dtype(df[col]): continue
                try:
                    pd.to_datetime(df[col].dropna().head(20), errors='raise')
                    self.datetime_cols.append(col)
                except: pass

    def create_datetime_features(self, df):
        """
        Chuyển cột datetime sang các đặc trưng số: year, month, day, weekday.
        Sau đó xóa cột datetime gốc.

        Args:
            df (pd.DataFrame): DataFrame đầu vào.
        Returns:
            pd.DataFrame: DataFrame đã có thêm các feature thời gian mới.
        """
        if df is None: return None
        df = df.copy()
        if not self.datetime_cols:
             self._detect_column_types(df)

        for col in self.datetime_cols:
            if col not in df.columns: continue
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = pd.Categorical(df[col].dt.month, categories=range(1,13))
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_weekday"] = df[col].dt.weekday
                df.drop(columns=[col], inplace=True)
            except: pass
        return df
    
    def remove_duplicates(self, df):
        """
        Loại bỏ các dòng dữ liệu bị trùng lặp hoàn toàn.

        Args:
            df (pd.DataFrame): DataFrame đầu vào.
        Returns:
            pd.DataFrame: DataFrame đã loại bỏ các dòng trùng lặp.
        """
        if df is None: return None
        before = len(df)
        df = df.drop_duplicates(keep='first').reset_index(drop=True)
        after = len(df)
        if after != before:
            print(f"Đã loại bỏ {before - after} dòng trùng lặp")
        return df
    
    def fill_missing(self, df, num_strategy="mean", cat_strategy="mode", mode='fit_transform'):
        """
        Điền giá trị bị thiếu (Missing Values Imputation).
        
        Args:
            df (pd.DataFrame): DataFrame đầu vào.
            num_strategy: Mặc định 'mean'.
            cat_strategy: Mặc định 'mode'.
            mode: 'fit_transform' (học giá trị điền) hoặc 'transform' (dùng giá trị đã học). Mặc định 'fit_transform'.
        Returns:
            pd.DataFrame: DataFrame đã được điền dữ liệu thiếu.
        """
        if df is None: return None
        df = df.copy()

        if mode == 'fit_transform':
            for col in self.numeric_cols:
                if col in df.columns:
                    if num_strategy == "mean": val = df[col].mean()
                    elif num_strategy == "median": val = df[col].median()
                    else: val = 0
                    self.impute_values[col] = val
            
            for col in self.categorical_cols:
                if col in df.columns:
                    if cat_strategy == "mode":
                        mode_val = df[col].mode()
                        val = mode_val[0] if not mode_val.empty else "Unknown"
                    else:
                        val = "Unknown"
                    self.impute_values[col] = val

        for col, val in self.impute_values.items():
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(val)
        
        df = df.fillna(0)
        return df

    def handle_imbalance(self, df, min_freq=0.05, mode='fit_transform'):
        """
        Gom các giá trị hiếm (< min_freq) thành nhóm 'Other' để giảm chiều dữ liệu.

        Args:
            df (pd.DataFrame): DataFrame đầu vào.
            min_freq (float, optional): Ngưỡng tần suất tối thiểu (0-1). Các giá trị dưới ngưỡng này bị coi là hiếm. Mặc định 0.05.
            mode (str, optional): 'fit_transform' hoặc 'transform'.
        Returns:
            pd.DataFrame: DataFrame sau khi đã xử lý mất cân bằng.
        """
        if df is None: return None
        df = df.copy()
        cols_check = [c for c in self.categorical_cols if c in df.columns]
        
        if mode == 'fit_transform':
            for col in cols_check:
                freq = df[col].value_counts(normalize=True)
                keep = freq[freq >= min_freq].index.tolist()
                self.imbalance_map[col] = keep
                df[col] = df[col].apply(lambda x: x if x in keep else "Other")
        else:
            for col in cols_check:
                if col in self.imbalance_map:
                    keep = self.imbalance_map[col]
                    df[col] = df[col].apply(lambda x: x if x in keep else "Other")
        return df

    def remove_high_corr(self, df, threshold=0.9, mode='fit_transform'):
        """
        Loại bỏ các cột số có độ tương quan (correlation) quá cao với nhau (đa cộng tuyến).
        Giữ lại 1 cột, xóa các cột còn lại nếu corr > threshold.

        Args:
            df (pd.DataFrame): DataFrame đầu vào.
            threshold (float, optional): Ngưỡng tương quan để loại bỏ. Mặc định 0.9.
            mode (str, optional): 'fit_transform' hoặc 'transform'.
        Returns:
            pd.DataFrame: DataFrame đã loại bỏ các cột tương quan cao.
        """
        if df is None: return None
        df = df.copy()
        if mode == 'fit_transform':
            num_df = df.select_dtypes(include=np.number)
            if not num_df.empty:
                corr = num_df.corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                self.cols_to_drop_corr = [c for c in upper.columns if any(upper[c] > threshold)]
                if self.cols_to_drop_corr:
                    print(f"   -> Loại bỏ cột tương quan cao: {self.cols_to_drop_corr}")
                    df.drop(columns=self.cols_to_drop_corr, inplace=True)
                    self.numeric_cols = [c for c in self.numeric_cols if c not in self.cols_to_drop_corr]
        else:
            if self.cols_to_drop_corr:
                exist = [c for c in self.cols_to_drop_corr if c in df.columns]
                if exist: df.drop(columns=exist, inplace=True)
        return df

    def encode_categorical(self, df, mode='fit_transform'):
        """
        Mã hóa dữ liệu phân loại (Encoding).
        - <= 2 giá trị unique: Label Encoding (0, 1).
        - > 15 giá trị unique: Frequency Encoding (thay bằng tần suất xuất hiện).
        - Còn lại: One-Hot Encoding.
        
        Tự động gọi `clean_column_names` sau khi One-Hot để sửa tên cột có chứa ký tự đặc biệt.

        Args:
            df (pd.DataFrame): DataFrame đầu vào.
            mode (str, optional): 'fit_transform' hoặc 'transform'.
        Returns:
            pd.DataFrame: DataFrame đã được mã hóa dữ liệu.
        """
        if df is None: return None
        df = df.copy()
        cols_to_encode = [c for c in self.categorical_cols if c in df.columns]
        cols_onehot = []
        
        for col in cols_to_encode:
            n_unique = df[col].nunique()
            if n_unique <= 2:
                if mode == 'fit_transform':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
                elif col in self.encoders:
                    le = self.encoders[col]
                    df[col] = df[col].astype(str).map(lambda s: int(le.transform([s])[0]) if s in le.classes_ else 0)
            elif n_unique > 15:
                if mode == 'fit_transform':
                    freq = df[col].value_counts(normalize=True)
                    self.freq_maps[col] = freq
                    df[col] = df[col].map(freq).fillna(0)
                elif col in self.freq_maps:
                    df[col] = df[col].map(self.freq_maps[col]).fillna(0)
            else:
                cols_onehot.append(col)

        if cols_onehot:
            # Sinh ra cột chứa ký tự lạ (VD: Transient-Party)
            df = pd.get_dummies(df, columns=cols_onehot, drop_first=False, dtype=int)
            
            # --- OPTIMIZATION: Làm sạch tên cột ngay lập tức ---
            df = self.clean_column_names(df)

        if mode == 'fit_transform':
            self.onehot_columns = df.columns.tolist()
        elif self.onehot_columns:
            # Reindex để đảm bảo thứ tự cột và số lượng cột khớp với lúc train
            df = df.reindex(columns=self.onehot_columns, fill_value=0)
            
        return df

    def scale_features(self, df, method="standard", mode='fit_transform'):
        """
        Chuẩn hóa dữ liệu số (Scaling) bằng StandardScaler hoặc MinMaxScaler.
        
        Args:
            df (pd.DataFrame): DataFrame đầu vào.
            method (str, optional): 'standard' (StandardScaler) hoặc 'minmax' (MinMaxScaler). Mặc định 'standard'.
            mode (str, optional): 'fit_transform' hoặc 'transform'.
        Returns:
            pd.DataFrame: DataFrame chứa các cột số đã được chuẩn hóa.
        """
        if df is None: return None
        cols_scale = [c for c in self.numeric_cols if c in df.columns]
        if not cols_scale: return df
        
        if mode == 'fit_transform':
            self.scaler = MyStandardScaler() if method == "standard" else MyMinMaxScaler()
            df = self.scaler.fit_transform(df, cols_scale)
        elif self.scaler:
            df = self.scaler.transform(df, cols_scale)
        return df

    def remove_outliers_isolation_forest(self, df):
        """
        Loại bỏ ngoại lai sử dụng thuật toán Isolation Forest với contamination=0.05.
        Args:
            df (pd.DataFrame): DataFrame đầu vào.
        Returns:
            pd.DataFrame: DataFrame đã loại bỏ các dòng ngoại lai.
        """
        if not self.numeric_cols or df is None: return df
        cols_check = [c for c in self.numeric_cols if c in df.columns]
        if not cols_check: return df
        X = df[cols_check].fillna(0)
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(X)
        mask = preds == 1
        print(f"   -> IsoForest: Loại {len(df) - mask.sum()} dòng ngoại lai.")
        return df[mask].reset_index(drop=True)

    def remove_outliers_iqr(self, df):
        """
        Loại bỏ ngoại lai sử dụng phương pháp IQR (Interquartile Range).
        Giữ lại giá trị trong khoảng [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

        Args:
            df (pd.DataFrame): DataFrame đầu vào.
        Returns:
            pd.DataFrame: DataFrame đã loại bỏ các dòng ngoại lai.
        """
        if not self.numeric_cols or df is None: return df
        cols_check = [c for c in self.numeric_cols if c in df.columns]
        mask = np.ones(len(df), dtype=bool)
        for col in cols_check:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            mask &= (df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)
        print(f"   -> IQR: Loại {len(df) - mask.sum()} dòng ngoại lai.")
        return df[mask].reset_index(drop=True)

    def _final_cleanup(self, df):
        """
        Bước làm sạch cuối cùng: Đảm bảo không còn NaN hoặc giá trị vô cực (inf)
        trước khi đưa vào mô hình huấn luyện.

        Args:
            df (pd.DataFrame): DataFrame đầu vào.
        Returns:
            pd.DataFrame: DataFrame đã được làm sạch triệt để.
        """
        if df is None: return None
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        return df

    def fit_transform(self, X, y=None):
        """
        PIPELINE CHO TẬP HUẤN LUYỆN (TRAINING SET).
        Thực hiện toàn bộ quy trình xử lý và HỌC các tham số (mean, mode, min, max...).
        
        Args:
            X (pd.DataFrame): DataFrame chứa các Features.
            y (pd.Series, optional): Series chứa biến mục tiêu (Target).
        Returns:
            tuple hoặc pd.DataFrame: 
                - Nếu có y: Trả về (X_clean, y_clean).
                - Nếu không có y: Trả về df_clean.
        """
        # 1: Clean tên cột ban đầu
        X = self.clean_column_names(X)
        
        if y is not None:
            X, y = X.reset_index(drop=True), y.reset_index(drop=True)
            self.df = pd.concat([X, y], axis=1)
        else:
            self.df = X.copy()

        # 2. Features & Detect
        self.df = self.create_datetime_features(self.df)
        self._detect_column_types(self.df) 
        # 3. Xóa duplicates
        self.df = self.remove_duplicates(self.df)
        # 4. Fill missing
        self.df = self.fill_missing(self.df, num_strategy='mean', cat_strategy='mode', mode='fit_transform')
        # 5. Outliers
        # Tự động chọn phương pháp xử lý outlier dựa trên độ lệch (skewness)
        skew = self.df[self.numeric_cols].skew().mean() if self.numeric_cols else 0
        if abs(skew) > 1:
            print(f"   -> Skew={skew:.2f}. Dùng IsolationForest.")
            self.df = self.remove_outliers_isolation_forest(self.df)
            self.scale_method = 'minmax'
        else:
            print(f"   -> Skew={skew:.2f}. Dùng IQR.")
            self.df = self.remove_outliers_iqr(self.df)
            self.scale_method = 'standard'

        # 5. Others
        self.df = self.handle_imbalance(self.df, mode='fit_transform')
        self.df = self.remove_high_corr(self.df, mode='fit_transform') 
        self.df = self.encode_categorical(self.df, mode='fit_transform')
        
        # 6. Cập nhật lại danh sách cột số sau khi encode
        self.numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if self.target_col in self.numeric_cols: 
            self.numeric_cols.remove(self.target_col)
        
        self.df = self.scale_features(self.df, method=self.scale_method, mode='fit_transform')
        
        # 5. Clean & split
        self.df = self._final_cleanup(self.df)
        
        if y is not None and self.target_col in self.df.columns:
            y_clean = self.df[self.target_col]
            X_clean = self.df.drop(columns=[self.target_col])
            return X_clean, y_clean
        return self.df

    def transform(self, X):
        """
        PIPELINE CHO TẬP KIỂM THỬ (TEST SET) HOẶC DỮ LIỆU MỚI.
        Áp dụng các tham số đã học từ fit_transform, KHÔNG học lại.
        
        Args:
            X (pd.DataFrame): DataFrame dữ liệu đầu vào (Features).
        Returns:
            pd.DataFrame: DataFrame đã được xử lý hoàn chỉnh.
        """
        df = X.copy()
        
        # Clean tên cột input
        df = self.clean_column_names(df)
        
        df = self.create_datetime_features(df)
        df = self.fill_missing(df, mode='transform')
        df = self.handle_imbalance(df, mode='transform')
        df = self.remove_high_corr(df, mode='transform')
        
        # Encode (đã bao gồm clean_column_names bên trong)
        df = self.encode_categorical(df, mode='transform')
        
        method = getattr(self, 'scale_method', 'standard')
        df = self.scale_features(df, method=method, mode='transform')
        
        if self.target_col and self.target_col in df.columns:
            df.drop(columns=[self.target_col], inplace=True)
            
        df = self._final_cleanup(df)
        return df

    def save_processed_data(self, df, filename, output_dir="data/processed", y=None):
        """
        Lưu DataFrame đã xử lý ra file CSV.

        Args:
            df (pd.DataFrame): DataFrame chứa Features (X).
            filename (str): Tên file lưu trữ (ví dụ: 'train_processed.csv').
            output_dir (str, optional): Thư mục đầu ra. Mặc định 'data/processed'.
            y (pd.Series, optional): Series chứa biến mục tiêu (Target) để ghép vào nếu cần.

        Returns:
            None: Hàm thực hiện ghi file và in ra log.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            if y is not None:
                df = df.reset_index(drop=True)
                y = y.reset_index(drop=True)
                df_to_save = pd.concat([df, y], axis=1)
            else:
                df_to_save = df
            file_path = os.path.join(output_dir, filename)
            df_to_save.to_csv(file_path, index=False)
            print(f"Đã lưu dữ liệu vào: {file_path} (Shape: {df_to_save.shape})")
        except Exception as e:
            print(f"*** Lỗi khi lưu dữ liệu: {e}")
            
    def save_preprocessor(self, path):
        """
        Lưu toàn bộ object DataPreprocessor (bao gồm các tham số đã học) ra file .joblib.
        Giúp tái sử dụng pipeline cho việc dự đoán sau này (Inference).
        Args:
            path (str): Đường dẫn file lưu trữ (ví dụ: 'reports/preprocessor.joblib').
        Returns:
            None: Hàm thực hiện lưu file.
        """
        joblib.dump(self, path)
        print(f"Đã lưu Preprocessor tại {path}")
