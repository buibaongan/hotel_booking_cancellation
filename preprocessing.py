import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
import joblib
import os

class MyStandardScaler:
    """
    Bộ chuẩn hóa dữ liệu theo công thức Standardization:
    (x - mean) / std.

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

        :param df: DataFrame đầu vào.
        :param columns: Danh sách tên cột cần chuẩn hóa.
        :return: self
        """
        for col in columns:
            self.means[col] = df[col].mean()
            self.stds[col] = df[col].std() if df[col].std() != 0 else 1
        return self
    def transform(self, df, columns):
        """
        Chuẩn hóa dữ liệu dựa trên mean và std đã học từ hàm fit().

        :param df: DataFrame cần chuẩn hóa.
        :param columns: Danh sách cột cần chuẩn hóa.
        :return: DataFrame đã chuẩn hóa.
        """
        df_transformed = df.copy()
        for col in columns:
            if col in self.means:
                df_transformed[col] = (df[col] - self.means[col]) / self.stds[col]
        return df_transformed
    def fit_transform(self, df, columns):
        """
        Gọi fit() rồi transform() cho dữ liệu.

        :param df: DataFrame đầu vào.
        :param columns: Danh sách cột cần chuẩn hóa.
        :return: DataFrame đã chuẩn hóa.
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
        Khởi tạo scaler với dict lưu min và max cho mỗi cột
        """
        self.mins = {}
        self.maxs = {}
    def fit(self, df, columns):
        """
        Học min/max cho từng cột.

        :param df: DataFrame đầu vào.
        :param columns: Danh sách cột cần scale.
        :return: self
        """
        for col in columns:
            self.mins[col] = df[col].min()
            self.maxs[col] = df[col].max() if df[col].max() != df[col].min() else df[col].min() + 1
        return self
    def transform(self, df, columns):
        """
        Chuẩn hóa dữ liệu bằng min/max đã học.

        :param df: DataFrame cần chuẩn hóa.
        :param columns: Danh sách cột.
        :return: DataFrame chuẩn hóa.
        """
        df_transformed = df.copy()
        for col in columns:
            if col in self.mins:
                df_transformed[col] = (df[col] - self.mins[col]) / (self.maxs[col] - self.mins[col])
        return df_transformed
    def fit_transform(self, df, columns):
        """
        Fit và transform dữ liệu.

        :param df: DataFrame đầu vào.
        :param columns: Danh sách cột cần scale.
        :return: DataFrame đã chuẩn hóa.
        """
        self.fit(df, columns)
        return self.transform(df, columns)

class DataPreprocessor:
    """
    Lớp xử lý dữ liệu đầy đủ theo pipeline:
    - Nhận dạng kiểu dữ liệu
    - Xử lý ngày giờ
    - Xóa trùng lặp
    - Điền thiếu
    - Loại outlier (IsolationForest hoặc IQR)
    - Giải quyết mất cân bằng phân loại hiếm
    - Loại bỏ cột tương quan cao
    - Mã hóa categorical (Label, One-hot, Frequency)
    - Scale dữ liệu
    - Làm sạch cuối
    - Lưu dữ liệu và lưu preprocessor

    Phục vụ cho chuẩn hóa dữ liệu trước huấn luyện mô hình ML.
    """
    def __init__(self, target_col=None):
        self.target_col = target_col
        self.df = None
        
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.impute_values = {}
        self.imbalance_map = {}
        self.cols_to_drop_corr = []
        self.encoders = {}
        self.onehot_columns = None
        self.freq_maps = {}
        self.scaler = None
        self.scale_method = 'standard'

    def _detect_column_types(self, df):
        """
        Phân loại cột:
        - numeric
        - categorical
        - datetime

        Dựa vào dtype và thử parse datetime.
        """
        self.numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.target_col in self.numeric_cols: self.numeric_cols.remove(self.target_col)
        
        # Detect datetime_col
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
        Chuyển cột datetime sang các đặc trưng:
        - year
        - month
        - day
        - weekday

        Sau đó xóa cột datetime gốc.

        :param df: DataFrame đầu vào.
        :return: DataFrame đã xử lý.
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
    
    def remove_duplicates(self,df):
        """
        Xóa các dòng duplicate trong DataFrame.

        :param df: DataFrame đầu vào.
        :return: DataFrame đã loại bỏ trùng lặp.
        """
        if df is None:
            return None
        before = len(df)
        df = df.drop_duplicates(keep = 'first').reset_index(drop = True)
        after = len(df)
        
        if after != before:
            print(f"Đã loại bỏ {before - after} dòng trùng lặp")
        else:
            print("Không có dòng trùng lặp nào")
        return df
    
    def fill_missing(self, df, num_strategy="mean", cat_strategy="mode", mode='fit_transform'):
        """
        Điền giá trị thiếu cho cột số và cột phân loại.

        :param num_strategy: mean / median / zero.
        :param cat_strategy: mode / constant.
        :param mode: 'fit_transform' hoặc 'transform'.
        :return: DataFrame sau khi điền thiếu.
        """
        if df is None: return None
        df = df.copy()

        if mode == 'fit_transform':

            for col in self.numeric_cols:
                if col in df.columns:
                    if num_strategy == "mean":
                        val = df[col].mean()
                    elif num_strategy == "median":
                        val = df[col].median()
                    elif num_strategy == "zero":
                        val = 0
                    else:
                        val = df[col].median()
                    self.impute_values[col] = val
            
            for col in self.categorical_cols:
                if col in df.columns:
                    if cat_strategy == "mode":
                        mode_val = df[col].mode()
                        val = mode_val[0] if not mode_val.empty else "Unknown"
                    elif cat_strategy == "constant":
                        val = "Missing"
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
        Gom các giá trị hiếm (< min_freq) thành 'Other' để tránh tạo quá nhiều category.

        :param df: DataFrame đầu vào.
        :param min_freq: Ngưỡng xuất hiện tối thiểu.
        :param mode: 'fit_transform' hoặc 'transform'.
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
        Loại bỏ các cột numeric có tương quan lớn hơn ngưỡng.

        :param df: DataFrame.
        :param threshold: Ngưỡng tương quan.
        :param mode: fit_transform hoặc transform.
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
        Mã hóa biến phân loại theo luật:
        - <=2 giá trị → Label Encoding
        - >15 giá trị → Frequency Encoding
        - còn lại → One-hot Encoding

        :param df: DataFrame đầu vào.
        :param mode: fit_transform hoặc transform.
        :return: DataFrame đã mã hóa.
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
            df = pd.get_dummies(df, columns=cols_onehot, drop_first=False, dtype=int)
        if mode == 'fit_transform':
            self.onehot_columns = df.columns.tolist()
        elif self.onehot_columns:
            df = df.reindex(columns=self.onehot_columns, fill_value=0)
        return df

    def scale_features(self, df, method="standard", mode='fit_transform'):
        """
        Chuẩn hóa dữ liệu numeric:
        - standard (Z-score)
        - minmax (0–1)

        :param df: DataFrame.
        :param method: 'standard' hoặc 'minmax'.
        :param mode: fit_transform hoặc transform.
        :return: DataFrame đã scale.
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
        Loại bỏ outliers bằng Isolation Forest với contamination=0.05.

        :param df: DataFrame đầu vào.
        :return: DataFrame đã lọc outliers.
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
        Loại bỏ outliers theo quy tắc IQR:
        (Q1 - 1.5*IQR, Q3 + 1.5*IQR)

        :param df: DataFrame.
        :return: DataFrame đã lọc.
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
        """Đảm bảo không còn NaN hay Inf trước khi đưa vào model"""
        if df is None: return None
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        return df

    # --- PIPELINE ---
    def fit_transform(self, X, y=None):
        """
        Chạy toàn bộ pipeline xử lý dữ liệu cho tập huấn luyện:
        - Detect type
        - Date features
        - Remove duplicates
        - Fill missing
        - Remove outliers
        - Handle imbalance
        - Remove high correlation
        - Encode categorical
        - Scale data
        - Cleanup cuối

        :param X: DataFrame features.
        :param y: Series target (tùy chọn).
        :return: (X_clean, y_clean) hoặc DataFrame clean.
        """
        if y is not None:
            X, y = X.reset_index(drop=True), y.reset_index(drop=True)
            self.df = pd.concat([X, y], axis=1)
        else:
            self.df = X.copy()

        # 1. Features & Detect
        self.df = self.create_datetime_features(self.df)
        self._detect_column_types(self.df) 
        
        # 2. Xóa duplicates
        self.df = self.remove_duplicates(self.df)

        # 3. Missing
        self.df = self.fill_missing(self.df, num_strategy='mean', cat_strategy='mode', mode='fit_transform')

        # 4. Outlier
        skew = self.df[self.numeric_cols].skew().mean() if self.numeric_cols else 0
        if abs(skew) > 1:
            print(f"   -> Skew={skew:.2f}. Dùng IsolationForest.")
            self.df = self.remove_outliers_isolation_forest(self.df)
            self.scale_method = 'minmax'
        else:
            print(f"   -> Skew={skew:.2f}. Dùng IQR.")
            self.df = self.remove_outliers_iqr(self.df)
            self.scale_method = 'standard'

        # 4. Others
        self.df = self.handle_imbalance(self.df, mode='fit_transform')
        self.df = self.remove_high_corr(self.df, mode='fit_transform')
        self.df = self.encode_categorical(self.df, mode='fit_transform')
        
        self.numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if self.target_col in self.numeric_cols: self.numeric_cols.remove(self.target_col)
        
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
        Áp dụng pipeline đã học lên tập test (không học lại).
        
        Các bước giống fit_transform nhưng dùng tham số đã lưu.

        :param X: DataFrame test.
        :return: DataFrame đã xử lý.
        """
        df = X.copy()
        df = self.create_datetime_features(df)
        
        df = self.fill_missing(df, mode='transform')
        
        df = self.handle_imbalance(df, mode='transform')
        df = self.remove_high_corr(df, mode='transform')
        df = self.encode_categorical(df, mode='transform')
        
        method = getattr(self, 'scale_method', 'standard')
        df = self.scale_features(df, method=method, mode='transform')
        
        if self.target_col and self.target_col in df.columns:
            df.drop(columns=[self.target_col], inplace=True)
            
        df = self._final_cleanup(df)
        return df

    def save_processed_data(self, df, filename, output_dir="data/processed", y=None):
        """
        Lưu DataFrame ra file CSV.
        :param df: DataFrame chứa Features (X).
        :param filename: Tên file (vd: 'train_clean.csv').
        :param output_dir: Thư mục lưu (Mặc định: 'data/processed').
        :param y: (Optional) Series chứa Target để ghép vào trước khi lưu.
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
        Lưu toàn bộ Preprocessor (tất cả tham số đã học) bằng joblib.

        :param path: Đường dẫn file .joblib
        """
        joblib.dump(self, path)
        print(f"Đã lưu Preprocessor tại {path}")
