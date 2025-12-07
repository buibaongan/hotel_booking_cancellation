import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
import joblib
import os

# =========================================================
# 1. CUSTOM SCALERS
# =========================================================
class MyStandardScaler:
    def __init__(self):
        self.means = {}
        self.stds = {}
    def fit(self, df, columns):
        for col in columns:
            self.means[col] = df[col].mean()
            self.stds[col] = df[col].std() if df[col].std() != 0 else 1
        return self
    def transform(self, df, columns):
        df_transformed = df.copy()
        for col in columns:
            if col in self.means:
                df_transformed[col] = (df[col] - self.means[col]) / self.stds[col]
        return df_transformed
    def fit_transform(self, df, columns):
        self.fit(df, columns)
        return self.transform(df, columns)

class MyMinMaxScaler:
    def __init__(self):
        self.mins = {}
        self.maxs = {}
    def fit(self, df, columns):
        for col in columns:
            self.mins[col] = df[col].min()
            self.maxs[col] = df[col].max() if df[col].max() != df[col].min() else df[col].min() + 1
        return self
    def transform(self, df, columns):
        df_transformed = df.copy()
        for col in columns:
            if col in self.mins:
                df_transformed[col] = (df[col] - self.mins[col]) / (self.maxs[col] - self.mins[col])
        return df_transformed
    def fit_transform(self, df, columns):
        self.fit(df, columns)
        return self.transform(df, columns)

# =========================================================
# 2. DATA PREPROCESSOR CLASS (Phiên bản "Sạch sẽ tuyệt đối")
# =========================================================
class DataPreprocessor:
    def __init__(self, target_col=None):
        self.target_col = target_col
        self.df = None
        
        # Metadata
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

    # --- Helpers ---
    def _detect_column_types(self, df):
        self.numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.target_col in self.numeric_cols: self.numeric_cols.remove(self.target_col)
        
        # Detect datetime (simple check)
        if not self.datetime_cols:
            for col in df.columns:
                if col in self.numeric_cols or col == self.target_col: continue
                # Skip if mostly numeric
                if pd.api.types.is_numeric_dtype(df[col]): continue
                try:
                    pd.to_datetime(df[col].dropna().head(20), errors='raise')
                    self.datetime_cols.append(col)
                except: pass

    # --- Logic ---
    def create_datetime_features(self, df):
        if df is None: return None
        df = df.copy()
        if not self.datetime_cols:
             self._detect_column_types(df)

        for col in self.datetime_cols:
            if col not in df.columns: continue
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_weekday"] = df[col].dt.weekday
                df.drop(columns=[col], inplace=True)
            except: pass
        return df
    
    def remove_duplicates(self,df):
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
        Điền giá trị thiếu.
        :param num_strategy: 'mean' (mặc định) hoặc 'median' hoặc 'zero'.
        :param cat_strategy: 'mode' (mặc định) hoặc 'constant' (điền chữ 'Missing').
        """
        if df is None: return None
        df = df.copy()

        if mode == 'fit_transform':
            # 1. Học cho cột số
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
            
            # 2. Học cho cột phân loại
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

        # Điền giá trị (Transform)
        for col, val in self.impute_values.items():
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(val)
        
        # Điền 0 cho mọi ô trống còn sót lại (ví dụ các cột mới tạo)
        df = df.fillna(0)
        return df


    def handle_imbalance(self, df, min_freq=0.05, mode='fit_transform'):
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
            df = pd.get_dummies(df, columns=cols_onehot, drop_first=True, dtype=int)
        if mode == 'fit_transform':
            self.onehot_columns = df.columns.tolist()
        elif self.onehot_columns:
            df = df.reindex(columns=self.onehot_columns, fill_value=0)
        return df

    def scale_features(self, df, method="standard", mode='fit_transform'):
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

    # --- HÀM VỆ SINH CUỐI CÙNG ---
    def _final_cleanup(self, df):
        """Đảm bảo không còn NaN hay Inf trước khi đưa vào model"""
        if df is None: return None
        # 1. Fill NaN bằng 0 (cho các cột mới tạo mà fill_missing bỏ sót)
        df = df.fillna(0)
        # 2. Thay thế vô cực (Inf) bằng số cực lớn/nhỏ hoặc 0
        df = df.replace([np.inf, -np.inf], 0)
        return df

    # --- PIPELINE ---
    def fit_transform(self, X, y=None):
        if y is not None:
            X, y = X.reset_index(drop=True), y.reset_index(drop=True)
            self.df = pd.concat([X, y], axis=1)
        else:
            self.df = X.copy()

        # 1. Features & Detect
        self.df = self.create_datetime_features(self.df)
        self._detect_column_types(self.df) # Detect LẠI để cập nhật cột mới
        
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

        # 5. VỆ SINH CUỐI CÙNG & TÁCH
        self.df = self._final_cleanup(self.df)
        
        if y is not None and self.target_col in self.df.columns:
            y_clean = self.df[self.target_col]
            X_clean = self.df.drop(columns=[self.target_col])
            return X_clean, y_clean
        return self.df

    def transform(self, X):
        df = X.copy()
        df = self.create_datetime_features(df)
        
        # Không detect lại types ở Test, dùng logic đã học
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

    # ---------------------------
    # 10. Helper: Lưu dữ liệu đã xử lý
    # ---------------------------
    def save_processed_data(self, df, filename, output_dir="data/processed", y=None):
        """
        Lưu DataFrame ra file CSV.
        :param df: DataFrame chứa Features (X).
        :param filename: Tên file (vd: 'train_clean.csv').
        :param output_dir: Thư mục lưu (Mặc định: 'data/processed').
        :param y: (Optional) Series chứa Target để ghép vào trước khi lưu.
        """
        try:
            # 1. Tạo thư mục nếu chưa tồn tại
            os.makedirs(output_dir, exist_ok=True)
            
            # 2. Ghép X và y nếu có (để lưu thành 1 file trọn vẹn)
            if y is not None:
                # Reset index để đảm bảo ghép đúng dòng
                df = df.reset_index(drop=True)
                y = y.reset_index(drop=True)
                df_to_save = pd.concat([df, y], axis=1)
            else:
                df_to_save = df
            
            # 3. Tạo đường dẫn file
            file_path = os.path.join(output_dir, filename)
            
            # 4. Lưu file
            df_to_save.to_csv(file_path, index=False)
            print(f"Đã lưu dữ liệu vào: {file_path} (Shape: {df_to_save.shape})")
            
        except Exception as e:
            print(f"*** Lỗi khi lưu dữ liệu: {e}")
            
        
    def save_preprocessor(self, path):
        joblib.dump(self, path)
        print(f"Đã lưu Preprocessor tại {path}")
