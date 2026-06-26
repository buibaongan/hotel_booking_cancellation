import os
import re
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd


class BaseTransformer:
    """Small fit/transform contract used by all preprocessing steps."""

    def fit(self, df: pd.DataFrame) -> "BaseTransformer":
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


class ColumnCleaner(BaseTransformer):
    def fit(self, df: pd.DataFrame) -> "ColumnCleaner":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result.columns = [self.clean_name(col) for col in result.columns]
        return result

    @staticmethod
    def clean_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned.lower()


class MissingValueImputer(BaseTransformer):
    def __init__(self, num_strategy: str = "median", cat_strategy: str = "mode") -> None:
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.impute_values: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "MissingValueImputer":
        self.impute_values = {}
        numeric_cols = df.select_dtypes(include=["number", "bool"]).columns
        categorical_cols = df.select_dtypes(exclude=["number", "bool"]).columns

        for col in numeric_cols:
            if self.num_strategy == "mean":
                value = df[col].mean()
            elif self.num_strategy == "median":
                value = df[col].median()
            else:
                value = 0
            self.impute_values[col] = 0 if pd.isna(value) else value

        for col in categorical_cols:
            if self.cat_strategy == "mode":
                mode = df[col].mode(dropna=True)
                value = mode.iloc[0] if not mode.empty else "Unknown"
            else:
                value = "Unknown"
            self.impute_values[col] = value

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col, value in self.impute_values.items():
            if col in result.columns:
                result[col] = result[col].fillna(value)
        return result.fillna(0)


class CategoricalEncoder(BaseTransformer):
    """One-hot encoder that preserves the train-time feature schema."""

    def __init__(self) -> None:
        self.categorical_cols: list[str] = []
        self.encoded_columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        self.categorical_cols = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
        encoded = pd.get_dummies(df, columns=self.categorical_cols, drop_first=False)
        self.encoded_columns = encoded.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        source = df.copy()
        for col in self.categorical_cols:
            if col not in source.columns:
                source[col] = "Unknown"
        result = pd.get_dummies(source, columns=self.categorical_cols, drop_first=False)
        result = result.reindex(columns=self.encoded_columns, fill_value=0)
        return result.astype(float)


class NumericScaler(BaseTransformer):
    def __init__(self, method: str = "minmax") -> None:
        self.method = method
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.mins: Dict[str, float] = {}
        self.maxs: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "NumericScaler":
        numeric_cols = df.select_dtypes(include=["number", "bool"]).columns
        self.means = {}
        self.stds = {}
        self.mins = {}
        self.maxs = {}

        for col in numeric_cols:
            series = df[col].astype(float)
            self.means[col] = float(series.mean())
            std = float(series.std())
            self.stds[col] = std if std != 0 else 1.0
            self.mins[col] = float(series.min())
            self.maxs[col] = float(series.max())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy().astype(float)
        if self.method == "standard":
            for col, mean in self.means.items():
                if col in result.columns:
                    result[col] = (result[col] - mean) / self.stds[col]
            return result

        for col, min_value in self.mins.items():
            if col in result.columns:
                denominator = self.maxs[col] - min_value
                denominator = denominator if denominator != 0 else 1.0
                result[col] = (result[col] - min_value) / denominator
        return result


class HotelFeatureEngineer(BaseTransformer):
    """Feature engineering specific to the hotel booking dataset."""

    def fit(self, df: pd.DataFrame) -> "HotelFeatureEngineer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        stays_cols = ["stays_in_weekend_nights", "stays_in_week_nights"]
        if set(stays_cols).issubset(result.columns):
            result["total_nights"] = result[stays_cols].sum(axis=1)
            result = result.drop(columns=stays_cols)

        guest_cols = ["adults", "children", "babies"]
        if set(guest_cols).issubset(result.columns):
            result[guest_cols] = result[guest_cols].fillna(0)
            result["total_guests"] = result[guest_cols].sum(axis=1)
            result["has_children"] = (result[["children", "babies"]].sum(axis=1) > 0).astype(int)
            result = result.drop(columns=guest_cols)

        if "country" in result.columns:
            result["is_domestic"] = (result["country"] == "PRT").astype(int)
            result = result.drop(columns=["country"])

        if {"reserved_room_type", "assigned_room_type"}.issubset(result.columns):
            result["is_room_changed"] = (
                result["reserved_room_type"] != result["assigned_room_type"]
            ).astype(int)

        return result


class DataPreprocessor:
    """Train-fitted preprocessing pipeline for modeling and inference."""

    DEFAULT_DROP_COLUMNS = [
        "reservation_status",
        "reservation_status_date",
        "assigned_room_type",
        "arrival_date_year",
        "agent",
        "company",
    ]

    def __init__(
        self,
        target_col: Optional[str] = None,
        scale_method: str = "minmax",
        drop_columns: Optional[list[str]] = None,
    ) -> None:
        self.target_col = ColumnCleaner.clean_name(target_col) if target_col else None
        self.scale_method = scale_method
        self.drop_columns = drop_columns or self.DEFAULT_DROP_COLUMNS
        self.column_cleaner = ColumnCleaner()
        self.feature_engineer = HotelFeatureEngineer()
        self.imputer = MissingValueImputer()
        self.encoder = CategoricalEncoder()
        self.scaler = NumericScaler(method=scale_method)
        self.feature_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.raw_feature_columns_: list[str] = []
        self.required_raw_columns_: list[str] = []

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.column_cleaner.transform(df)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(keep="first").reset_index(drop=True)

    def drop_unusable_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = [ColumnCleaner.clean_name(col) for col in self.drop_columns]
        return df.drop(columns=[col for col in columns if col in df.columns])

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.feature_engineer.transform(df)

    def fill_missing(
        self,
        df: pd.DataFrame,
        num_strategy: str = "median",
        cat_strategy: str = "mode",
        mode: str = "fit_transform",
    ) -> pd.DataFrame:
        if mode == "fit_transform":
            self.imputer = MissingValueImputer(num_strategy, cat_strategy)
            return self.imputer.fit_transform(df)
        return self.imputer.transform(df)

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        if fit:
            return self.encoder.fit_transform(df)
        return self.encoder.transform(df)

    def scale_numeric(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        if fit:
            self.scaler = NumericScaler(method=self.scale_method)
            return self.scaler.fit_transform(df)
        return self.scaler.transform(df)

    def prepare_raw_data(self, df: pd.DataFrame, remove_duplicates: bool = False) -> pd.DataFrame:
        result = self.clean_column_names(df)
        if remove_duplicates:
            result = self.remove_duplicates(result)
        result = self.engineer_features(result)
        result = self.drop_unusable_columns(result)
        return result

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataPreprocessor":
        cleaned = self.clean_column_names(df)
        self.raw_feature_columns_ = cleaned.columns.tolist()
        drop_columns = {ColumnCleaner.clean_name(col) for col in self.drop_columns}
        # assigned_room_type is dropped after it is used to derive is_room_changed.
        columns_used_before_drop = {"assigned_room_type"}
        ignored_columns = drop_columns - columns_used_before_drop
        self.required_raw_columns_ = [
            col for col in self.raw_feature_columns_ if col not in ignored_columns
        ]
        prepared = self.prepare_raw_data(df)
        imputed = self.fill_missing(prepared, mode="fit_transform")
        self.numeric_columns_ = imputed.select_dtypes(include=["number", "bool"]).columns.tolist()
        self.categorical_columns_ = imputed.select_dtypes(exclude=["number", "bool"]).columns.tolist()
        encoded = self.encode_categorical(imputed, fit=True)
        scaled = self.scale_numeric(encoded, fit=True)
        self.feature_columns_ = scaled.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = self.prepare_raw_data(df)
        imputed = self.fill_missing(prepared, mode="transform")
        encoded = self.encode_categorical(imputed, fit=False)
        scaled = self.scale_numeric(encoded, fit=False)
        return scaled.reindex(columns=self.feature_columns_, fill_value=0)

    def fit_transform(
        self, df: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        self.fit(df, y)
        transformed = self.transform(df)
        return transformed, y.reset_index(drop=True) if y is not None else None

    def split_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        cleaned = self.clean_column_names(df)
        if not self.target_col or self.target_col not in cleaned.columns:
            raise KeyError(f"Target column '{self.target_col}' was not found.")
        return cleaned.drop(columns=[self.target_col]), cleaned[self.target_col]

    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: str,
        output_dir: str = "data/processed",
        y: Optional[pd.Series] = None,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        output = df.reset_index(drop=True)
        if y is not None:
            output[self.target_col] = y.reset_index(drop=True)
        path = os.path.join(output_dir, filename)
        output.to_csv(path, index=False)
        print(f"Saved data to: {path} (Shape: {output.shape})")

    def save_preprocessor(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self, path)
        print(f"Saved preprocessor to {path}")
