import argparse
import configparser
import logging
import os

import pandas as pd

from src.model import ModelTrainer
from src.preprocessing import DataPreprocessor


MODEL_ALIASES = {
    "auto": "Auto",
    "catboost": "CatBoost",
    "cat": "CatBoost",
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",
    "randomforest": "RandomForest",
    "rf": "RandomForest",
}


def configure_logging() -> logging.Logger:
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("activity.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


logger = configure_logging()


class BookingCancellationPipeline:
    """Application workflow for preprocessing, training, and reporting."""

    def __init__(self, config_path: str = "config.ini") -> None:
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding="utf-8")
        self.target_col = self.config["DATA"]["target"]
        self.input_path = self.config["PREPROCESSING"]["inputpath"]
        self.train_path = self.config["DATA"].get("trainpath", "data/processed/train_processed.csv")
        self.test_path = self.config["DATA"].get("testpath", "data/processed/test_processed.csv")
        self.test_size = float(self.config["TRAINING"].get("test_size", 0.2))
        self.random_state = int(self.config["TRAINING"].get("random_state", 42))

    def run_preprocessing(self) -> None:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {self.input_path}")

        logger.info("Load dữ liệu gốc: %s", self.input_path)
        raw = pd.read_csv(self.input_path, dtype={"agent": "object", "company": "object"})
        preprocessor = DataPreprocessor(target_col=self.target_col)

        cleaned = preprocessor.clean_column_names(raw)
        cleaned = preprocessor.remove_duplicates(cleaned)
        X, y = preprocessor.split_target(cleaned)

        trainer_tool = ModelTrainer(config_path=self.config_path)
        X_train_raw, X_test_raw, y_train, y_test = trainer_tool.split_data(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        logger.info("Fit preprocessing on train data only.")
        X_train, y_train = preprocessor.fit_transform(X_train_raw, y_train)
        X_test = preprocessor.transform(X_test_raw)

        train_dir, train_file = os.path.split(self.train_path)
        test_dir, test_file = os.path.split(self.test_path)
        train_dir = train_dir or "."
        test_dir = test_dir or "."
        preprocessor.save_processed_data(X_train, train_file, output_dir=train_dir, y=y_train)
        preprocessor.save_processed_data(X_test, test_file, output_dir=test_dir, y=y_test)
        preprocessor.save_preprocessor("reports/preprocessor.joblib")
        logger.info("HOÀN TẤT PREPROCESSING.")

    def run_training(self, model_name: str) -> None:
        trainer = ModelTrainer(config_path=self.config_path)
        trainer.load_data()

        if model_name == "Auto":
            trainer.auto_select_model()
            trainer.plot_evaluation_results()
            return

        trainer.run_model(model_name)
        trainer.save_model(filename=f"{model_name}.pkl")
        logger.info("Đã lưu model tại models/%s.pkl", model_name)

    def run(self, model_name: str = "Auto") -> None:
        self.run_preprocessing()
        self.run_training(model_name)

def ask_model_name() -> str:
    print("\n" + "=" * 70)
    print("GÕ 'model' ĐỂ TIẾP TỤC CHẠY MODEL (Gõ 'exit' để thoát)")
    print("=" * 70)

    while True:
        action = input(">> ").strip().lower()
        if action == "exit":
            raise KeyboardInterrupt
        if action == "model":
            break
        print("Nhập sai! Hãy gõ 'model' để tiếp tục hoặc 'exit' để thoát.")

    print("\n" + "-" * 50)
    print("CHỌN THUẬT TOÁN: Auto, CatBoost, XGBoost, LightGBM, RandomForest")
    raw = input(">> Nhập tên model (Mặc định 'Auto' nếu bỏ trống): ").strip().lower()
    return MODEL_ALIASES.get(raw, "Auto")


def parse_args():
    parser = argparse.ArgumentParser(description="Hotel booking cancellation ML pipeline")
    parser.add_argument("--config", default="config.ini", help="Path to configuration file")
    parser.add_argument("--model", choices=["Auto", "CatBoost", "XGBoost", "LightGBM", "RandomForest"])
    parser.add_argument("--no-prompt", action="store_true", help="Run without interactive prompts")
    parser.add_argument("--preprocess-only", action="store_true", help="Only generate processed train/test data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = BookingCancellationPipeline(config_path=args.config)

    try:
        pipeline.run_preprocessing()
        if args.preprocess_only:
            return
        model_name = args.model or ("Auto" if args.no_prompt else ask_model_name())
        print(f"-> Thuật toán được chọn: {model_name}")
        pipeline.run_training(model_name)
        print("\n*** ĐÃ HOÀN TẤT HUẤN LUYỆN MODEL!")
    except KeyboardInterrupt:
        print("\nĐã thoát chương trình.")
    except Exception as error:
        logger.exception("Pipeline failed")
        print(f"*** Lỗi pipeline: {error}")


if __name__ == "__main__":
    main()
