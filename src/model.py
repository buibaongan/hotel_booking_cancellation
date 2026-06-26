import configparser
import glob
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    train_path: str
    test_path: str
    features: list[str]
    target: str
    n_iter: int
    cv: int
    random_state: int
    model_dir: str = "models"

    @classmethod
    def from_file(cls, config_path: str) -> "TrainingConfig":
        config = configparser.ConfigParser()
        config.read(config_path, encoding="utf-8")
        features = [feature.strip() for feature in config["DATA"]["features"].split(",") if feature.strip()]
        return cls(
            train_path=config["DATA"]["trainpath"],
            test_path=config["DATA"]["testpath"],
            features=features,
            target=config["DATA"]["target"],
            n_iter=int(config["TRAINING"].get("n_iter", 10)),
            cv=int(config["TRAINING"].get("cv", 5)),
            random_state=int(config["TRAINING"].get("random_state", 42)),
        )


class ModelFactory:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def create(self, model_name: str) -> Tuple[Any, Dict[str, list[Any]]]:
        if model_name == "CatBoost":
            return CatBoostClassifier(verbose=0, random_state=self.random_state), {
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "iterations": [200, 500],
            }
        if model_name == "LightGBM":
            return LGBMClassifier(verbose=-1, boosting_type="gbdt", random_state=self.random_state), {
                "num_leaves": [31, 63, 127],
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [200, 500],
            }
        if model_name == "RandomForest":
            return RandomForestClassifier(random_state=self.random_state), {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "bootstrap": [True, False],
            }
        if model_name == "XGBoost":
            return XGBClassifier(eval_metric="logloss", random_state=self.random_state), {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 6, 10],
                "subsample": [0.7, 0.8, 1.0],
            }
        raise ValueError(f"Unsupported model name: {model_name}")


class ModelRepository:
    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def save_model(self, model: Any, filename: str) -> str:
        if model is None:
            raise ValueError("Model not trained.")
        path = os.path.join(self.model_dir, filename)
        joblib.dump(model, path)
        logger.info("Model saved to %s", path)
        return path

    def load_model(self, filename: str) -> Any:
        return joblib.load(os.path.join(self.model_dir, filename))

    def save_json(self, payload: Dict[str, Any], filename: str) -> str:
        path = os.path.join(self.model_dir, filename)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)
        return path

    def save_dataframe(self, df: pd.DataFrame, filename: str) -> str:
        path = os.path.join(self.model_dir, filename)
        df.to_csv(path, index=False)
        return path


class ModelEvaluator:
    @staticmethod
    def classification_report(y_true, y_pred) -> Dict[str, Any]:
        return metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    @staticmethod
    def confusion_matrix(y_true, y_pred) -> list[list[int]]:
        return metrics.confusion_matrix(y_true, y_pred).tolist()

    @staticmethod
    def roc_curve_data(y_true, y_proba) -> Dict[str, Any]:
        if y_proba is None:
            return {}
        fpr, tpr, _ = metrics.roc_curve(y_true, y_proba[:, 1])
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": metrics.auc(fpr, tpr),
        }


class EvaluationPlotter:
    def __init__(self, model_dir: str = "models") -> None:
        self.model_dir = model_dir

    def plot(self, file_pattern: str = "evaluation_*.json") -> None:
        search_path = os.path.join(os.path.abspath(self.model_dir), file_pattern)
        files = glob.glob(search_path)
        if not files:
            print(f"*** KHÔNG TÌM THẤY FILE KẾT QUẢ NÀO TRONG: {self.model_dir}")
            return

        results = []
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as file:
                results.append(json.load(file))

        self._plot_accuracy(results)
        self._plot_roc(results)
        self._plot_confusion_matrices(results)

    def _plot_accuracy(self, results: list[Dict[str, Any]]) -> None:
        scores = pd.DataFrame(
            {
                "Model": [result["model_name"] for result in results],
                "Accuracy": [result["report"]["accuracy"] for result in results],
            }
        ).sort_values("Accuracy", ascending=False)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=scores, x="Model", y="Accuracy", hue="Model", legend=False)
        plt.ylim(0, 1.1)
        plt.title("Model Accuracy Comparison")
        plt.ylabel("Accuracy Score")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        for patch in ax.patches:
            if patch.get_height() > 0:
                ax.annotate(
                    f"{patch.get_height():.4f}",
                    (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
                    ha="center",
                    va="center",
                    xytext=(0, 9),
                    textcoords="offset points",
                    fontweight="bold",
                )
        path = os.path.join(self.model_dir, "comparison_barplot.png")
        plt.savefig(path)
        plt.close()
        print(f"Saved: {path}")

    def _plot_roc(self, results: list[Dict[str, Any]]) -> None:
        plt.figure(figsize=(10, 8))
        plotted = False
        for result in results:
            roc_data = result.get("roc_curve_data") or {}
            if roc_data.get("fpr"):
                plt.plot(
                    roc_data["fpr"],
                    roc_data["tpr"],
                    linewidth=2,
                    label=f"{result['model_name']} (AUC = {roc_data['auc']:.3f})",
                )
                plotted = True

        if plotted:
            plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Guess")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve Comparison")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            path = os.path.join(self.model_dir, "comparison_roc_curve.png")
            plt.savefig(path)
            print(f"Saved: {path}")
        plt.close()

    def _plot_confusion_matrices(self, results: list[Dict[str, Any]]) -> None:
        cols = 2
        rows = math.ceil(len(results) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
        axes = np.array(axes).reshape(-1)

        for index, result in enumerate(results):
            sns.heatmap(
                np.array(result["confusion_matrix"]),
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[index],
                cbar=False,
            )
            axes[index].set_title(result["model_name"])
            axes[index].set_ylabel("Actual Label")
            axes[index].set_xlabel("Predicted Label")

        for index in range(len(results), len(axes)):
            axes[index].axis("off")

        plt.tight_layout()
        path = os.path.join(self.model_dir, "comparison_confusion_matrices.png")
        plt.savefig(path)
        plt.close()
        print(f"Saved: {path}")


class ModelTrainer:
    """Facade used by the CLI and pipeline while delegating responsibilities."""

    MODEL_NAMES = ["CatBoost", "LightGBM", "RandomForest", "XGBoost"]

    def __init__(self, config_path: str = "config.ini") -> None:
        self.config_path = config_path
        self.config = TrainingConfig.from_file(config_path)
        self.features = self.config.features
        self.target = self.config.target
        self.trainpath = self.config.train_path
        self.testpath = self.config.test_path
        self.n_iter = self.config.n_iter
        self.cv = self.config.cv
        self.train: Optional[pd.DataFrame] = None
        self.test: Optional[pd.DataFrame] = None
        self.model: Any = None
        self.model_name: Optional[str] = None
        self.model_dir = self.config.model_dir
        self.factory = ModelFactory(random_state=self.config.random_state)
        self.repository = ModelRepository(self.model_dir)
        self.evaluator = ModelEvaluator()
        self.plotter = EvaluationPlotter(self.model_dir)

    def load_data(self) -> None:
        self.train = pd.read_csv(self.trainpath)
        self.test = pd.read_csv(self.testpath)
        if self.target not in self.train.columns:
            raise KeyError(f"Training data is missing target column: {self.target}")
        processed_features = [col for col in self.train.columns if col != self.target]
        configured_features = set(self.features)
        if configured_features and configured_features != set(processed_features):
            logger.info("Using processed CSV schema instead of stale config feature list.")
        self.features = processed_features

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.3,
        random_state: int = 42,
        stratify=None,
    ):
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    def optimize_params(self, model_name: str) -> Tuple[Any, float]:
        if self.train is None:
            raise ValueError("Data not loaded")

        estimator, param_dist = self.factory.create(model_name)
        X_train = self.train[self.features]
        y_train = self.train[self.target]

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        print(f"Tuning hyperparameters for {model_name}...")
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.model_name = model_name
        logger.info("Best params for %s: %s", model_name, search.best_params_)
        logger.info("Best CV score for %s: %.4f", model_name, search.best_score_)
        return self.model, float(search.best_score_)

    def train_predict(self):
        if self.test is None:
            raise ValueError("Data not loaded")
        if self.model is None:
            raise ValueError("Model not initialized. Please optimize parameters first.")

        X_test = self.test[self.features]
        y_test = self.test[self.target]
        y_pred = self.model.predict(X_test)
        report = self.evaluate(y_pred, y_test)
        logger.info("Report:\n%s", json.dumps(report, indent=4))
        return y_pred, y_test

    @staticmethod
    def evaluate(y_pred, y_test) -> Dict[str, Any]:
        return ModelEvaluator.classification_report(y_test, y_pred)

    def save_metrics(
        self,
        model_name: str,
        y_proba,
        y_test,
        report,
        filename: str,
        best_cv_score: Optional[float] = None,
    ) -> str:
        if self.test is None or self.model is None:
            raise ValueError("Data and model are required before saving metrics.")
        y_pred = self.model.predict(self.test[self.features])
        payload = {
            "model_name": model_name,
            "best_cv_score": best_cv_score,
            "report": report,
            "confusion_matrix": self.evaluator.confusion_matrix(y_test, y_pred),
            "roc_curve_data": self.evaluator.roc_curve_data(y_test, y_proba),
        }
        path = self.repository.save_json(payload, filename)
        logger.info("Metrics for %s saved to %s", model_name, path)
        return path

    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model is not trained.")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "get_feature_importance"):
            importances = self.model.get_feature_importance()
        else:
            raise ValueError("Model does not support feature importance.")

        df = pd.DataFrame({"feature": self.features, "importance": importances})
        df = df.sort_values("importance", ascending=False)
        self.repository.save_dataframe(df, f"feature_importance_{model_name}.csv")
        print(df)
        return df

    def run_model(self, model_name: str) -> Dict[str, Any]:
        model, cv_score = self.optimize_params(model_name)
        y_pred, y_test = self.train_predict()
        report = self.evaluate(y_pred, y_test)
        y_proba = model.predict_proba(self.test[self.features]) if hasattr(model, "predict_proba") else None
        self.save_metrics(
            model_name,
            y_proba,
            y_test,
            report,
            f"evaluation_{model_name}.json",
            best_cv_score=cv_score,
        )
        try:
            self.get_feature_importance(model_name)
        except ValueError as error:
            logger.warning("%s", error)

        return {
            "Model": model_name,
            "Best_CV_Score": cv_score,
            "Accuracy": report["accuracy"],
            "Precision_0": report["0"]["precision"],
            "Recall_0": report["0"]["recall"],
            "F1_Score_0": report["0"]["f1-score"],
            "Precision_1": report["1"]["precision"],
            "Recall_1": report["1"]["recall"],
            "F1_Score_1": report["1"]["f1-score"],
            "Macro_Avg_F1": report["macro avg"]["f1-score"],
        }

    def auto_select_model(self) -> None:
        if self.train is None or self.test is None:
            raise ValueError("Data not loaded")

        summary_rows = []
        best_score = -1.0
        best_model = None
        best_name = ""

        for model_name in self.MODEL_NAMES:
            try:
                row = self.run_model(model_name)
                summary_rows.append(row)
                print(f"    {model_name} score: {row['Best_CV_Score']:.4f}")
                if row["Best_CV_Score"] > best_score:
                    best_score = row["Best_CV_Score"]
                    best_model = self.model
                    best_name = model_name
            except Exception as error:
                logger.exception("Failed to train %s", model_name)
                print(f"     > {model_name} failed: {error}")

        if not summary_rows or best_model is None:
            raise ValueError("All models have failed")

        summary = pd.DataFrame(summary_rows).sort_values("Best_CV_Score", ascending=False)
        self.repository.save_dataframe(summary, "model_comparison_summary.csv")
        print("\n[INFO] Đã lưu bảng so sánh mô hình vào: models/model_comparison_summary.csv")
        print(summary)

        self.model = best_model
        self.model_name = best_name
        self.save_model("best_model.pkl")
        print(f"\nTHE BEST MODEL IS: {best_name} with CV score {best_score:.4f}")

    def save_model(self, filename: str = "model.pkl") -> None:
        self.repository.save_model(self.model, filename)

    def load_model(self, filename: str = "model.pkl") -> None:
        self.model = self.repository.load_model(filename)

    def plot_evaluation_results(self, file_pattern: str = "evaluation_*.json") -> None:
        self.plotter.plot(file_pattern)
