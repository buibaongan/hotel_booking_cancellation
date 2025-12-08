# Load libraries
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from scipy.stats import randint
import joblib  # Used for saving/loading models
import argparse
import configparser
from xgboost import XGBClassifier
import json
import glob
import seaborn as sns
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

"""
logging.basicConfig(
    filename='activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
"""
logger = logging.getLogger(__name__) 

class ModelTrainer: 
    """
    Lớp ModelTrainer dùng để:
        - Tải dữ liệu huấn luyện
        - Tinh chỉnh các siêu tham số vủa 4 thuật toán (Random Forest, CatBoost, LightGBM, XGBoost)
        - Huấn luyện mô hình
        - Đánh giá mô hình 
        - Tự chọn ra mô hình tốt nhất 
        - Ghi lại kết quả thực nghiệm
        - So sánh kết quả giữa các mô hình và lưu biểu đồ đánh giá

    Attributes:

        - config : ConfigParser
            Đối tượng đọc thông tin từ file config.ini.
        - features : list[str]
            Danh sách các tên cột được dùng làm biến độc lập (feature).
        - target : str
            Tên cột mục tiêu (label) trong dữ liệu.
        - trainpath : str
            Đường dẫn file CSV của tập huấn luyện.
        - testpath : str
            Đường dẫn file CSV của tập kiểm tra.
        - n_iter : int
            Số lượng tổ hợp tham số sẽ thử trong RandomizedSearchCV.
        - cv : int
            Số fold cross-validation.
        - train : pandas.DataFrame
            Dữ liệu huấn luyện sau khi load.
        - test : pandas.DataFrame
            Dữ liệu kiểm tra sau khi load.
        - model : object
            Mô hình đã được huấn luyện hoặc mô hình tốt nhất sau tinh chỉnh.

    Methods:

    - load_data():
        Đọc dữ liệu từ đường dẫn trong config và lưu vào self.train và self.test.

    - evaluate(y_pred, y_test):
        Tính classification report và trả về dạng dictionary.

    - optimize_params(model_name):
        Tinh chỉnh mô hình bằng RandomizedSearchCV dựa trên tên thuật toán.

    - train_predict():
        Dự đoán trên tập test và trả về y_pred và y_test.

    - auto_select_model():
        Chạy toàn bộ các mô hình, chọn mô hình tốt nhất dựa trên điểm CV.

    - save_metrics(model_name, y_proba, y_test, report, filename):
        Lưu báo cáo đánh giá (report, confusion matrix, ROC) dưới dạng JSON.

    - plot_evaluation_results(file_pattern):
        Vẽ biểu đồ so sánh Accuracy, ROC, và Confusion Matrix của nhiều mô hình.

    - save_model(filename):
        Lưu mô hình vào file .pkl.

    - load_model(filename):
        Tải mô hình từ file .pkl.
    """

    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8')

        self.features = self.config['DATA']['features'].split(',')
        self.target = self.config['DATA']['target']
        self.testpath = self.config['DATA']['testpath']
        self.trainpath = self.config['DATA']['trainpath']
        self.n_iter = int(self.config['TRAINING']['n_iter'])
        self.cv = int(self.config['TRAINING']['cv'])
        self.train = None
        self.test = None
        self.model = None

        # --- TẠO THƯ MỤC LƯU KẾT QUẢ MODELS ---
        self.model_dir = 'models'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            logging.info(f"Created directory: {self.model_dir}")
            
        logging.info("\n\n\nModelTrainer initialized.")

    def load_data(self):
        """
        Tải dữ liệu từ đường dẫn cấu hình trong file config.
        Raises:
            ValueError Nếu không tìm thấy trainpath hoặc testpath.
        """
        if self.trainpath and self.testpath:
            logging.info(f"Loading data from {self.trainpath}...")
            self.train = pd.read_csv(self.trainpath)
            logging.info(f"Loading data from {self.testpath}...")
            self.test = pd.read_csv(self.testpath)
        else:
            logging.error(f"Error loading data!!!")
            raise ValueError("trainpath or testpath not found in config.")

    # Trong file src/model.py -> class ModelTrainer
    def split_data(self, X, y, test_size=0.3, random_state=42, stratify=None):
        """
        Chia dữ liệu thành tập huấn luyện và tập kiểm tra.

        Args:
            X: Dữ liệu đặc trưng.
            y: Nhãn mục tiêu.
            test_size: Tỉ lệ dữ liệu dành cho kiểm tra (mặc định 0.3).
            random_state: Giá trị cố định để tái lập (mặc định 42).
            stratify: Mảng dùng để phân tầng khi chia (mặc định None).
        Returns:
            X_train, X_test, y_train, y_test: Các tập dữ liệu sau khi chia.
        """
        
        # Thêm tham số stratify vào hàm train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=stratify
        )
        return X_train, X_test, y_train, y_test
    
    
    @staticmethod
    def evaluate(y_pred, y_test):
        """
        Tính toán classification report 

        :param y_pred: Nhãn dự đoán từ mô hình.
        :param y_test: Nhãn thực tế.
        Returns
            dict báo cáo đánh giá (precision, recall, f1-score, accuracy).
        """
        report = metrics.classification_report(y_test, y_pred, output_dict = True)
        return report
    
    def optimize_params(self, model_name):
        """
        Tinh chỉnh siêu tham số của mô hình theo RandomizedSearchCV.

        Parameters: 
            model_name : str
                Tên thuật toán cần tinh chỉnh. 
                Một trong: ['CatBoost', 'LightGBM', 'RandomForest', 'XGBoost'].

        Returns
            best_estimator : object
                Mô hình tốt nhất sau tinh chỉnh.
            best_score : float
                Điểm cross-validation cao nhất.
        Raises
            ValueError Nếu dữ liệu chưa được load hoặc tên model không hợp lệ.
        """
        if self.train is None:
            logging.error("Data not loaded!!!")
            raise ValueError("Data not loaded")
        
        logging.info(f"Tuning hyperparameters for {model_name}...")
        print(f"Tuning hyperparameters for {model_name}...")

        X_train = self.train[self.features]
        y_train = self.train[self.target]
        base_model = None 
        param_dist = {}

        if model_name == "CatBoost":
            base_model = CatBoostClassifier(verbose=0, random_state=42)
            param_dist = {
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [200, 500]
            }
        
        elif model_name == "LightGBM":
            param_dist = {
                'num_leaves': [31, 63, 127],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [200, 500]
            }
            base_model = LGBMClassifier(verbose=-1,boosting_type="gbdt",random_state=42)

        elif model_name == "RandomForest":
            param_dist = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'bootstrap': [True, False]
            }
            base_model = RandomForestClassifier(random_state=42)

        elif model_name == 'XGBoost':
            base_model = XGBClassifier(eval_metric='logloss', random_state=42)
            param_dist = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10],
                'subsample': [0.7, 0.8, 1.0]
            }

        else:
            logging.error("Unsupported model name")
            raise ValueError("Unsupported model name.")
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_
        
        logging.info(f"Optimization complete. Best Params: {random_search.best_params_}")
        logging.info(f"Best CV Score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_score_

    def train_predict(self):
        """
        Dự đoán trên tập test bằng mô hình đã tinh chỉnh.

        Returns
            tuple
                (y_pred, y_test)

        Raises
            ValueError
                Nếu không có dữ liệu hoặc mô hình chưa được tinh chỉnh.
        """
        if self.test is None:
            raise ValueError("Data not loaded")
        if self.model is None:
            raise ValueError("Model not initialized. Please optimize parameters first.")
        
        X_test = self.test[self.features]
        y_test = self.test[self.target]

        y_pred = self.model.predict(X_test)
        report = self.evaluate(y_pred, y_test)
        logging.info("Report: ")
        logging.info(f"\n{json.dumps(report, indent=4)}")
        return y_pred, y_test

    def auto_select_model(self):
        """
        Tự động chạy tất cả mô hình, đánh giá, chọn mô hình tốt nhất 
        và lưu bảng so sánh ra CSV.
        """
        print("Starting auto-selecting model....")
        logging.info("Starting auto-selecting model....")

        models = ['CatBoost', 'LightGBM', 'RandomForest', 'XGBoost']
        best_score = -1 
        best_model = None 
        best_name = ''
        
        # Danh sách chứa kết quả để lưu CSV
        summary_results = []

        for m in models:
            try:
                model, score = self.optimize_params(model_name = m)
                self.model = model
                self.model_name = m
                print(f"    {m}     score: {score:.4f}")

                y_pred, y_test = self.train_predict()
                self.get_feature_importance(m)
                report = self.evaluate(y_pred, y_test)

                y_proba = None
                if hasattr(self.model, "predict_proba"):
                    y_proba = self.model.predict_proba(self.test[self.features])
                
                # Lưu file JSON chi tiết
                self.save_metrics(m, y_proba, y_test, report, filename = f'evaluation_{m}.json')
                
                # --- Thêm thông tin vào bảng tổng hợp ---
                summary_results.append({
                    "Model": m,
                    "Best_CV_Score": score,
                    "Accuracy": report['accuracy'],
                    "Precision_0": report['0']['precision'],
                    "Recall_0": report['0']['recall'],
                    "F1_Score_0": report['0']['f1-score'],
                    "Precision_1": report['1']['precision'],
                    "Recall_1": report['1']['recall'],
                    "F1_Score_1": report['1']['f1-score'],
                    "Macro_Avg_F1": report['macro avg']['f1-score']
                })

                if score > best_score: 
                    best_score = score 
                    best_model = model 
                    best_name = m 
            except Exception as e:
                logging.error(f"Failed to train {m} : {e}")
                print(f"     >{m} failed")

        # --- Lưu bảng so sánh ra file CSV ---
        if summary_results:
            summary_df = pd.DataFrame(summary_results)
            # Sắp xếp theo Accuracy giảm dần
            summary_df = summary_df.sort_values(by="Accuracy", ascending=False)
            
            # Tạo đường dẫn lưu file
            csv_path = os.path.join(self.model_dir, "model_comparison_summary.csv")
            summary_df.to_csv(csv_path, index=False)
            
            print(f"\n[INFO] Đã lưu bảng so sánh mô hình vào: {csv_path}")
            logging.info(f"Model comparison summary saved to {csv_path}")
            print("-" * 50)
            print(summary_df) # In ra màn hình để xem nhanh
            print("-" * 50)

        if best_model is not None: 
            self.model = best_model 
            print(f"\n\nTHE BEST MODEL IS : {best_name} with accuracy {best_score}")
            logging.info(f"\n\nTHE BEST MODEL IS : {best_name} with accuracy {best_score}\n")
        else: 
            raise ValueError("All models have failed")
                    

    def save_metrics(self, model_name, y_proba, y_test, report, filename):
        """
        Lưu kết quả đánh giá của mô hình xuống file JSON.
        Return dữ liệu để vẽ ROC curve, confusion matrix, barplot.

        Parameters
            model_name : str
                Tên mô hình.
            y_proba : array-like or None
                Xác suất dự đoán (dùng cho ROC). Có thể None nếu model không hỗ trợ.
            y_test : array-like
                Nhãn thực tế.
            report : dict
                Classification report dạng dictionary.
            filename : str
                Tên file lưu JSON.
        """
            
        roc_data = {}

        if y_proba is not None: 
            fpr, tpr, _ = metrics.roc_curve(y_test, y_proba[:, 1])
            roc_auc = metrics.auc(fpr, tpr)

            roc_data = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": roc_auc
            }
            cm = metrics.confusion_matrix(y_test, self.model.predict(self.test[self.features]))

        eval_data = {
            "model_name": model_name,
            "best_cv_score": self.model.best_score_ if hasattr(self.model, 'best_score_') else None,
            "report": report,
            "confusion_matrix": cm.tolist(),
            "roc_curve_data": roc_data
        }
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        # Tạo đường dẫn đầy đủ đến file
        file_path = os.path.join(self.model_dir, filename)

        with open(file_path, 'w') as f:
            json.dump(eval_data, f, indent = 4)
        logging.info(f"Mectrics for {model_name} saved to {file_path}")

    def get_feature_importance(self, model_name):
        """
        Tính và lưu độ quan trọng của các đặc trưng (feature importance).

        Tham số
            model_name : str
                Tên mô hình dùng để đặt tên file xuất ra.

        Trả về
            DataFrame
                Bảng gồm tên feature và mức độ quan trọng, đã sắp xếp giảm dần.
        """

        if self.model is None:
            raise ValueError("Model is not trained.")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "get_feature_importance"):
            importances = self.model.get_feature_importance()
        else:
            raise ValueError("Model does not support feature importance.")

        df = pd.DataFrame({
            "feature": self.features,
            "importance": importances
        }).sort_values("importance", ascending=False)

        print(df)
        
        # Lưu file csv vào thư mục models
        file_path = os.path.join(self.model_dir, f"feature_importance_{model_name}.csv")
        df.to_csv(file_path, index=False)
        logging.info(f" saved {file_path}")
        
        return df

    def plot_evaluation_results(self, file_pattern = "evaluation_*.json"):
        """
        Đọc tất cả file JSON đánh giá và vẽ:
            - Biểu đồ so sánh Accuracy
            - ROC Curve của các mô hình
            - Confusion matrix của từng mô hình

        Parameters

        file_pattern : str
            Pattern để đọc các file JSON (mặc định: "evaluation_*.json").
        """
        # Tạo đường dẫn tìm kiếm chính xác
        # os.path.abspath giúp chuyển đổi đường dẫn tương đối thành tuyệt đối để debug dễ hơn
        search_dir = os.path.abspath(self.model_dir)
        search_path = os.path.join(search_dir, file_pattern)
        
        print(f"\n[DEBUG] Đang tìm kiếm file tại: {search_path}")
        logging.info(f"Searching for evaluation files at: {search_path}")
        
        files = glob.glob(search_path)
        
        if not files:
            msg = f"❌ KHÔNG TÌM THẤY FILE KẾT QUẢ NÀO TRONG: {search_dir}"
            logging.error(msg)
            print(msg)
            
            # Thử liệt kê các file đang có trong thư mục models để xem có file nào không
            if os.path.exists(search_dir):
                print(f"Các file hiện có trong thư mục models: {os.listdir(search_dir)}")
            else:
                print(f"Thư mục models không tồn tại!")
            return 
        
        print(f"[DEBUG] Tìm thấy {len(files)} file kết quả: {[os.path.basename(f) for f in files]}")
        
        results = [] 
        for file in files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                logging.error(f"Lỗi khi đọc file {file}: {e}")

        if not results:
            logging.error("No valid evaluation data loaded")
            print("No valid evaluation data loaded")
            return

        logging.info(f"Found {len(results)} model evaluations. Generating plots....")
        print(f"Generating plots for {len(results)} models....")
        
        # --- PHẦN VẼ BIỂU ĐỒ (Giữ nguyên logic cũ nhưng cập nhật đường dẫn save) ---
        
        # 1. BARPLOT
        model_names = [res['model_name'] for res in results]
        accuracies = [res['report']['accuracy'] for res in results]

        df_scores = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies})
        df_scores = df_scores.sort_values(by='Accuracy', ascending=False)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_scores, x='Model', y='Accuracy', palette='viridis', hue='Model', legend=False)
        plt.ylim(0, 1.1)
        plt.title('Model Accuracy Comparison', fontsize=16)
        plt.ylabel('Accuracy Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for p in ax.patches:
            if p.get_height() > 0: 
                ax.annotate(f'{p.get_height():.4f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontweight='bold')
        
        # Lưu vào self.model_dir
        save_path = os.path.join(self.model_dir, 'comparison_barplot.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

        # 2. ROC CURVE
        plt.figure(figsize=(10, 8))
        roc_plotted = False

        for res in results:
            name = res['model_name']
            roc_data = res.get('roc_curve_data')
            
            if roc_data and roc_data.get('fpr'): 
                fpr = roc_data['fpr']
                tpr = roc_data['tpr']
                auc = roc_data['auc']
                plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')
                roc_plotted = True
        
        if roc_plotted:
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Comparison')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            
            save_path = os.path.join(self.model_dir, 'comparison_roc_curve.png')
            plt.savefig(save_path)
            print(f"Saved: {save_path}")
        plt.close()

        # 3. CONFUSION MATRICES
        num_models = len(results)
        cols = 2
        rows = math.ceil(num_models / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
        if num_models == 1: axes = [axes]
        else: axes = axes.flatten()

        for i, res in enumerate(results):
            name = res['model_name']
            cm = np.array(res['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
            axes[i].set_title(f'{name}', fontsize=14)
            axes[i].set_ylabel('Actual Label')
            axes[i].set_xlabel('Predicted Label')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        # Lưu Confusion Matrix vào thư mục models
        save_path = os.path.join(self.model_dir, 'comparison_confusion_matrices.png')
        plt.savefig(save_path)
        plt.close()
        logging.info("Saved: comparison_confusion_matrices.png")
        print(f"Saved: {save_path}")
        
    def save_model(self, filename = 'model.pkl'):
        """
        Lưu mô hình đã tinh chỉnh xuống file .pkl.

        Parameters
        filename : str
            Tên file để lưu mô hình.

        Raises
            ValueError
                Nếu mô hình chưa được huấn luyện.
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        file_path = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, file_path)
        logging.info(f"Model saved to {file_path}")

    def load_model(self, filename = 'model.pkl'):
        """
        Tải mô hình đã lưu từ file .pkl.

        Parameters

            filename : str
                File mô hình.

        Nếu file không tồn tại, log sẽ thông báo.
        """
        file_path = os.path.join(self.model_dir, filename)
        try: 
            self.model = joblib.load(file_path)
            logging.info(f"Model loaded from {file_path}")
        except FileNotFoundError:
            logging.info(f"File {file_path} not found.")
