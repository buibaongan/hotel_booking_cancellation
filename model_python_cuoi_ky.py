# Load libraries
import pandas as pd
import logging
#from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from scipy.stats import randint
import joblib  # Used for saving/loading models
import argparse
import configparser
#from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import json
import glob
import seaborn as sns
import math
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

logging.basicConfig(
    filename='activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
        self.config.read(config_path)

        self.features = self.config['DATA']['features'].split(',')
        self.target = self.config['DATA']['target']
        self.testpath = self.config['DATA']['testpath']
        self.trainpath = self.config['DATA']['trainpath']
        self.n_iter = int(self.config['TRAINING']['n_iter'])
        self.cv = int(self.config['TRAINING']['cv'])
        self.train = None
        self.test = None
        self.model = None

        logging.info("\n\n\nModelTrainer initialized.")

    def load_data(self):
        """
       Tải dữ liệu từ đường dẫn cấu hình trong file config.

        Raises:
            ValueError
                Nếu không tìm thấy trainpath hoặc testpath.
        """
        if self.trainpath and self.testpath:
            logging.info(f"Loading data from {self.trainpath}...")
            self.train = pd.read_csv(self.trainpath)
            logging.info(f"Loading data from {self.testpath}...")
            self.test = pd.read_csv(self.testpath)
        else:
            logging.error(f"Error loading data!!!")
            raise ValueError("trainpath or testpath not found in config.")

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
            ValueError
                Nếu dữ liệu chưa được load hoặc tên model không hợp lệ.
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
        Tự động chạy tất cả mô hình (CatBoost, LightGBM, RandomForest, XGBoost),
        tinh chỉnh tham số cho từng mô hình, đánh giá và chọn mô hình tốt nhất.

        Ghi log toàn bộ quá trình và lưu file JSON kết quả từng mô hình.

        Raises
            ValueError
                Nếu tất cả mô hình đều lỗi.
        """
        print("Starting auto-selecting model....")
        logging.info("Starting auto-selecting model....")

        models = ['CatBoost', 'LightGBM', 'RandomForest', 'XGBoost']
        best_score = -1 
        best_model = None 
        best_name = ''

        for m in models:
            try:
                model, score = self.optimize_params(model_name = m)
                self.model = model
                self.model_name = m
                print(f"    {m}     score: {score:.4f}")

                y_pred, y_test = self.train_predict()
                report = self.evaluate(y_pred, y_test)

                y_proba = None
                if hasattr(self.model, "predict_proba"):
                    y_proba = self.model.predict_proba(self.test[self.features])
                
                self.save_metrics(m, y_proba, y_test, report, filename = f'evaluation_{m}.json')

                if score > best_score: 
                    best_score = score 
                    best_model = model 
                    best_name = m 
            except Exception as e:
                logging.error(f"Failed to train {m} : {e}")
                print(f"     >{m} failed")

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

        with open(filename, 'w') as f:
            json.dump(eval_data, f, indent = 4)
        logging.info(f"Mectrics for {model_name} saved to {filename}")


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
        files = glob.glob(file_pattern)
        if not files:
            logging.error("No evaluation files found")
            print("No evaluation files found")
            return 
        
        results = [] 
        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)
                results.append(data)

        if not results:
            logging.error("No evaluation data loaded")
            print("No evaluation data loaded")

        logging.info(f"Found {len(results)} model evaluations. Generating plots....")
        print(f"Found {len(results)} model evaluations. Generating plots....")
        
        #---------------  BARPLOT (score comparison) -------------------
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
            if p.get_height() > 0: # Safety check
                ax.annotate(f'{p.get_height():.4f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontweight='bold')
                
        plt.savefig('comparison_barplot.png')
        plt.close()
        logging.info("Saved: comparison_barplot.png")
        print("Saved: comparison_barplot.png")

        #---------------------ROC CURVE---------------------
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
            plt.savefig('comparison_roc_curve.png')
            logging.info("Saved: comparison_roc_curve.png")
            print("Saved: comparison_roc_curve.png")
        else:
            logging.error("No roc data found in files")
            print("No roc data found in files")
        plt.close()


        #-------------------------CONFUSION MATRICES---------------

        num_models = len(results)
        cols = 2
        rows = math.ceil(num_models / cols)
        
        # Adjust figure size based on number of rows
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
        
        # 1 model case
        if num_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, res in enumerate(results):
            name = res['model_name']
            cm = np.array(res['confusion_matrix']) # Convert list back to numpy array
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
            axes[i].set_title(f'{name}', fontsize=14)
            axes[i].set_ylabel('Actual Label')
            axes[i].set_xlabel('Predicted Label')

        # Turn off unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig('comparison_confusion_matrices.png')
        plt.close()
        logging.info("Saved: comparison_confusion_matrices.png")
        print("Saved: comparison_confusion_matrices.png")



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
        joblib.dump(self.model, filename)
        logging.info(f"Model saved to {filename}")

    def load_model(self, filename = 'model.pkl'):
        """
        Tải mô hình đã lưu từ file .pkl.

        Parameters

            filename : str
                File mô hình.

        Nếu file không tồn tại, log sẽ thông báo.
        """
        try: 
            self.model = joblib.load(filename)
            logging.info(f"Model loaded from {filename}")
        except FileNotFoundError:
            logging.info(f"File {filename} not found.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CatBoost Model")
    
    # Add arguments
    parser.add_argument('--config', type=str, default='config.ini', help='Path to configuration file')
    parser.add_argument('--tune', action='store_true', help='Flag to run hyperparameter tuning')
    parser.add_argument('--model', type=str, default='CatBoost', 
                        choices=['CatBoost', 'LightGBM', 'XGBoost' , 'RandomForest', 'Auto'], 
                        help='Which algorithm to use')

    # Parse arguments
    args = parser.parse_args()

    # workflow
    trainer = ModelTrainer(config_path=args.config)
    trainer.load_data()
    
    if args.tune:
        if args.model == 'Auto':
            trainer.auto_select_model()
            trainer.plot_evaluation_results()
        else:
            print(f"Starting hyperparameter tuning for {args.model}...")
            trainer.optimize_params(model_name=args.model)
        trainer.train_predict()
        trainer.save_model()
    else:
        print("No action selected. Use --tune to train the model.") 