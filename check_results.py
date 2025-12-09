import joblib
import json
import os
import sys
import numpy as np

# Thêm đường dẫn để load được class tự định nghĩa trong src (nếu cần)
sys.path.append(os.getcwd())

def read_pkl_or_joblib(path):
    """
    Hàm đọc chung cho cả .pkl và .joblib (SỬ DỤNG JOBLIB.LOAD).
    Kết hợp logic hiển thị của cả 2 hàm cũ.
    """
    print(f"\n--- ĐANG ĐỌC FILE (JOBLIB/PKL): {path} ---")
    try:
        # --- DÙNG JOBLIB ĐỂ LOAD ---
        content = joblib.load(path)
        
        print("--- LOAD THÀNH CÔNG ---")
        print(f"Loại object: {type(content)}")
        print("\n--- CHI TIẾT ---")

        # Nếu object có hàm get_params (thường là model sklearn/xgboost...), in tham số ra
        if hasattr(content, 'get_params'):
            print("\nTHAM SỐ MODEL:")
            print(content.get_params())

        # --- IN TOP 5 FEATURE IMPORTANCE ---
        if hasattr(content, 'feature_importances_'):
            print("\nTOP 5 FEATURE IMPORTANCE:")
            try:
                importances = content.feature_importances_
                indices = np.argsort(importances)[::-1] # Sắp xếp giảm dần
                
                # Cố gắng tìm tên cột (Feature Names)
                feature_names = None
                
                # Sklearn thường lưu trong feature_names_in_
                if hasattr(content, 'feature_names_in_'):
                    feature_names = content.feature_names_in_
                # CatBoost/XGBoost có thể lưu trong feature_names_
                elif hasattr(content, 'feature_names_'):
                    feature_names = content.feature_names_
                
                # In ra Top 5
                print(f" (Tổng số features: {len(importances)})")
                for i in range(min(5, len(importances))):
                    idx = indices[i]
                    score = importances[idx]
                    
                    if feature_names is not None and len(feature_names) > idx:
                        name = feature_names[idx]
                        print(f"  {i+1}. {name}: {score:.4f}")
                    else:
                        print(f"  {i+1}. Feature_{idx}: {score:.4f}")
            except Exception as e:
                print(f"  -> Không thể trích xuất (Lỗi: {e})")

        # NẾU LÀ CLASS TỰ ĐỊNH NGHĨA / PIPELINE (Dùng vars)
        # Dùng hàm vars() để liệt kê các thuộc tính
        try:
            attributes = vars(content)
            if attributes:
                print("\nCÁC THUỘC TÍNH:")
                for key, value in attributes.items():
                    print(f"[Thuộc tính]: {key}")
                    print(f"  -> Giá trị: {value}")
                    
                    # xem ColumnTransformer
                    if hasattr(value, 'transformers_'): 
                        print("  -> (Đây là ColumnTransformer, các bước xử lý:)")
                        for name, trans, cols in value.transformers_:
                            print(f"      + Bước '{name}': áp dụng lên {cols}")
            else:
                # Nếu không có vars (ví dụ list, dict thuần), in trực tiếp
                if not hasattr(content, 'get_params'): # Tránh in lại nếu đã in params
                    print(f"Nội dung: {content}")

        except TypeError:
            pass

    except Exception as e:
        print(f"Lỗi: {e}")

def read_json(path):
    print(f"\n--- ĐANG ĐỌC FILE JSON: {path} ---")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        print("--- TÓM TẮT NỘI DUNG ---")
        for key, value in content.items():
            if isinstance(value, list):
                print(f"- {key}: Danh sách gồm {len(value)} phần tử (Đã ẩn chi tiết)")
            elif isinstance(value, dict):
                print(f"- {key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        print(f"  + {sub_key}: Danh sách {len(sub_value)} phần tử")
                    else:
                        print(f"  + {sub_key}: {sub_value}")
            else:
                print(f"- {key}: {value}")
                
    except Exception as e:
        print("Lỗi:", e)

def get_list_files():
    files = []
    folders_to_scan = ['models', 'reports']
    
    for folder in folders_to_scan:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(('.pkl', '.joblib', '.json')):
                    files.append(os.path.join(folder, f))
    return files

def main():
    while True:
        print(" --- CHƯƠNG TRÌNH KIỂM TRA FILE KẾT QUẢ ---")
        
        files = get_list_files()
        
        if not files:
            print("*** Không tìm thấy file nào trong thư mục models/ hoặc reports/")
            break

        for i, file_path in enumerate(files):
            print(f"[{i + 1}] {file_path}")
        print("[0] Thoát")

        choice = input("\n*** Nhập số thứ tự file muốn xem: ")

        if choice == '0':
            print("Đã thoát chương trình.")
            break
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                selected_file = files[idx]
                
                # cùng chạy file .pkl và .joblib
                if selected_file.endswith(('.joblib', '.pkl')):
                    read_pkl_or_joblib(selected_file)
                elif selected_file.endswith('.json'):
                    read_json(selected_file)
                
                input("\n(Nhấn Enter để quay lại menu...)")
            else:
                print("\n*** Lỗi: Số không hợp lệ!")
        except ValueError:
            print("\n*** Lỗi: Vui lòng chỉ nhập số!")

if __name__ == "__main__":
    main()

    # Chạy lệnh sau trong terminal
    # python check_results.py