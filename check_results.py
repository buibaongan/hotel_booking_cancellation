import joblib
import json
import os
import sys

# Thêm đường dẫn hiện tại để máy hiểu (tránh lỗi nếu file nằm trong thư mục con)
sys.path.append('.')

def xem_file_pkl_joblib(duong_dan):
    print("\n" + "-"*30)
    print(f"ĐANG ĐỌC FILE: {duong_dan}")
    
    try:
        # Load file lên
        data = joblib.load(duong_dan)
        print("Đã load xong!")
        print(f"Loại dữ liệu: {type(data)}")
        
        # 1. NẾU LÀ MODEL (Có tham số)
        # Kiểm tra xem biến 'data' có hàm lấy tham số không
        if hasattr(data, 'get_params'):
            print("\n[THAM SỐ CỦA MODEL]:")
            params = data.get_params()
            print(params)

        # 2. FEATURE IMPORTANCES
        # Kiểm tra xem model có tính năng này không
        if hasattr(data, 'feature_importances_'):
            print("\n[TOP 5 ĐẶC TRƯNG QUAN TRỌNG]:")
            
            # Lấy danh sách điểm số
            scores = data.feature_importances_
            
            # Cố tìm tên cột (nếu có lưu)
            feature_names = []
            if hasattr(data, 'feature_names_in_'):
                feature_names = data.feature_names_in_
            elif hasattr(data, 'feature_names_'):
                feature_names = data.feature_names_
            
            # Tạo danh sách các cặp (tên, điểm) để sắp xếp
            danh_sach = []
            for i in range(len(scores)):
                diem = scores[i]
                # Nếu có tên cột thì lấy, không thì đặt là Feature_0, Feature_1...
                if len(feature_names) > i:
                    ten = feature_names[i]
                else:
                    ten = f"Feature_{i}"
                danh_sach.append((ten, diem))
            
            # Sắp xếp danh sách theo điểm giảm dần
            danh_sach_sap_xep = sorted(danh_sach, key=lambda x: x[1], reverse=True)
            
            # In ra 5 cái đầu tiên
            for i in range(5):
                if i < len(danh_sach_sap_xep):
                    print(f"  {i+1}. {danh_sach_sap_xep[i][0]}: {danh_sach_sap_xep[i][1]:.4f}")

        # 3. NẾU LÀ OBJECT KHÁC (Pipeline)
        # Dùng hàm vars() để xem các biến bên trong
        print("\n[THÔNG TIN KHÁC]:")
        try:
            thong_tin = vars(data)
            for ten, gia_tri in thong_tin.items():
                print(f"- {ten}: {gia_tri}")
        except:
            print("Không đọc được chi tiết biến bên trong.")

    except Exception as loi:
        print(f"Có lỗi xảy ra: {loi}")

def xem_file_json(duong_dan):
    print("\n" + "-"*30)
    print(f"ĐANG ĐỌC FILE JSON: {duong_dan}")
    
    try:
        with open(duong_dan, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("--- NỘI DUNG ---")
        # Duyệt qua từng dòng trong json
        for key in data:
            value = data[key]
            
            # Nếu là danh sách dài quá thì chỉ in số lượng
            if type(value) == list:
                print(f"- {key}: (Danh sách có {len(value)} phần tử)")
            
            # Nếu là từ điển con (dict) thì in chi tiết hơn xíu
            elif type(value) == dict:
                print(f"- {key}:")
                for k_con, v_con in value.items():
                    print(f"   + {k_con}: {v_con}")
            
            # Còn lại in bình thường
            else:
                print(f"- {key}: {value}")
                
    except Exception as loi:
        print(f"Lỗi đọc JSON: {loi}")

def lay_danh_sach_file():
    ds_file = []
    cac_thu_muc = ['models', 'reports']
    
    for thu_muc in cac_thu_muc:
        # Kiểm tra thư mục có tồn tại không
        if os.path.exists(thu_muc):
            # Lấy tất cả file trong thư mục
            file_trong_folder = os.listdir(thu_muc)
            for ten_file in file_trong_folder:
                # Chỉ lấy đúng đuôi file mình cần
                if ten_file.endswith('.pkl') or ten_file.endswith('.joblib') or ten_file.endswith('.json'):
                    # Ghép tên thư mục với tên file (ví dụ: models/model.pkl)
                    duong_dan_day_du = os.path.join(thu_muc, ten_file)
                    ds_file.append(duong_dan_day_du)
    return ds_file

def main():
    while True:
        print("\nMENU KIỂM TRA FILE")
        
        danh_sach = lay_danh_sach_file()
        
        if len(danh_sach) == 0:
            print("Không tìm thấy file!")
            break

        # In danh sách ra màn hình
        for i in range(len(danh_sach)):
            print(f"[{i + 1}] {danh_sach[i]}")
        print("[0] Thoát")

        chon = input("\nNhập số file muốn xem: ")

        if chon == '0':
            print("KẾT THÚC CHƯƠNG TRÌNH!")
            break
        
        try:
            so_thu_tu = int(chon) - 1
            
            if so_thu_tu >= 0 and so_thu_tu < len(danh_sach):
                file_duoc_chon = danh_sach[so_thu_tu]
                
                # Kiểm tra đuôi file để gọi hàm đúng
                if file_duoc_chon.endswith('.json'):
                    xem_file_json(file_duoc_chon)
                else:
                    xem_file_pkl_joblib(file_duoc_chon)
                
                input("\n(Ấn Enter để tiếp tục...)")
            else:
                print("Số nhập không đúng!")
        except:
            print("Vui lòng nhập số!")

if __name__ == "__main__":
    main()

    # Chạy lệnh sau trong terminal
    # python check_results.py
