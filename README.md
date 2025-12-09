# **PROJECT CUỐI KỲ**
**MÔN HỌC: PYTHON CHO KHOA HỌC DỮ LIỆU**

**Mã học phần:** MTH10605

**CHỦ ĐỀ: DỰ ĐOÁN KHẢ NĂNG HỦY ĐẶT PHÒNG KHÁCH SẠN**
*(Hotel Booking Cancellation Prediction)*

## I. GIỚI THIỆU
Dự án xây dựng mô hình Machine Learning nhằm dự đoán khách hàng có hủy đặt phòng hay không, hỗ trợ khách sạn:
* Giảm rủi ro phòng trống đột xuất
* Tối ưu doanh thu
* Chủ động chính sách đặt phòng

**Dataset**: `hotel_bookings.csv`
* **Input**: Thông tin khách hàng, loại phòng, thời gian đặt, tiền cọc...
* **Output**: `0` (Không hủy) hoặc `1` (Hủy).

## II. CÀI ĐẶT MÔI TRƯỜNG
**1. Yêu cầu hệ thống**
* Python 3.8 trở lên.
* RAM: Khuyến nghị 8GB trở lên.

**2. Cài đặt thư viện**
Chạy lệnh sau trong Terminal/Command Prompt để cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```
***Lưu ý:*** Đảm bảo bạn đang đứng ở thư mục **MY_PROJECT** chứa file `requirements.txt`.

## III. CẤU HÌNH (Config.ini)
* Dự án sử dụng file `config.ini` có sẵn trong folder **MY_PROJECT** để quản lý mọi tham số.
* ***Lưu ý***:  Thay đổi đường dẫn `path` trong file bằng đường dẫn trong máy bạn để chạy cho đúng 
* **Ví dụ**
```ini
[DATA]
# Đường dẫn đến file dữ liệu
[PREPROCESSING]
inputpath =  D:\Program Files\MY_PROJECT\data\raw\hotel_bookings.csv

trainpath = D:\Program Files\MY_PROJECT\data\processed\train_processed.csv
testpath = D:\Program Files\MY_PROJECT\data\processed\test_processed.csv
...
# Phần còn lại giữ nguyên
```

## IV. QUY TRÌNH CHẠY DỰ ÁN
Hãy thực hiện tuần tự theo các bước sau để đảm bảo chương trình chạy đúng:

**Bước 1: Khám phá dữ liệu (EDA)**
1. Mở folder `src`
2. Chạy file `generate_EDA_report.py`
3. **Kết quả**: Mở file `reports/images/FULL_EDA_REPORT.html` để xem báo cáo EDA chi tiết.

**Bước 2: Tiền xử lý dữ liệu (Preprocessing)**
Làm sạch dữ liệu, mã hóa biến phân loại và chia tập Train/Test.
* Chạy file `src/main.py`
* Dữ liệu sau khi xử lý và chia train/test được lưu vào:
  * `MY_PROJECT/data/processed/train.csv` 
  * `MY_PROJECT/data/processed/test.csv`

**Bước 3: Huấn luyện & Dự đoán (Modeling)**
Trong Terminal, gõ các lệnh sau:
1. Gõ `model` để tiếp tục chạy model (Gõ `exit` để thoát)
2. Gõ tên model muốn chạy

**A. Chế độ Tự động (Khuyên dùng)**
* Hệ thống chạy tất cả model (CatBoost, XGBoost, LightGBM, RandomForest) và chọn cái tốt nhất:
```bash
Auto hoặc auto
```
**B. Chạy từng thuật toán riêng lẻ**
```bash
# CatBoost
CatBoost hoặc catboost hoặc cat

# XGBoost
XGBoost  hoặc xgboost hoặc xgb

# LightGBM
LightGBM hoặc lightgbm hoặc lgbm

# RandomForest
RandomForest hoặc randomforest hoặc rf
```

## V. HƯỚNG DẪN ĐỌC CÁC FILE KẾT QUẢ (.pkl, .joblib, .json)

Các file `.pkl` (Pickle) và `.joblib` là dạng file nhị phân lưu trữ object của Python (Model, Preprocessor), do đó **không thể mở xem trực tiếp bằng Notepad** (sẽ bị lỗi font).

Để xem nội dung các file này một cách chi tiết, hãy thực hiện theo các bước sau:

**Bước 1: Khởi tạo môi trường ảo**

Mở Terminal tại thư mục gốc của dự án (`MY_PROJECT`), chạy lệnh:

```bash
python -m venv venv
```

**Bước 2:** Kích hoạt môi trường:
 * Với Windows (PowerShell / VS Code):
```bash
.\venv\Scripts\activate
```
* Với macOS / Linux:
```bash
source venv/bin/activate
```
*Sau khi kích hoạt, đầu dòng lệnh Terminal sẽ xuất hiện chữ màu xanh:* **(venv)**



**Bước 3:** 
* Mở file `check_results.py` tại thư mục gốc của dự án (`MY_PROJECT/`)

* Mở Terminal (phím tắt: `ctrl + ~`) tại thư mục dự án và chạy lệnh:
```bash
python check_results.py
```
**Bước 4:** 
*  Nhập số thứ tự file muốn xem
```bash
========================================
 CHƯƠNG TRÌNH KIỂM TRA FILE KẾT QUẢ
========================================
[1] models\evaluation_CatBoost.json
[2] models\evaluation_LightGBM.json
[3] models\evaluation_RandomForest.json
[4] models\evaluation_XGBoost.json
[5] reports\preprocessor.joblib
[0] Thoát

```


## VI. CẤU TRÚC THƯ MỤC DỰ ÁN
* Sau khi chạy xong, thư mục dự án có cấu trúc như sau:
```bash
MY_PROJECT/
│
├── main.py                     # Hàm chạy chính
├── config.ini                  # File cấu hình chính
├── README.md                   # File hướng dẫn
├── requirements.txt            # Danh sách thư viện
├── activity.log                # Nhật ký chạy (Log file)
├── check_results.py            # Xem kết quả các file joblib, pkl, json
│
├── ven                         # Môi trường ven để chạy file check_results.py
│
├── data/                       # Dữ liệu
│   ├── raw
│   │   └── hotel_bookings.csv
│   └── processed
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── models/                     # Chứa Model và Kết quả đánh giá 
│   ├── model.pkl                       # File model chính đã train
│   │
│   ├── comparison_barplot.png          # Biểu đồ so sánh
│   ├── comparison_confusion_matrices.png
│   ├── comparison_roc_curve.png
│   │
│   ├── evaluation_CatBoost.json        # Kết quả đánh giá chi tiết (JSON)
│   ├── evaluation_LightGBM.json
│   ├── evaluation_RandomForest.json
│   ├── evaluation_XGBoost.json
│   │
│   ├── feature_importance_CatBoost.csv # Mức độ quan trọng của biến (CSV)
│   ├── feature_importance_LightGBM.csv
│   ├── feature_importance_RandomForest.csv
│   ├── feature_importance_XGBoost.csv
│   └── model_comparison_summary.csv    # Bảng tổng hợp so sánh
│
├── reports/                    # Báo cáo 
│   ├── images/                 # Thư mục chứa ảnh bổ trợ 
│   ├── FULL_EDA_REPORT.html    # Báo cáo HTML
│   ├── eda_activity            # Nhật ký chạy EDA (log file)
│   └── preprocessor.joblib     # File xử lý dữ liệu
│
└── src/                        # Mã nguồn
    ├── __init__.py             # File rỗng báo Python biết src là 1 package
    ├── generate_EDA_report.py
    ├── preprocessing.py
    └── model.py
```

## SỰ CỐ THƯỜNG GẶP
|        Vấn đề       |          Nguyên nhân         |                           Cách khắc phục                          |
|:-------------------:|:----------------------------:|:-----------------------------------------------------------------:|
| ModuleNotFoundError | Chưa cài thư viện            | Chạy lại lệnh: pip install -r requirements.txt                    |
| Lỗi MemoryError     | Dữ liệu quá lớn gây tràn RAM | Giảm n_jobs xuống số nhỏ (ví dụ 2). Giảm cv xuống 3 trong config. |
| File not found      | Sai đường dẫn trong config   | Kiểm tra lại mục [DATA] trong config.ini, đảm bảo tên file đúng.  |
| UnicodeDecodeError  | File config bị lỗi font chữ  | Mở file config.ini, chọn Save As -> Encoding: UTF-8.              |


