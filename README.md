# **PROJECT CUỐI KỲ**
**MÔN HỌC: PYTHON CHO KHOA HỌC DỮ LIỆU**

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
trainpath = data/processed/train.csv
testpath = data/processed/test.csv
...
# Phần còn lại giữ nguyên
```

## IV. QUY TRÌNH CHẠY DỰ ÁN
Hãy thực hiện tuần tự theo các bước sau để đảm bảo chương trình chạy đúng:

**Bước 1: Khám phá dữ liệu (EDA)**
1.  Mở folder `src`
2.  Chạy file generate_EDA_report.py
3. Kết quả: Mở file `reports/images/FULL_EDA_REPORT.html` để xem báo cáo EDA chi tiết.

**Bước 2: Tiền xử lý dữ liệu (Preprocessing)**
Làm sạch dữ liệu, mã hóa biến phân loại và chia tập Train/Test.
* Chạy file `src/main.py`
* Dữ liệu sau khi xử lý và chia train/test được lưu vào:
  * `data/processed/train.csv` 
  * `data/processed/test.csv`

**Bước 3: Huấn luyện & Dự đoán (Modeling)**
Mở Terminal tại thư mục dự án và sử dụng các lệnh sau:

**A. Chế độ Tự động (Khuyên dùng)**
* Hệ thống chạy tất cả model (Logistic, DecisionTree, RandomForest, XGBoost) và chọn cái tốt nhất:
```bash
python src/model.py --tune --model Auto
```
**B. Chạy từng thuật toán riêng lẻ**
```bash
# Random Forest
python src/model.py --tune --model RandomForest

# XGBoost
python src/model.py --tune --model XGBoost

# Logistic Regression
python src/model.py --tune --model LogisticRegression

# Decision Tree
python src/model.py --tune --model DecisionTree

```
## V. CẤU TRÚC THƯ MỤC DỰ ÁN
```bash
MY_PROJECT/
│
├── config.ini                  # File cấu hình chính (QUAN TRỌNG)
├── README.md                   # File hướng dẫn này
├── requirements.txt            # Danh sách thư viện
├── activity.log                # Nhật ký chạy (Log file)
├── model.pkl                   # File model đã train xong
│
├── data/                       # Thư mục chứa dữ liệu
│   ├── raw                     # Dữ liệu gốc
│   │   └── hotel_bookings.csv
│   └── processed
│        ├── train_processed.csv 
│        ├── test_processed.csv      
├── reports/                    # Báo cáo 
│   └── images
│        ├── FULL_EDA_REPORT.html
│        ├── model.pkl 
│        ├── final_comparison.csv
│        └── roc_comparison.png
│
└── src/                        # Mã nguồn
    ├── main.py
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
