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
* ***Lưu ý***: Thay đổi đường dẫn `inputpath`, `trainpath`, `testpath` nếu bạn đặt dữ liệu ở vị trí khác.
* **Ví dụ**
```ini
[PREPROCESSING]
inputpath = data/raw/hotel_bookings.csv

[DATA]
trainpath = data/processed/train_processed.csv
testpath = data/processed/test_processed.csv
target = is_canceled
```

## IV. QUY TRÌNH CHẠY DỰ ÁN
Hãy thực hiện tuần tự theo các bước sau để đảm bảo chương trình chạy đúng:

**Bước 1: Khám phá dữ liệu (EDA)**
1. Đứng ở thư mục gốc dự án.
2. Chạy `python src/generate_EDA_report.py`.
3. **Kết quả**: Mở file `reports/FULL_EDA_REPORT.html` để xem báo cáo EDA chi tiết.

**Bước 2: Tiền xử lý dữ liệu (Preprocessing)**
Làm sạch dữ liệu, mã hóa biến phân loại và chia tập Train/Test.
* Chạy:
```bash
python main.py --preprocess-only
```
* Dữ liệu sau khi xử lý và chia train/test được lưu vào:
  * `MY_PROJECT/data/processed/train_processed.csv`
  * `MY_PROJECT/data/processed/test_processed.csv`

**Bước 3: Huấn luyện & Dự đoán (Modeling)**
Có thể chạy không cần nhập tương tác:

**A. Chế độ Tự động (Khuyên dùng)**
* Hệ thống chạy tất cả model (CatBoost, XGBoost, LightGBM, RandomForest) và chọn model tốt nhất theo CV score:
```bash
python main.py --model Auto --no-prompt
```
**B. Chạy từng thuật toán riêng lẻ**
```bash
python main.py --model CatBoost --no-prompt
python main.py --model XGBoost --no-prompt
python main.py --model LightGBM --no-prompt
python main.py --model RandomForest --no-prompt
```

Nếu chạy `python main.py` không truyền `--model`, chương trình sẽ hỏi tên model trong terminal.

## V. HƯỚNG DẪN ĐỌC CÁC FILE KẾT QUẢ (.pkl, .joblib, .json)

Các file `.pkl` (Pickle) và `.joblib` là dạng file nhị phân lưu trữ object của Python (Model, Preprocessor), do đó **không thể mở xem trực tiếp bằng Notepad** (sẽ bị lỗi font).

Để xem nội dung các file này một cách chi tiết, hãy thực hiện theo các bước sau:

**Bước 1:**
* Mở Terminal (phím tắt: `ctrl + ~`) tại thư mục dự án và chạy lệnh:
```bash
python check_results.py
```
**Bước 2:** 
*  Nhập số thứ tự file muốn xem
```bash
 --- CHƯƠNG TRÌNH KIỂM TRA FILE KẾT QUẢ ---
[1] models\best_model.pkl
[2] models\evaluation_CatBoost.json
[3] models\evaluation_LightGBM.json
[4] models\evaluation_RandomForest.json
[5] models\evaluation_XGBoost.json
[6] reports\preprocessor.joblib
[0] Thoát

*** Nhập số thứ tự file muốn xem: 

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
├── data/                       # Dữ liệu
│   ├── raw
│   │   └── hotel_bookings.csv
│   └── processed
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── models/                     # Chứa Model và Kết quả đánh giá 
│   ├── best_model.pkl                  # File model tốt nhất khi chạy Auto
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
│   ├── eda_activity.log        # Nhật ký chạy EDA
│   └── preprocessor.joblib     # File xử lý dữ liệu
│
└── src/                        # Mã nguồn
    ├── __init__.py             # File rỗng báo Python biết src là 1 package
    ├── generate_EDA_report.py
    ├── preprocessing.py
    ├── model.py
    ├── api.py
    └── pipeline.py
```

## SỰ CỐ THƯỜNG GẶP
|        Vấn đề       |          Nguyên nhân         |                           Cách khắc phục                          |
|:-------------------:|:----------------------------:|:-----------------------------------------------------------------:|
| ModuleNotFoundError | Chưa cài thư viện            | Chạy lại lệnh: pip install -r requirements.txt                    |
| Lỗi MemoryError     | Dữ liệu quá lớn gây tràn RAM | Giảm n_jobs xuống số nhỏ (ví dụ 2). Giảm cv xuống 3 trong config. |
| File not found      | Sai đường dẫn trong config   | Kiểm tra lại mục [DATA] trong config.ini, đảm bảo tên file đúng.  |
| UnicodeDecodeError  | File config bị lỗi font chữ  | Mở file config.ini, chọn Save As -> Encoding: UTF-8.              |

## VII. FASTAPI PREDICTION API

Dự án có API dự đoán trong `src/api.py`.

**Cài thư viện**
```bash
pip install -r requirements.txt
```

**Chạy API**
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Sau đó mở:
```text
http://localhost:8000/docs
```

**Health check**
```bash
curl http://localhost:8000/health
```

**Dự đoán với dữ liệu đã xử lý**

Nếu chỉ có `models/best_model.pkl` mà chưa có `reports/preprocessor.joblib`, hãy gửi đầy đủ các feature đã xử lý và thêm `processed=true`. Xem danh sách feature bằng:

```bash
curl http://localhost:8000/metadata
```

```bash
curl -X POST "http://localhost:8000/predict?processed=true" \
  -H "Content-Type: application/json" \
  -d '{
    "booking": {
      "hotel": 0,
      "lead_time": 0.12,
      "arrival_date_week_number": 0.5,
      "arrival_date_day_of_month": 0.3,
      "is_repeated_guest": 0,
      "previous_cancellations": 0,
      "previous_bookings_not_canceled": 0,
      "booking_changes": 0,
      "deposit_type": 0,
      "days_in_waiting_list": 0,
      "adr": 0.02,
      "required_car_parking_spaces": 0,
      "total_of_special_requests": 0.2,
      "total_nights": 0.05,
      "total_guests": 0.04,
      "has_children": 0,
      "is_domestic": 1,
      "is_room_changed": 0
    }
  }'
```

**Dự đoán với dữ liệu booking thô**

Muốn gửi dữ liệu thô như `hotel`, `meal`, `country`, `adults`, `children`..., cần có `reports/preprocessor.joblib`. Model và preprocessor nên được tạo từ cùng một lần chạy pipeline để tránh lệch schema feature.

```bash
python main.py --model Auto --no-prompt
```

Nếu model hiện tại đã được train đúng với preprocessor hiện tại, có thể chỉ chạy `python main.py --preprocess-only`.

Ví dụ request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "booking": {
      "hotel": "City Hotel",
      "lead_time": 120,
      "arrival_date_month": "August",
      "arrival_date_week_number": 32,
      "arrival_date_day_of_month": 12,
      "stays_in_weekend_nights": 1,
      "stays_in_week_nights": 3,
      "adults": 2,
      "children": 0,
      "babies": 0,
      "meal": "BB",
      "country": "PRT",
      "market_segment": "Online TA",
      "distribution_channel": "TA/TO",
      "reserved_room_type": "A",
      "assigned_room_type": "A",
      "deposit_type": "No Deposit",
      "customer_type": "Transient",
      "is_repeated_guest": 0,
      "previous_cancellations": 0,
      "previous_bookings_not_canceled": 0,
      "booking_changes": 0,
      "days_in_waiting_list": 0,
      "adr": 100,
      "required_car_parking_spaces": 0,
      "total_of_special_requests": 1
    }
  }'
```

API trả về:
```json
{
  "prediction": 1,
  "label": "Canceled",
  "probability": 0.78,
  "risk_level": "High"
}
```
