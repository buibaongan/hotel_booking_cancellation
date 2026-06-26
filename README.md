# **FINAL PROJECT**
**COURSE: PYTHON FOR DATA SCIENCE**

**Course code:** MTH10605

**TOPIC: HOTEL BOOKING CANCELLATION PREDICTION**

## I. Introduction
This project builds a Machine Learning model to predict whether a customer will cancel a hotel booking. The goal is to help hotels:
* Reduce the risk of unexpected vacant rooms
* Optimize revenue
* Make booking policies more proactive

**Dataset**: `hotel_bookings.csv`
* **Input**: Customer information, room type, booking timing, deposit details, and related fields.
* **Output**: `0` (Not canceled) or `1` (Canceled).

## II. Environment Setup
**1. System requirements**
* Python 3.8 or later.
* RAM: 8 GB or more is recommended.

**2. Install dependencies**
Run the following command in Terminal or Command Prompt:
```bash
pip install -r requirements.txt
```
***Note:*** Make sure you are in the **MY_PROJECT** directory that contains `requirements.txt`.

## III. Configuration (`config.ini`)
* The project uses `config.ini` in the **MY_PROJECT** folder to manage parameters.
* ***Note:*** Change `inputpath`, `trainpath`, and `testpath` if your data is stored somewhere else.
* **Example**
```ini
[PREPROCESSING]
inputpath = data/raw/hotel_bookings.csv

[DATA]
trainpath = data/processed/train_processed.csv
testpath = data/processed/test_processed.csv
target = is_canceled
```

## IV. Project Workflow
Run the steps below in order to make sure the program works correctly.

**Step 1: Exploratory Data Analysis (EDA)**
1. Open a terminal at the project root.
2. Run `python src/generate_EDA_report.py`.
3. **Result**: Open `reports/FULL_EDA_REPORT.html` to view the detailed EDA report.

**Step 2: Data Preprocessing**
Clean the data, encode categorical variables, and split the data into train/test sets.
* Run:
```bash
python main.py --preprocess-only
```
* The processed train/test files are saved to:
  * `MY_PROJECT/data/processed/train_processed.csv`
  * `MY_PROJECT/data/processed/test_processed.csv`

**Step 3: Training and Prediction**
You can run the pipeline without interactive input.

**A. Automatic mode (recommended)**
* The system trains all models (CatBoost, XGBoost, LightGBM, RandomForest) and selects the best model by CV score:
```bash
python main.py --model Auto --no-prompt
```

**B. Run one algorithm**
```bash
python main.py --model CatBoost --no-prompt
python main.py --model XGBoost --no-prompt
python main.py --model LightGBM --no-prompt
python main.py --model RandomForest --no-prompt
```

If you run `python main.py` without `--model`, the program asks for the model name in the terminal.

## V. Reading Result Files (`.pkl`, `.joblib`, `.json`)

`.pkl` (Pickle) and `.joblib` files are binary files that store Python objects such as models and preprocessors, so they **cannot be read directly in Notepad**.

To inspect these files in detail, follow these steps:

**Step 1:**
* Open a terminal at the project directory and run:
```bash
python check_results.py
```

**Step 2:**
* Enter the number of the file you want to inspect.
```bash
 --- RESULT FILE INSPECTION PROGRAM ---
[1] models\best_model.pkl
[2] models\evaluation_CatBoost.json
[3] models\evaluation_LightGBM.json
[4] models\evaluation_RandomForest.json
[5] models\evaluation_XGBoost.json
[6] reports\preprocessor.joblib
[0] Exit

*** Enter the file number to inspect:
```

## VI. Project Directory Structure
After the pipeline finishes, the project structure is:
```bash
MY_PROJECT/
│
├── main.py                     # Main runner
├── config.ini                  # Main configuration file
├── README.md                   # User guide
├── requirements.txt            # Dependency list
├── activity.log                # Runtime log file
├── check_results.py            # Inspect joblib, pkl, and json result files
│
├── data/                       # Data files
│   ├── raw
│   │   └── hotel_bookings.csv
│   └── processed
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── models/                     # Trained models and evaluation outputs
│   ├── best_model.pkl                  # Best model from Auto mode
│   │
│   ├── comparison_barplot.png          # Comparison chart
│   ├── comparison_confusion_matrices.png
│   ├── comparison_roc_curve.png
│   │
│   ├── evaluation_CatBoost.json        # Detailed evaluation results
│   ├── evaluation_LightGBM.json
│   ├── evaluation_RandomForest.json
│   ├── evaluation_XGBoost.json
│   │
│   ├── feature_importance_CatBoost.csv # Feature importance table
│   ├── feature_importance_LightGBM.csv
│   ├── feature_importance_RandomForest.csv
│   ├── feature_importance_XGBoost.csv
│   └── model_comparison_summary.csv    # Model comparison summary
│
├── reports/                    # Reports
│   ├── images/                 # Supporting images
│   ├── FULL_EDA_REPORT.html    # HTML report
│   ├── eda_activity.log        # EDA runtime log
│   └── preprocessor.joblib     # Saved preprocessing object
│
└── src/                        # Source code
    ├── __init__.py             # Marks src as a Python package
    ├── generate_EDA_report.py
    ├── preprocessing.py
    ├── model.py
    ├── api.py
    └── pipeline.py
```

## Common Issues
| Issue | Cause | Fix |
|:---:|:---:|:---:|
| ModuleNotFoundError | Dependencies are not installed | Run `pip install -r requirements.txt` again |
| MemoryError | The dataset is too large for available RAM | Reduce `n_jobs` to a smaller number, for example 2. Reduce `cv` to 3 in `config.ini`. |
| File not found | The path in `config.ini` is wrong | Check the `[DATA]` section in `config.ini` and make sure the file names are correct. |
| UnicodeDecodeError | The configuration file has an encoding issue | Open `config.ini`, choose Save As, and save with UTF-8 encoding. |

## VII. FastAPI Prediction API

The project includes a prediction API in `src/api.py`.

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run the API**
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Then open:
```text
http://localhost:8000/docs
```

**Health check**
```bash
curl http://localhost:8000/health
```

**Prediction with processed data**

If you only have `models/best_model.pkl` and do not have `reports/preprocessor.joblib`, send all processed features and add `processed=true`. View the feature list with:

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

**Prediction with raw booking data**

To send raw data such as `hotel`, `meal`, `country`, `adults`, and `children`, you need `reports/preprocessor.joblib`. The model and preprocessor should be created from the same pipeline run to avoid feature schema mismatches.

```bash
python main.py --model Auto --no-prompt
```

If the current model was trained with the current preprocessor, you can run only `python main.py --preprocess-only`.

Example request:
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

API response:
```json
{
  "prediction": 1,
  "label": "Canceled",
  "probability": 0.78,
  "risk_level": "High"
}
```
