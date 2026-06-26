import pandas as pd
from src.preprocessing import DataPreprocessor


def test_clean_column_names_and_missing_fill():
    df = pd.DataFrame({"Customer Type": ["A", None, "A"], "Age": [10, None, 30]})
    prep = DataPreprocessor(target_col="is_canceled")
    cleaned = prep.clean_column_names(df)
    assert "customer_type" in cleaned.columns
    assert "age" in cleaned.columns

    processed = prep.fill_missing(cleaned, num_strategy="median", cat_strategy="mode", mode="fit_transform")
    assert processed["age"].isna().sum() == 0
    assert processed["customer_type"].isna().sum() == 0


def test_transform_reuses_train_schema_for_unseen_categories():
    train = pd.DataFrame(
        {
            "hotel": ["City Hotel", "Resort Hotel"],
            "lead_time": [10, 20],
            "meal": ["BB", "HB"],
        }
    )
    test = pd.DataFrame(
        {
            "hotel": ["New Hotel"],
            "lead_time": [15],
            "meal": ["SC"],
        }
    )

    prep = DataPreprocessor(target_col="is_canceled")
    train_processed, _ = prep.fit_transform(train)
    test_processed = prep.transform(test)

    assert list(test_processed.columns) == list(train_processed.columns)
    assert "hotel_City Hotel" in train_processed.columns
    assert "hotel_New Hotel" not in test_processed.columns


def test_room_change_feature_is_created_before_dropping_assigned_room_type():
    train = pd.DataFrame(
        {
            "reserved_room_type": ["A", "A"],
            "assigned_room_type": ["A", "D"],
            "lead_time": [10, 20],
        }
    )

    prep = DataPreprocessor(target_col="is_canceled")
    train_processed, _ = prep.fit_transform(train)

    assert "is_room_changed" in train_processed.columns
    assert "assigned_room_type_A" not in train_processed.columns
