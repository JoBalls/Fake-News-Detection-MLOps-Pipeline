import os
import sys
import time

from load_data import load_raw_data
from preprocess import preprocess, split_and_vectorize
from train import train_model, evaluate_model, save_artifacts

RAW_FILES = ['data/raw/Fake.csv', 'data/raw/True.csv']


def check_inputs():
    missing = [p for p in RAW_FILES if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Dataset not found: {', '.join(missing)}. "
            "Place Fake.csv and True.csv in the data/raw/ folder."
        )


def run():
    start = time.time()

    print("[1/4] Loading raw data...")
    check_inputs()
    df = load_raw_data()
    print(f"      {len(df)} rows loaded")

    print("[2/4] Preprocessing and feature engineering...")
    df = preprocess(df)
    X_train, X_test, y_train, y_test = split_and_vectorize(df)

    print("[3/4] Training model...")
    model = train_model(X_train, y_train)

    print("[4/4] Evaluating and saving artifacts...")
    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, metrics)

    print(f"Pipeline finished in {time.time() - start:.1f}s")
    print(metrics)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)