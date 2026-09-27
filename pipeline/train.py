import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json


def load_processed_data(data_dir='data/processed'):
    X_train, y_train = joblib.load(os.path.join(data_dir, 'train.pkl'))
    X_test, y_test = joblib.load(os.path.join(data_dir, 'test.pkl'))
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    return metrics


def save_artifacts(model, metrics, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed_data()

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_artifacts(model, metrics)

    print("Model trained.")
    print(metrics)