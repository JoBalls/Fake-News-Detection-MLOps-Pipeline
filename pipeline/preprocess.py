import pandas as pd
import re
import string
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from load_data import load_raw_data


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)                    
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) 
    text = re.sub(r'\d+', '', text)                         
    text = re.sub(r'\s+', ' ', text).strip()                
    return text


def preprocess(df):
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 0]  
    return df


def split_and_vectorize(df, save_dir='data/processed'):
    os.makedirs(save_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    joblib.dump(vectorizer, os.path.join(save_dir, 'tfidf_vectorizer.pkl'))
    joblib.dump((X_train_vec, y_train), os.path.join(save_dir, 'train.pkl'))
    joblib.dump((X_test_vec, y_test), os.path.join(save_dir, 'test.pkl'))

    return X_train_vec, X_test_vec, y_train, y_test


if __name__ == "__main__":
    df = load_raw_data()
    df = preprocess(df)
    df.to_csv('data/processed/clean_dataset.csv', index=False)

    X_train, X_test, y_train, y_test = split_and_vectorize(df)
    print("Data preprocessed.")
    print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)