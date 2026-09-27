import pandas as pd
import os

def load_raw_data(fake_path='data/raw/Fake.csv', true_path='data/raw/True.csv'):
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    fake['label'] = 0   # 0 = fake
    true['label'] = 1   # 1 = real

    df = pd.concat([fake, true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    os.makedirs('data/raw', exist_ok=True)
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print("Total data:", df.shape)
    print(df['label'].value_counts())