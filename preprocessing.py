import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Convert diagnosis to numeric: M = 1 (malignant), B = 0 (benign)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Select features (drop ID column if present)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
