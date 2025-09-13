import joblib, pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression

def train_and_save(df: pd.DataFrame, target: str, path: str = "model.joblib"):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[target].astype(float).values
    model = LinearRegression().fit(X, y)
    joblib.dump(model, path)
    print(f"Model saved to {path}")
