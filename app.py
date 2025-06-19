from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# --- Simulate Data (replace with actual DB or CSV input in production) ---
def generate_data(n=10000):
    np.random.seed(42)
    df = pd.DataFrame({
        'transaction_id': range(n),
        'user_id': np.random.randint(1, 1000, n),
        'amount': np.random.exponential(scale=200, size=n),
        'transaction_time': pd.date_range(start='2023-01-01', periods=n, freq='T'),
        'merchant_category': np.random.choice(['grocery', 'electronics', 'fashion', 'crypto', 'atm'], n)
    })
    return df

# --- Preprocessing ---
def preprocess(df):
    df = df.copy()
    df['hour'] = df['transaction_time'].dt.hour
    df = pd.get_dummies(df, columns=['merchant_category'], drop_first=True)
    features = ['amount', 'hour'] + [col for col in df.columns if 'merchant_category_' in col]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    return df, X

# --- ML Model ---
def detect_anomalies(X):
    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    model.fit(X)
    preds = model.predict(X)
    return preds

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    df = generate_data()
    df, X = preprocess(df)
    df['anomaly'] = detect_anomalies(X)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1 = normal, -1 = anomaly
    anomalies = df[df['anomaly'] == 1].to_dict(orient='records')
    return jsonify(anomalies)

if __name__ == '__main__':
    app.run(debug=True)