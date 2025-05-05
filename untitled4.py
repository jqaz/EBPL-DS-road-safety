"""
AI-Driven Traffic Accident Analysis – Works in Google Colab or Jupyter (no GUI needed)
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ========== Step 1: File Upload (Colab-compatible) ==========
try:
    from google.colab import files
    uploaded = files.upload()  # Prompts user to upload a CSV file
    for filename in uploaded.keys():
        csv_path = filename
except ImportError:
    csv_path = input("Enter the path to your CSV file: ")

if not os.path.exists(csv_path):
    print(f"❌ File not found: {csv_path}")
    exit(1)

print(f"\n✅ File selected: {csv_path}")

# ========== Step 2: Load Data ==========
df = pd.read_csv(csv_path)

# ========== Step 3: Preprocessing ==========
print("\n🔎 Data preview:")
print(df.head())

df.dropna(inplace=True)

required_columns = ['Weather', 'Road_Condition', 'Speed_Limit', 'Severity']
for col in required_columns:
    if col not in df.columns:
        print(f"❌ Missing column: {col}")
        exit(1)

X = df[['Weather', 'Road_Condition', 'Speed_Limit']]
X = pd.get_dummies(X)
y = df['Severity']

# ========== Step 4: Train-Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== Step 5: Train Model ==========
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ========== Step 6: Evaluate ==========
y_pred = model.predict(X_test)
print("\n📊 Model Evaluation:")
print(classification_report(y_test, y_pred))

# ========== Step 7: Save Model ==========
os.makedirs("models", exist_ok=True)
model_path = "models/traffic_model.pkl"
joblib.dump(model, model_path)
print(f"\n✅ Model saved to: {model_path}")
