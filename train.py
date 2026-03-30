import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 🔹 Load dataset
data = pd.read_csv("heart.csv", header=None)

# 🔹 Add column names
data.columns = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal','target'
]

# 🔹 Clean data
data = data.replace('?', 0)
data = data.astype(float)

# 🔹 Convert target (0 = Safe, 1 = Risk)
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# 🔹 Check distribution
print("Target values:\n", data['target'].value_counts())

# 🔹 Split features & label
X = data.drop('target', axis=1)
y = data['target']

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Create model (improved)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 🔹 Train model
model.fit(X_train, y_train)

# 🔹 Predict
y_pred = model.predict(X_test)

# 🔹 Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

# 🔹 Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved successfully!")