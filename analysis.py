import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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

# 🔹 Convert target
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# 🔹 Split data
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 Predictions
y_pred = model.predict(X_test)

# 🔹 Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")

# ================================
# 📊 1. TARGET DISTRIBUTION GRAPH
# ================================
sns.countplot(x='target', data=data)
plt.title("Target Distribution (0=Safe, 1=Risk)")
plt.show()

# ================================
# 📊 2. AGE VS TARGET
# ================================
plt.scatter(data['age'], data['target'])
plt.title("Age vs Heart Disease")
plt.xlabel("Age")
plt.ylabel("Target")
plt.show()

# ================================
# 📉 3. CONFUSION MATRIX
# ================================
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Confusion Matrix:\n", cm)

# ================================
# 📄 4. CLASSIFICATION REPORT
# ================================
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)