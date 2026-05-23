import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("heart.csv", header=None)

data.columns = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal','target'
]

# Convert target
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# 1. Target Distribution
data['target'].value_counts().plot(kind='bar')
plt.title("Heart Disease Distribution")
plt.xlabel("0 = Safe, 1 = Risk")
plt.ylabel("Count")
plt.show()

# 2. Age vs Target
plt.scatter(data['age'], data['target'])
plt.title("Age vs Heart Disease")
plt.xlabel("Age")
plt.ylabel("Target")
plt.show()

# 3. Cholesterol vs Target
plt.scatter(data['chol'], data['target'])
plt.title("Cholesterol vs Heart Disease")
plt.xlabel("Cholesterol")
plt.ylabel("Target")
plt.show()