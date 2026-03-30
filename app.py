from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("model.pkl")

# 🔥 Add your model accuracy here (from train.py output)
MODEL_ACCURACY = 0.87   # change this value

# Function to handle empty inputs
def get_value(val):
    return float(val) if val != "" else 0

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Correct feature order
        features = [
            get_value(data.get('age')),
            get_value(data.get('sex')),
            get_value(data.get('cp')),
            get_value(data.get('trestbps')),
            get_value(data.get('chol')),
            get_value(data.get('fbs')),
            get_value(data.get('restecg')),
            get_value(data.get('thalach')),
            get_value(data.get('exang')),
            get_value(data.get('oldpeak')),
            get_value(data.get('slope')),
            get_value(data.get('ca')),
            get_value(data.get('thal'))
        ]

        # Convert to numpy array
        prediction = model.predict([features])

        # 🔥 Add probability (confidence)
        prob = model.predict_proba([features])[0][1]

        return jsonify({
            'result': int(prediction[0]),
            'probability': float(prob),      # 🔥 new
            'accuracy': MODEL_ACCURACY       # 🔥 new
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)