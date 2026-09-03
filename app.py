from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import traceback
import os

app = Flask(__name__)
CORS(app)

# --- Tải model và preprocessor ---
model = None
preprocessor = None

try:
    with open('diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Model load error: {e}")

try:
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    print("✅ Preprocessor loaded")
except Exception as e:
    print(f"❌ Preprocessor load error: {e}")

# --- Route chính (kiểm tra) ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Diabetes Prediction API is running",
        "endpoints": {
            "/predict": "POST (form-urlencoded)",
            "/health": "GET"
        }
    })

# --- Route dự đoán ---
@app.route('/predict', methods=['POST'])
def predict():
    # Kiểm tra model đã tải
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Đọc dữ liệu từ form
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        hbA1c_level = float(request.form['hbA1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])
        gender = request.form['gender']
        smoking_history = request.form['smoking_history']
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        race = request.form['race']

        # Tạo DataFrame đúng cấu trúc
        input_data = {
            'age': [age],
            'bmi': [bmi],
            'hbA1c_level': [hbA1c_level],
            'blood_glucose_level': [blood_glucose_level],
            'gender': [gender],
            'smoking_history': [smoking_history],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'race': [race]
        }
        df = pd.DataFrame(input_data)

        # Tiền xử lý
        X_processed = preprocessor.transform(df)

        # Dự đoán
        pred = model.predict(X_processed)[0]
        prob = model.predict_proba(X_processed)[0][1]

        return jsonify({
            'prediction': int(pred),
            'probability': round(float(prob), 4)
        })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# --- Route kiểm tra sức khỏe ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

# --- Chạy server ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)