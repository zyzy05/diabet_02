from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
import traceback

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

# --- Route chính ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Diabetes Prediction API is running"})

# --- Route dự đoán ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Lấy dữ liệu từ form
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        hbA1c_level = float(request.form['hbA1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])
        gender = request.form['gender']
        smoking_history = request.form['smoking_history']
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        race = request.form['race']

        # --- Tạo DataFrame với đúng cột như khi huấn luyện ---
        # Các cột số và các cột phân loại còn lại giữ nguyên (gender, smoking_history, ...)
        # Riêng race phải được one‑hot encode thành các cột riêng
        input_data = {
            'age': [age],
            'bmi': [bmi],
            'hbA1c_level': [hbA1c_level],
            'blood_glucose_level': [blood_glucose_level],
            'gender': [gender],
            'smoking_history': [smoking_history],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            # Tạo 5 cột race one‑hot
            'raceAfricanAmerican': [1 if race == 'AfricanAmerican' else 0],
            'raceAsian': [1 if race == 'Asian' else 0],
            'raceCaucasian': [1 if race == 'Caucasian' else 0],
            'raceHispanic': [1 if race == 'Hispanic' else 0],
            'raceOther': [1 if race == 'Other' else 0]
        }

        df = pd.DataFrame(input_data)

        # In ra cấu trúc để debug (nếu cần)
        app.logger.info(f"Input columns: {df.columns.tolist()}")
        app.logger.info(f"Input shape: {df.shape}")

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
        return jsonify({'error': f'Thiếu trường dữ liệu: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Giá trị không hợp lệ: {str(e)}'}), 400
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)