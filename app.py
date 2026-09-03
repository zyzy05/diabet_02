from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import traceback
import os

app = Flask(__name__)
CORS(app)  # Cho phép mọi nguồn gốc

# =============================================
# TẢI MODEL VÀ PREPROCESSOR
# =============================================
MODEL_PATH = 'diabetes_model.pkl'
PREPROCESSOR_PATH = 'preprocessor.pkl'

model = None
preprocessor = None

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Không tìm thấy file {MODEL_PATH}")
except Exception as e:
    print(f"❌ Lỗi khi tải model: {e}")

try:
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    print("✅ Preprocessor loaded successfully.")
except FileNotFoundError:
    print(f"❌ Không tìm thấy file {PREPROCESSOR_PATH}")
except Exception as e:
    print(f"❌ Lỗi khi tải preprocessor: {e}")

# =============================================
# API PREDICT
# =============================================
@app.route('/predict', methods=['POST'])
def predict():
    # Kiểm tra model và preprocessor
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model hoặc preprocessor chưa được tải. Vui lòng kiểm tra server.'}), 500

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

        # Tạo DataFrame với đúng tên cột như khi huấn luyện
        input_dict = {
            'age': [age],
            'bmi': [bmi],
            'hbA1c_level': [hbA1c_level],
            'blood_glucose_level': [blood_glucose_level],
            'gender': [gender],
            'smoking_history': [smoking_history],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'race': [race]  # QUAN TRỌNG: truyền cột gốc để preprocessor tự one-hot
        }
        df = pd.DataFrame(input_dict)

        # In ra cấu trúc để debug (nếu cần)
        app.logger.info(f"Input DataFrame columns: {df.columns.tolist()}")
        app.logger.info(f"Input DataFrame shape: {df.shape}")

        # Tiền xử lý
        X_processed = preprocessor.transform(df)

        # Dự đoán
        pred = model.predict(X_processed)[0]
        proba = model.predict_proba(X_processed)[0][1]

        return jsonify({
            'prediction': int(pred),
            'probability': round(float(proba), 4)
        })

    except KeyError as e:
        app.logger.error(f"KeyError: {e}")
        return jsonify({'error': f'Thiếu trường dữ liệu: {str(e)}'}), 400
    except ValueError as e:
        app.logger.error(f"ValueError: {e}")
        return jsonify({'error': f'Giá trị không hợp lệ: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Lỗi server nội bộ: {str(e)}'}), 500

# =============================================
# KIỂM TRA TRẠNG THÁI
# =============================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

if __name__ == '__main__':
    app.run(debug=True)