from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Tải model và preprocessor
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Lấy dữ liệu từ form
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        hbA1c_level = float(request.form['hbA1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])
        gender = request.form['gender']
        smoking_history = request.form['smoking_history']
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        race = request.form['race']   # giữ nguyên giá trị gốc

        # 2. Tạo DataFrame với đúng tên cột như khi huấn luyện
        #    (giả sử thứ tự cột là: age, bmi, hbA1c_level, blood_glucose_level,
        #     gender, smoking_history, hypertension, heart_disease, race)
        input_dict = {
            'age': [age],
            'bmi': [bmi],
            'hbA1c_level': [hbA1c_level],
            'blood_glucose_level': [blood_glucose_level],
            'gender': [gender],
            'smoking_history': [smoking_history],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'race': [race]   # <--- quan trọng: cột gốc, không phải one‑hot
        }
        df = pd.DataFrame(input_dict)

        # 3. Tiền xử lý (preprocessor sẽ tự one‑hot race và chuẩn hóa số)
        X_processed = preprocessor.transform(df)

        # 4. Dự đoán
        pred = model.predict(X_processed)[0]
        proba = model.predict_proba(X_processed)[0][1]

        return jsonify({
            'prediction': int(pred),
            'probability': round(float(proba), 4)
        })

    except Exception as e:
        # Ghi log lỗi để dễ debug (tuỳ chọn)
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)