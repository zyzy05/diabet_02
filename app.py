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

        # Tạo DataFrame với các cột đúng thứ tự như khi huấn luyện
        input_dict = {
            'age': [age],
            'bmi': [bmi],
            'hbA1c_level': [hbA1c_level],
            'blood_glucose_level': [blood_glucose_level],
            'gender': [gender],
            'smoking_history': [smoking_history],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'race:AfricanAmerican': [1 if race == 'AfricanAmerican' else 0],
            'race:Asian': [1 if race == 'Asian' else 0],
            'race:Caucasian': [1 if race == 'Caucasian' else 0],
            'race:Hispanic': [1 if race == 'Hispanic' else 0],
            'race:Other': [1 if race == 'Other' else 0],
        }
        df = pd.DataFrame(input_dict)

        # Tiền xử lý
        X_processed = preprocessor.transform(df)

        # Dự đoán
        pred = model.predict(X_processed)[0]
        proba = model.predict_proba(X_processed)[0][1]

        return jsonify({
            'prediction': int(pred),
            'probability': round(float(proba), 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)