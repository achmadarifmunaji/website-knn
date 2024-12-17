from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load model KNN
with open('knn_model.pkl', 'rb') as f:
    model = joblib.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data["fitur"]
    # print(features)
    # Melakukan prediksi dengan model KNN untuk mendapatkan indeks kelas
    prediction_index = model.predict([features])[0]

    #  Array ini digunakan untuk memeriksa hasil prediksi sesuai dengan urutan kelas
    class_check = ['Setosa', 'Versicolor', 'Virginica']

    # Melakukan pemeriksaan untuk memastikan hasil prediksi cocok dengan nama kelas
    if 0 <= prediction_index < len(class_check):
        predicted_class = class_check[prediction_index]
    else:
        predicted_class = "Kelas Tidak Diketahui"

    # Mengirimkan hasil prediksi sebagai nama kelas
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)