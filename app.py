from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Загружаем модель один раз при старте сервера
model = joblib.load('models/linear_regression_model.joblib')

@app.route('/predict-price', methods=['GET'])
def predict_price():
    try:
        day_of_week = int(request.args.get('day_of_week'))
        rolling_mean_3 = float(request.args.get('rolling_mean_3'))
        rolling_std_3 = float(request.args.get('rolling_std_3'))
        price_diff = float(request.args.get('price_diff'))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing query parameters"}), 400

    # Формируем DataFrame с входными данными
    X_new = pd.DataFrame({
        'day_of_week': [day_of_week],
        'rolling_mean_3': [rolling_mean_3],
        'rolling_std_3': [rolling_std_3],
        'price_diff': [price_diff]
    })

    # Делаем прогноз
    pred = model.predict(X_new)[0]

    return jsonify({'predicted_price': pred})

if __name__ == '__main__':
    app.run(debug=True)
