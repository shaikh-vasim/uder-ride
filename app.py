import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
import math
import pandas as pd


app = Flask(__name__)
model = joblib.load('uber-ride-model-save')

@app.route('/')
def home():
    return render_template('html/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    final_features = np.array(input_features)
    output = model.predict([final_features])
    # output = model.predict([final_features])[0][0].round(2)
    # output = int(predict)
    return render_template('html/index.html',prediction_text="Number of Weekly Rides Should be {}".format(int(output[0])))

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)
