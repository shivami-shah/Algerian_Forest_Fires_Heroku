from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = pickle.load(open('models/model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/', methods = ['POST', 'GET'])
def predict():
    if request.method == "POST":
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        BUI = float(request.form.get('BUI'))
        Bejaia = float(request.form.get('Region_Bejaia'))
        
        new_data_scaled = scaler.transform([[Rain,FFMC,DMC,ISI,BUI,Bejaia]])
        result = model.predict(new_data_scaled)

        return render_template('index.html',result=f"The predicted value is {result[0]}")

        
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()