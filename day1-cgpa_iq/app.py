import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import flask
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    prediction = model.predict(final_features) # making prediction
    if prediction ==1:
        prediction_text = "Placement is confirmed"
    else:
        prediction_text = "Placement is not confirmed"
        
 


    return render_template('index.html', prediction_class='Predicted Class: {}'.format(prediction), prediction_text =prediction_text)


if __name__ == "__main__":
    app.run(debug=True)