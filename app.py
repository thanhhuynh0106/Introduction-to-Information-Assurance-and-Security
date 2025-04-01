#Imports
from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import warnings
warnings.filterwarnings('ignore')

#App
app = Flask(__name__)

#Loads models
model_path = os.path.join(os.path.dirname(__file__), 'models', 'rf_model_ver3.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'models', 'vectorizer_3.pkl')
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

#Predict function
def predict_attack(payload):
    payload = vectorizer.transform([payload])
    prediction = model.predict(payload)
    accuracy = model.predict_proba(payload).max()

    if (prediction == 1):
        return f"XSS attack detected! (Confidence: {accuracy:.2%})"
    if (prediction == 2):
        return f"SQL Injection attack detected! (Confidence: {accuracy:.2%})"
    if (prediction == 4):
        return f"Command Injection attack detected! (Confidence: {accuracy:.2%})"
    if (prediction == 3):
        return f"XML Injection attack detected! (Confidence: {accuracy:.2%})"

    return f"No attack detected! (Confidence: {accuracy:.2%})"


#Main page
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        payload = request.form['payload']
        result = predict_attack(payload)
    
    return render_template('index.html', result=result)



if __name__ in "__main__":
    app.run(debug=True)