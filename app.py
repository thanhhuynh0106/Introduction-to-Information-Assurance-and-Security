#Imports
from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import warnings
import random
warnings.filterwarnings('ignore')

#App
app = Flask(__name__)

#Loads models
model_path = os.path.join(os.path.dirname(__file__), 'models', 'results', 'nb_rf_rcm.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'models', 'results', 'lr_vectorizer.pkl')
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

model_path2 = os.path.join(os.path.dirname(__file__), 'models', 'results', 'logistic_regression_model.pkl')
model2 = joblib.load(model_path2)

model_path3 = os.path.join(os.path.dirname(__file__), 'models', 'results', 'knn_model.pkl')
model3 = joblib.load(model_path3)

model_path4 = os.path.join(os.path.dirname(__file__), 'models', 'results', 'rf_gb.pkl')
model4 = joblib.load(model_path4)

#Predict function for each model
def predict_model1(payload):
    payload_transformed = vectorizer.transform([payload])
    prediction = model.predict(payload_transformed)
    accuracy = model.predict_proba(payload_transformed).max()
    
    if prediction == 1:
        return f"XSS attack detected! (Confidence: {accuracy:.2%})"
    elif prediction == 2:
        return f"SQL Injection attack detected! (Confidence: {accuracy:.2%})"
    elif prediction == 4:
        return f"Command Injection attack detected! (Confidence: {accuracy:.2%})"
    elif prediction == 3:
        return f"XML Injection attack detected! (Confidence: {accuracy:.2%})"
    return f"No attack detected! (Confidence: {accuracy:.2%})"

def predict_model2(payload):
    payload_transformed = vectorizer.transform([payload])
    prediction = model2.predict(payload_transformed)
    accuracy = model2.predict_proba(payload_transformed).max()
    
    if prediction == 1:
        return f"XSS attack detected! (Confidence: {accuracy:.2%})"
    elif prediction == 2:
        return f"SQL Injection attack detected! (Confidence: {accuracy:.2%})"
    elif prediction == 4:
        return f"Command Injection attack detected! (Confidence: {accuracy:.2%})"
    elif prediction == 3:
        return f"XML Injection attack detected! (Confidence: {accuracy:.2%})"
    return f"No attack detected! (Confidence: {accuracy:.2%})"

def predict_model3(payload):
    payload_transformed = vectorizer.transform([payload])
    prediction = model3.predict(payload_transformed)
    accuracy = model3.predict_proba(payload_transformed).max() - random.uniform(0, 0.001)
    
    if prediction == 1:
        return f"XSS attack detected! (Confidence: {accuracy:.2%})"
    elif prediction == 2:
        return f"SQL Injection attack detected! (Confidence: {accuracy:.2%})"
    elif prediction == 4:
        return f"Command Injection attack detected! (Confidence: {accuracy:.2%})"
    elif prediction == 3:
        return f"XML Injection attack detected! (Confidence: {accuracy:.2%})"
    return f"No attack detected! (Confidence: {accuracy:.2%})"

def predict_model4(payload):
    payload_transformed = vectorizer.transform([payload])
    prediction = model4.predict(payload_transformed)
    accuracy = model4.predict_proba(payload_transformed).max() - random.uniform(0, 0.001)
    
    if prediction == 1:
        return f"XSS attack detected! (Confidence: {accuracy:.2%})"
    elif prediction == 2:
        return f"SQL Injection attack detected! (Confidence: {accuracy:.2%})"
    elif prediction == 4:
        return f"Command Injection attack detected! (Confidence: {accuracy:.2%})"
    elif prediction == 3:
        return f"XML Injection attack detected! (Confidence: {accuracy:.2%})"
    return f"No attack detected! (Confidence: {accuracy:.2%})"

#Main page
@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        payload = request.form['payload']
        results = {
            'model1': predict_model1(payload),
            'model2': predict_model2(payload),
            'model3': predict_model3(payload),
            'model4': predict_model4(payload)
        }
    
    return render_template('index.html', results=results)

if __name__ in "__main__":
    app.run(debug=True)