#Imports
from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import torch
import torch.nn as nn
import warnings
import random
warnings.filterwarnings('ignore')
import time

#App
app = Flask(__name__)

#Loads models
model_path = os.path.join(os.path.dirname(__file__), 'models', 'results', 'nb_rf_rcm_2.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'models', 'results', 'lr_vectorizer.pkl')
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

model_path2 = os.path.join(os.path.dirname(__file__), 'models', 'results', 'logistic_regression_model.pkl')
model2 = joblib.load(model_path2)

model_path3 = os.path.join(os.path.dirname(__file__), 'models', 'results', 'knn_model.pkl')
model3 = joblib.load(model_path3)

model_path4 = os.path.join(os.path.dirname(__file__), 'models', 'results', 'rf_gb_2.pkl')
model4 = joblib.load(model_path4)

#Predict function for each model
def predict_model1(payload):
    start_time = time.time()
    payload_transformed = vectorizer.transform([payload])
    prediction = model.predict(payload_transformed)
    probabilities = model.predict_proba(payload_transformed).max()
    confidence_scores = model.predict_proba(payload_transformed).flatten()
    
    inference_time = time.time() - start_time

    result_string = (
        f"Predicted class: {prediction[0]}\n"
        f"Confidence (max): {probabilities:.2%}\n"
        f"Time: {inference_time:.4f} seconds\n"
    )

    if prediction == 1:
        return f"XSS attack detected!\n {result_string}"
    elif prediction == 2:
        return f"SQL Injection attack detected!\n {result_string}"
    return f"No attack detected!\n {result_string}"

def predict_model2(payload):
    start_time = time.time()
    payload_transformed = vectorizer.transform([payload])
    prediction = model2.predict(payload_transformed)
    probabilities = model2.predict_proba(payload_transformed).max()
    confidence_scores = model2.predict_proba(payload_transformed).flatten()
    
    inference_time = time.time() - start_time

    result_string = (
        f"Predicted class: {prediction[0]}\n"
        f"Confidence (max): {probabilities:.2%}\n"
        f"Time: {inference_time:.4f} seconds\n"
    )

    if prediction == 1:
        return f"XSS attack detected!\n {result_string}"
    elif prediction == 2:
        return f"SQL Injection attack detected!\n {result_string}"
    return f"No attack detected!\n {result_string}"

def predict_model3(payload):
    start_time = time.time()
    payload_transformed = vectorizer.transform([payload])
    prediction = model3.predict(payload_transformed)
    probabilities = model3.predict_proba(payload_transformed).max()
    confidence_scores = model3.predict_proba(payload_transformed).flatten()
    
    inference_time = time.time() - start_time

    result_string = (
        f"Predicted class: {prediction[0]}\n"
        f"Confidence (max): {probabilities:.2%}\n"
        f"Time: {inference_time:.4f} seconds\n"
    )

    if prediction == 1:
        return f"XSS attack detected!\n {result_string}"
    elif prediction == 2:
        return f"SQL Injection attack detected!\n {result_string}"
    return f"No attack detected!\n {result_string}"

def predict_model4(payload):
    start_time = time.time()
    payload_transformed = vectorizer.transform([payload])
    prediction = model4.predict(payload_transformed)
    probabilities = model4.predict_proba(payload_transformed).max()
    
    inference_time = time.time() - start_time

    result_string = (
        f"Predicted class: {prediction[0]}\n"
        f"Confidence (max): {probabilities:.2%}\n"
        f"Time: {inference_time:.4f} seconds\n"
    )

    if prediction == 1:
        return f"XSS attack detected!\n {result_string}"
    elif prediction == 2:
        return f"SQL Injection attack detected!\n {result_string}"
    return f"No attack detected!\n {result_string}"


model_state_dict = torch.load(r"D:\NationalSecret\IE105\models\results\lstm_model.pth")
char2idx = joblib.load(r"D:\NationalSecret\IE105\models\results\char2idx.pkl")
label_encoder = joblib.load(r"D:\NationalSecret\IE105\models\results\label_encoder.pkl")

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


embedding_dim = 128
hidden_dim = 128
output_dim = len(label_encoder.classes_)
vocab_size = len(char2idx) + 1

lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
lstm_model.load_state_dict(model_state_dict)
lstm_model.eval()

def encode_and_pad(payload, char2idx, max_len=100):
    encoded = [char2idx.get(c, 0) for c in list(str(payload))]
    tensor = torch.tensor(encoded, dtype=torch.long)
    
    if len(tensor) < max_len:
        tensor = torch.cat([tensor, torch.zeros(max_len - len(tensor), dtype=torch.long)])
    else:
        tensor = tensor[:max_len]
    return tensor.unsqueeze(0)

import time

def predict_model5(payload):
    start_time = time.time()
    payload_tensor = encode_and_pad(payload, char2idx)
    
    with torch.no_grad():
        output = lstm_model(payload_tensor)
        probabilities = torch.softmax(output, dim=1).numpy()[0]
        _, predicted = torch.max(output, 1)
        prediction = label_encoder.inverse_transform(predicted.numpy())
        
        max_probability = probabilities.max()
        confidence_scores = probabilities.tolist()  
        
        inference_time = time.time() - start_time

    result_string = (
        f"Predicted class: {prediction[0]}\n"
        f"Confidence (max): {max_probability:.2%}\n"
        f"Time: {inference_time:.4f} seconds\n"
    )

    if predicted.item() == 1:
        return f"XSS attack detected! {result_string}"
    elif predicted.item() == 2:
        return f"SQL Injection attack detected! {result_string}"
    
    return f"No attack detected! {result_string}"

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
            'model4': predict_model4(payload),
            'model5': predict_model5(payload)
        }
    
    return render_template('index.html', results=results)

if __name__ in "__main__":
    app.run(debug=True)