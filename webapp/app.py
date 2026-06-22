from flask import Flask, request, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

model = load_model(r"D:\diabetic_retinopathy\models\dr_model.h5")
classes = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    pred = model.predict(img)
    predicted_class = classes[np.argmax(pred)]
    confidence = round(np.max(pred) * 100, 2)
    
    return render_template('index.html', 
                         prediction=predicted_class,
                         confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)