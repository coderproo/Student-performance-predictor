<<<<<<< HEAD
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('student_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    marks = float(request.form['marks'])
    attendance = float(request.form['attendance'])

    input_data = np.array([[marks, attendance]])
    print(f"Raw input: {input_data}") 

    input_scaled = scaler.transform(input_data)
    print(f"Scaled input: {input_scaled}") 

    prediction = model.predict(input_scaled)
    print(f"Model prediction: {prediction}") 

    result = 'Pass' if prediction[0] == 1 else 'Fail'
    
    return render_template('index.html', prediction_text=f"Predicted Result: {result}")

if __name__ == '__main__':
    app.run(debug=True)
=======
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('student_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    marks = float(request.form['marks'])
    attendance = float(request.form['attendance'])

    input_data = np.array([[marks, attendance]])
    print(f"Raw input: {input_data}") 

    input_scaled = scaler.transform(input_data)
    print(f"Scaled input: {input_scaled}") 

    prediction = model.predict(input_scaled)
    print(f"Model prediction: {prediction}") 

    result = 'Pass' if prediction[0] == 1 else 'Fail'
    
    return render_template('index.html', prediction_text=f"Predicted Result: {result}")

if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> 3919a7b8ab0fca107cd600067d17a608d8d75b23
