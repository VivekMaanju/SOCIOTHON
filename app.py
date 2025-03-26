from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('safety_model_punjab.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Convert input to numpy array in correct order
        features = np.array([
            data['crime_rate'],
            data['street_lights'],
            data['population'],
            data['emergency']
        ]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Scale prediction to 0-100 if needed (adjust based on your model's output)
        safety_score = np.clip(prediction, 0, 100)
        
        return jsonify({'prediction': safety_score})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
