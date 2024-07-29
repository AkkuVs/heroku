from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('fish_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract inputs from the form
        length = float(request.form.get('length', 0))
        weight = float(request.form.get('weight', 0))
        width = float(request.form.get('width', 0))
        height = float(request.form.get('height', 0))
        
        # Prepare the input data for prediction
        features = [[length, weight, width, height]]  # Adjust according to your model's input features
        prediction = model.predict(features)
        
        # Display the result
        result = f'Predicted Species: {prediction[0]}'
        return render_template('index.html', prediction=result)
    except Exception as e:
        # Handle errors and provide feedback
        return render_template('index.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
