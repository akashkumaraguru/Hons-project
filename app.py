import sys

try:
    from flask import Flask, render_template, request, jsonify
except ImportError:
    print("Flask is not installed. Please install it using 'pip install flask==2.2.5'")
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow is not installed. Please install it using 'pip install tensorflow==2.12.0'")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("NumPy is not installed. Please install it using 'pip install numpy==1.23.5'")
    sys.exit(1)

app = Flask(__name__)

# Load the trained model
try:
    model = tf.keras.models.load_model('static/model/mnist_model.h5')
    print("Model loaded successfully")
except:
    print("Error loading the model. Make sure you've run train_model.py to create the model first.")
    sys.exit(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.json['image']
    image = np.array(image_data).reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    print("Input shape:", image.shape)
    print("Input min/max:", image.min(), image.max())
    
    prediction = model.predict(image)
    print("Raw prediction:", prediction)
    
    predicted_class = prediction.argmax()
    print("Predicted class:", predicted_class)
    
    return jsonify({'prediction': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)