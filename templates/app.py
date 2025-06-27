import os
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask application
app = Flask(__name__)


MODEL_PATH = 'model.h5'

# Check if the model file exists before attempting to load
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    
    exit("Model file 'model.h5' not found. Please ensure it's in the same directory.")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = [
    'anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena',
    'combretum', 'croton', 'dipteryx', 'eucalipto', 'faramea',
    'hyptis', 'mabea', 'matayba', 'miconia', 'mimosa', 'myrcia',
    'protium', 'qualea', 'schefflera', 'senegalia', 'serjania',
    'syagrus', 'urochloa'
]

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/prediction', methods=['GET'])
def prediction_page():
    """Renders the prediction page."""
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and returns the prediction."""
    if 'image' not in request.files:
        return render_template('prediction.html', error='No image file provided.')

    f = request.files['image']

    if f.filename == '':
        return render_template('prediction.html', error='No image selected.')

    # Allowed extensions (basic security check)
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    if not allowed_file(f.filename):
        return render_template('prediction.html', error='Invalid file type. Only images (png, jpg, jpeg, gif) are allowed.')

    try:
     
        filename = f.filename # In a real app, use werkzeug.utils.secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        # Load and preprocess the image
        img = image.load_img(filepath, target_size=(128, 128))
        x = image.img_to_array(img)
        x = x / 255.0  # Normalize
        x = np.expand_dims(x, axis=0) # Add batch dimension

        # Make prediction
        predictions = model.predict(x)
        pred_index = np.argmax(predictions, axis=1)[0]

        # Get the result
        result = CLASS_NAMES[pred_index]

        # Clean up the uploaded file after prediction (optional but good practice)
        os.remove(filepath)

        return render_template('prediction.html', prediction_text=f'The pollen grain type is: {result}')

    except Exception as e:
        # It's good practice to log the full exception for debugging
        print(f"An error occurred during prediction: {e}")
        return render_template('prediction.html', error=f'Error during prediction. Please try again. ({e})')

# Main driver function
if __name__ == '__main__':
    # When deploying, set debug=False and use a production-ready WSGI server like Gunicorn
    app.run(debug=True)

