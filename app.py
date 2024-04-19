from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'


MODEL_URL = 'https://github.com/Dushyantsingh98/Crack-Detection/blob/main/saved_model.pb'

# Function to download the model from the provided URL
def download_model(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception("Failed to download the model")

# Load the model from the GitHub repository
def load_model_from_github(url):
    model_data = download_model(url)
    return tf.saved_model.load_from_string(model_data)

model = load_model_from_github(MODEL_URL)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        image_data = file.read()
        image_tensor = read_file_as_image(image_data)
        image_tensor = tf.expand_dims(image_tensor, axis=0)  # Add batch dimension

        # Predict using the TensorFlow model
        serving_input = model.signatures['serving_default']
        predictions = serving_input(tf.constant(image_tensor, dtype=tf.float32))['output_0']

# Assuming the model outputs a single probability for the positive class
# Since we are dealing with one image, predictions[0] gives us the probability of the image being positive
       
        result = 'Positive' if predictions[0] > 0.3 else 'Negative'
        return render_template('index.html', prediction=result)
    return render_template('index.html', prediction='No file or unsupported format')

def read_file_as_image(data) -> tf.Tensor:
    image = Image.open(BytesIO(data))
    image = image.convert('RGB')
    image = np.array(image)
    image = tf.image.resize(image, (224, 224)) / 255.0  # Resize and normalize
    return image

if __name__ == '__main__':
    app.run(debug=True)
