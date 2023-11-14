from flask import Flask, request, jsonify
import joblib
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

# Load the saved SVM model
model_filename = 'cornleaf/cornleaf_disease.joblib'
loaded_model = joblib.load(model_filename)

@app.route('/', methods=['POST'])
def predict():
    try:
        # Get the uploaded image
        file = request.files['file']
        img = image.load_img(file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features using the VGG16 model
        img_features = model.predict(img_array)

        # Make predictions using the SVM model
        prediction = loaded_model.predict(img_features)

        # Decode the prediction back to class labels
        predicted_class = int(np.argmax(prediction))

        result = {'class': predicted_class, 'message': 'success'}
    except Exception as e:
        result = {'message': 'error', 'error': str(e)}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
