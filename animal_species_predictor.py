import tensorflow as tf
from tensorflow.keras.preprocessing import image  
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions  
import numpy as np
import os

# Load pre-trained InceptionV3 model + higher level layers
base_model = InceptionV3(weights='imagenet')

# Function to classify a single image and return the predictions
def classify_image(img_path):
    # Load the image, resize it to 299x299 pixels (required input size for InceptionV3)
    img = image.load_img(img_path, target_size=(299, 299))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand the dimensions to match the expected input of the model (batch size of 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image for the InceptionV3 model
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = base_model.predict(img_array)

    # Decode the predictions to human-readable labels
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Store predictions in a dictionary
    predictions_dict = {}
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        predictions_dict[label] = f"Score: {score:.2f}"

    return predictions_dict

# Directory containing images
image_dir = 'images'  # Folder called 'images' where the images are stored

# Check if the directory exists
if not os.path.isdir(image_dir):
    print(f"The directory '{image_dir}' does not exist.")
else:
    # Dictionary to store predictions for all images
    all_predictions = {}

    # Loop through all the files in the directory
    for filename in os.listdir(image_dir):
        # Check if the file is an image (e.g., ends with .jpg, .png, etc.)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, filename)
            predictions = classify_image(img_path)

            # Store the predictions in the dictionary with the image name as the key
            all_predictions[filename] = predictions

    # Print the predictions in a user-friendly format
    print("\nPredictions for all images in the 'images' folder:\n")
    for img_name, predictions in all_predictions.items():
        print(f"Predictions for {img_name}:")
        for label, score in predictions.items():
            print(f"  - {label}: {score}")
        print("\n")
