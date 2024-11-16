# Animal Species Predictor

This project uses a pre-trained InceptionV3 model to classify images of animals. Given an image, it predicts the species of the animal in the image and displays the top 3 predictions along with their confidence scores.

## Requirements

This project requires Python and the following Python libraries:

- TensorFlow (which includes Keras)
- NumPy
- Pillow

You can install the required libraries using pip:

```bash
pip install tensorflow numpy pillow
```

## How to Run

1. Clone this repository to your local machine.
2. Install the required libraries using the command mentioned above.
3. Place the images you want to classify in a folder named `images` within the project directory.
4. Run the Python script `animal_species_predictor.py`
5. The Output will specify the name of teh image and the prediction below it.
