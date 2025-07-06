import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load model
model = load_model("image_classifier_model.h5")

# Image path
img_path = "test.jpg"
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
class_names = sorted(os.listdir("dataset"))
print("Predicted class:", class_names[np.argmax(prediction)])
