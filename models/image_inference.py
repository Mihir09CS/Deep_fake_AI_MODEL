import numpy as np
from PIL import Image
from models.load_image_model import image_model

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))  # adjust if your model differs

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_image(image_path):
    img = preprocess_image(image_path)

    prediction = image_model.predict(img)[0][0]

    return float(prediction)


