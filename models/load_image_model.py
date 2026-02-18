from tensorflow.keras.models import load_model

def load_image_model():
    model = load_model("models/deepfake_image_model.h5")
    return model

image_model = load_image_model()
