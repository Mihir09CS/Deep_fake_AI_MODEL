import joblib
import os

def load_audio_model():
    model_path = os.getenv("AUDIO_MODEL_PATH", "models/deepfake_audio_model.pkl")
    model = joblib.load(model_path)
    return model

audio_model = None
audio_model_loaded = False
audio_model_error = None

try:
    audio_model = load_audio_model()
    audio_model_loaded = True
except Exception as error:
    audio_model_error = str(error)
