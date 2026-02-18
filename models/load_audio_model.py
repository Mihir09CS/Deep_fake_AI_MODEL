import joblib

def load_audio_model():
    model = joblib.load("models/deepfake_audio_model.pkl")
    return model

audio_model = load_audio_model()
