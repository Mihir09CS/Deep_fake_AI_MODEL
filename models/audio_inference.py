import librosa
import numpy as np
from models.load_audio_model import audio_model

def extract_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Pad if shorter than 1 second
    if len(audio) < 16000:
        padding = 16000 - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=16000,
        n_mfcc=40
    )

    features = np.mean(mfcc.T, axis=0)

    return features


def predict_audio(audio_path):
    features = extract_features(audio_path)

    features = features.reshape(1, -1)

    if features.shape[1] != 40:
        raise ValueError("Feature dimension mismatch")

    prob = audio_model.predict_proba(features)[0][1]

    return float(prob)
