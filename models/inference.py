from utils.downloader import download_media
from models.audio_inference import predict_audio
from models.image_inference import predict_image
from models.video_inference import predict_video
import os

def classify_risk(prob):
    if prob >= 0.75:
        return "High"
    elif prob >= 0.45:
        return "Medium"
    else:
        return "Low"

def analyze_local_file(file_path, media_type):
    if media_type == "audio":
        prob = predict_audio(file_path)
    elif media_type == "image":
        prob = predict_image(file_path)
    elif media_type == "video":
        prob = predict_video(file_path)
    else:
        raise ValueError("Unsupported media type")

    risk = classify_risk(prob)
    return {
        "mediaType": media_type,
        "probability": round(prob, 3),
        "risk": risk
    }


def analyze_media(url):
    file_path, media_type = download_media(url)
    try:
        return analyze_local_file(file_path, media_type)
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
