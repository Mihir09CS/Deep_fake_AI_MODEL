


from utils.downloader import download_media
from models.audio_inference import predict_audio
from models.image_inference import predict_image

def classify_risk(prob):
    if prob >= 0.75:
        return "High"
    elif prob >= 0.45:
        return "Medium"
    else:
        return "Low"

def analyze_media(url):
    file_path, media_type = download_media(url)

    if media_type == "audio":
        prob = predict_audio(file_path)

    elif media_type == "image":
        prob = predict_image(file_path)

    else:
        raise ValueError("Unsupported media type")

    risk = classify_risk(prob)


    return {
        "mediaType": media_type,
        "probability": round(prob, 3),
        "risk": risk
    }
