from fastapi import FastAPI
from pydantic import BaseModel
from models.inference import analyze_media

app = FastAPI()

class MediaRequest(BaseModel):
    mediaUrl: str

@app.post("/analyze")
def analyze(request: MediaRequest):
    try:
        result = analyze_media(request.mediaUrl)
        return result
    except Exception as e:
        return {
            "mediaType": None,
            "probability": 0.0,
            "risk": "Error",
            "error": str(e)
        }
