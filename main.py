import os
import sys
import tempfile

from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile
from pydantic import BaseModel
try:
    from .models.inference import analyze_local_file, analyze_media
    from .models.load_audio_model import load_audio_model
    from .models.load_image_model import load_image_model
    from .models.load_video_model import load_video_model
    from .utils.downloader import infer_media_type
except ImportError:
    service_root = os.path.dirname(os.path.abspath(__file__))
    if service_root not in sys.path:
        sys.path.insert(0, service_root)
    from models.inference import analyze_local_file, analyze_media
    from models.load_audio_model import load_audio_model
    from models.load_image_model import load_image_model
    from models.load_video_model import load_video_model
    from utils.downloader import infer_media_type

app = FastAPI()

class MediaRequest(BaseModel):
    mediaUrl: str

@app.get("/health")
def health():
    errors = {}

    def _check_audio():
        try:
            load_audio_model()
            return True
        except Exception as error:
            errors["audio"] = str(error)
            return False

    def _check_image():
        try:
            load_image_model()
            return True
        except Exception as error:
            errors["image"] = str(error)
            return False

    def _check_video():
        try:
            load_video_model()
            return True
        except Exception as error:
            errors["video"] = str(error)
            return False

    models = {
        "audioModelLoaded": _check_audio(),
        "imageModelLoaded": _check_image(),
        "videoModelLoaded": _check_video(),
    }

    return {
        "success": True,
        "status": "ok" if all(models.values()) else "degraded",
        "models": models,
        "errors": errors,
    }

@app.post("/analyze")
def analyze(request: MediaRequest):
    try:
        result = analyze_media(request.mediaUrl)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "mediaType": None,
                "probability": 0.0,
                "risk": "Error",
                "error": str(e),
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "mediaType": None,
                "probability": 0.0,
                "risk": "Error",
                "error": str(e),
            },
        )


@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(
            status_code=400,
            detail={
                "mediaType": None,
                "probability": 0.0,
                "risk": "Error",
                "error": "File is required",
            },
        )

    media_type = infer_media_type(file.filename or "", file.content_type or "")
    if media_type is None:
        raise HTTPException(
            status_code=400,
            detail={
                "mediaType": None,
                "probability": 0.0,
                "risk": "Error",
                "error": "Unsupported uploaded file type",
            },
        )

    suffix = os.path.splitext(file.filename or "")[1] or {
        "image": ".jpg",
        "audio": ".wav",
        "video": ".mp4",
    }[media_type]

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)

        return analyze_local_file(temp_path, media_type)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "mediaType": None,
                "probability": 0.0,
                "risk": "Error",
                "error": str(e),
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "mediaType": None,
                "probability": 0.0,
                "risk": "Error",
                "error": str(e),
            },
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
