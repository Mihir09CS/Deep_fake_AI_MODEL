# AI Service Deployment Guide

## 1) Prerequisites
- Python 3.11 recommended
- `pip` + virtual environment support

This service supports:
- Audio model (`.pkl`)
- Image model (`.pkl`)
- Video model (`.pt`, PyTorch)

## 2) Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 3) Environment Variables
- `AUDIO_MODEL_PATH` (default: `models/deepfake_audio_model.pkl`)
- `IMAGE_MODEL_PATH` (default: `models/deepfake_image_model.pkl`)
- `IMAGE_INPUT_SIZE` (default: `224`)
- `VIDEO_MODEL_PATH` (default: `models/deepfake_video_model.pt`)
- `VIDEO_FRAME_INTERVAL` (default: `5`)
- `VIDEO_MAX_FRAMES` (default: `24`)

## 4) Run Service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 5) Health + Analyze Checks

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"mediaUrl":"https://example.com/sample.jpg"}'
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"mediaUrl":"https://example.com/sample.mp4"}'
```

## 6) Deployment Notes
- Ensure model files exist under `models/`:
  - `deepfake_audio_model.pkl`
  - `deepfake_image_model.pkl`
  - `deepfake_video_model.pt`
- Keep this service reachable from backend (`AI_SERVICE_URL`).
- Use HTTPS URL in backend env when deployed separately.
