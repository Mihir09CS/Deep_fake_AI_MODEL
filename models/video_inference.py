import cv2
import numpy as np
import torch

try:
    from .load_video_model import (
        get_video_frame_interval,
        get_video_max_frames,
        video_device,
        video_model,
        video_model_error,
        video_model_loaded,
    )
except ImportError:
    from models.load_video_model import (
        get_video_frame_interval,
        get_video_max_frames,
        video_device,
        video_model,
        video_model_error,
        video_model_loaded,
    )


def _preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (224, 224))
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _to_probability(output):
    if isinstance(output, (list, tuple)):
        output = output[0]

    if isinstance(output, dict):
        if "logits" in output:
            output = output["logits"]
        elif "pred" in output:
            output = output["pred"]
        else:
            output = next(iter(output.values()))

    if not isinstance(output, torch.Tensor):
        output = torch.tensor(output)

    flat = output.detach().float().view(-1)

    if flat.numel() == 0:
        raise ValueError("Video model output is empty")

    if flat.numel() == 1:
        value = float(flat.item())
        if 0.0 <= value <= 1.0:
            return value
        return float(torch.sigmoid(torch.tensor(value)).item())

    if flat.numel() >= 2:
        probs = torch.softmax(flat, dim=0)
        return float(probs[-1].item())

    raise ValueError("Unsupported video model output shape")


def predict_video(video_path):
    if not video_model_loaded or video_model is None:
        raise ValueError(f"Video model unavailable: {video_model_error or 'not loaded'}")

    frame_interval = get_video_frame_interval()
    max_frames = get_video_max_frames()

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Unable to open video file")

    frame_probs = []
    frame_index = 0

    try:
        while len(frame_probs) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_interval != 0:
                frame_index += 1
                continue

            frame_index += 1
            try:
                input_tensor = _preprocess_frame(frame).to(video_device)
                with torch.no_grad():
                    output = video_model(input_tensor)
                frame_probs.append(_to_probability(output))
            except Exception:
                continue
    finally:
        capture.release()

    if not frame_probs:
        raise ValueError("No valid frames could be processed from video")

    return float(np.mean(frame_probs))
