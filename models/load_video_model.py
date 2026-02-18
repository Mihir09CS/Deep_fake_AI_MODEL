import os
import sys
from collections import OrderedDict

import torch


def get_video_model_path():
    return os.getenv("VIDEO_MODEL_PATH", "models/deepfake_video_model.pt")


def get_video_frame_interval():
    return max(int(os.getenv("VIDEO_FRAME_INTERVAL", "5")), 1)


def get_video_max_frames():
    return max(int(os.getenv("VIDEO_MAX_FRAMES", "24")), 1)


def allow_unsafe_torch_load():
    return os.getenv("ALLOW_UNSAFE_TORCH_LOAD", "true").strip().lower() in ("1", "true", "yes")


def _resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_model_path(model_path):
    if os.path.isabs(model_path):
        return model_path

    service_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(service_root, model_path)


def _apply_timm_compatibility_aliases():
    import timm
    import timm.layers
    import timm.layers.adaptive_avgmax_pool as adaptive_avgmax_pool

    # Compatibility alias for older timm checkpoint references.
    sys.modules["timm.models.layers"] = timm.layers
    sys.modules["timm.models.layers.adaptive_avgmax_pool"] = adaptive_avgmax_pool


def load_video_model():
    model_path = _resolve_model_path(get_video_model_path())
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Video model not found at: {model_path}")

    try:
        import timm  # noqa: F401
    except Exception as error:
        raise ImportError(
            "Missing dependency 'timm' for this video checkpoint. "
            "Install with: python -m pip install timm"
        ) from error

    _apply_timm_compatibility_aliases()

    if not allow_unsafe_torch_load():
        raise ValueError(
            "ALLOW_UNSAFE_TORCH_LOAD must be true to load this checkpoint "
            "with weights_only=False."
        )

    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility for torch versions without weights_only arg.
        model = torch.load(model_path, map_location="cpu")

    if isinstance(model, dict):
        # Common checkpoint format: {"model": <nn.Module>, "model_state_dict": ...}
        embedded_model = model.get("model")
        if isinstance(embedded_model, torch.nn.Module):
            state_dict = model.get("model_state_dict")
            model = embedded_model
            if isinstance(state_dict, (dict, OrderedDict)):
                model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(
                "VIDEO_MODEL_PATH points to a checkpoint/state_dict dictionary. "
                "Model architecture is required to load these weights."
            )
    elif isinstance(model, OrderedDict):
        raise ValueError(
            "VIDEO_MODEL_PATH points to a checkpoint/state_dict dictionary. "
            "Model architecture is required to load these weights."
        )

    if not isinstance(model, torch.nn.Module):
        raise ValueError(
            "Loaded .pt object is not a torch.nn.Module. "
            "Provide a full exported model for inference."
        )

    device = _resolve_device()
    model = model.to(device)
    model.eval()
    return model, device


video_model = None
video_device = None
video_model_loaded = False
video_model_error = None

try:
    video_model, video_device = load_video_model()
    video_model_loaded = True
except Exception as error:
    video_model_error = str(error)
