import os
from tensorflow.keras.models import load_model, model_from_json


def _resolve_model_path(model_path):
    if os.path.isabs(model_path):
        return model_path

    service_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(service_root, model_path)


def _get_image_arch_path():
    value = os.getenv("IMAGE_MODEL_ARCH_PATH", "").strip()
    if not value:
        return None
    return _resolve_model_path(value)


def load_image_model():
    model_path = _resolve_model_path(os.getenv("IMAGE_MODEL_PATH", "models/deepfake_image_model.h5"))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Image model not found at: {model_path}")

    try:
        model = load_model(model_path, compile=False)
    except Exception as error:
        message = str(error)
        if "No model config found" in message:
            arch_path = _get_image_arch_path()
            if arch_path and os.path.exists(arch_path):
                with open(arch_path, "r", encoding="utf-8") as f:
                    model_json = f.read()
                model = model_from_json(model_json)
                model.load_weights(model_path)
                return model

            raise ValueError(
                "IMAGE_MODEL_PATH points to a weights-only .h5 file. "
                "Provide a full saved model (.keras or model.save(... .h5)) "
                "or set IMAGE_MODEL_ARCH_PATH to a model JSON architecture file "
                "so weights can be loaded."
            ) from error
        raise
    return model

image_model = None
image_model_loaded = False
image_model_error = None

try:
    image_model = load_image_model()
    image_model_loaded = True
except Exception as error:
    image_model_error = str(error)
