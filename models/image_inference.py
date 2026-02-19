import os

import numpy as np
from PIL import Image

try:
    from .load_image_model import image_model, image_model_error, image_model_loaded
except ImportError:
    from models.load_image_model import image_model, image_model_error, image_model_loaded


def _get_image_size():
    value = os.getenv("IMAGE_INPUT_SIZE", "224").strip()
    try:
        size = int(value)
    except Exception:
        size = 224
    return max(32, min(size, 2048))


def _clip01(value):
    return float(max(0.0, min(1.0, value)))


def preprocess_image(image_path):
    size = _get_image_size()
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_4d = np.expand_dims(img_array, axis=0)
    img_flat = img_4d.reshape(1, -1)
    return img_4d, img_flat


def _positive_class_index(model, width):
    classes = getattr(model, "classes_", None)
    if classes is not None:
        labels = [str(c).strip().lower() for c in classes]
        for positive in ("synthetic", "fake", "1", "true"):
            if positive in labels:
                return labels.index(positive)
    return 1 if width > 1 else 0


def _proba_from_predict_proba(model, img_4d, img_flat):
    errors = []
    for features in (img_flat, img_4d):
        try:
            proba = np.array(model.predict_proba(features))
            if proba.ndim == 2 and proba.shape[1] > 0:
                pos_idx = _positive_class_index(model, proba.shape[1])
                pos_idx = min(pos_idx, proba.shape[1] - 1)
                return _clip01(float(proba[0][pos_idx]))

            flat = np.ravel(proba)
            if flat.size > 0:
                return _clip01(float(flat[0]))
        except Exception as error:
            errors.append(str(error))
    raise ValueError("predict_proba failed: " + " | ".join(errors))


def _proba_from_predict(model, img_4d, img_flat):
    errors = []
    for features in (img_4d, img_flat):
        try:
            prediction = np.array(model.predict(features))
            flat = np.ravel(prediction)
            if flat.size == 1:
                return _clip01(float(flat[0]))

            if prediction.ndim >= 2 and prediction.shape[-1] >= 2:
                probs = prediction[0].astype(float)
                total = float(np.sum(probs)) or 1.0
                probs = probs / total
                pos_idx = _positive_class_index(model, probs.shape[0])
                pos_idx = min(pos_idx, probs.shape[0] - 1)
                return _clip01(float(probs[pos_idx]))

            return _clip01(float(np.max(flat)))
        except Exception as error:
            errors.append(str(error))
    raise ValueError("predict failed: " + " | ".join(errors))


def predict_image(image_path):
    if not image_model_loaded or image_model is None:
        raise ValueError(
            f"Image model unavailable: {image_model_error or 'not loaded'}"
        )

    img_4d, img_flat = preprocess_image(image_path)

    if hasattr(image_model, "predict_proba"):
        return _proba_from_predict_proba(image_model, img_4d, img_flat)

    if hasattr(image_model, "predict"):
        return _proba_from_predict(image_model, img_4d, img_flat)

    raise ValueError(
        "Unsupported image model interface: expected predict_proba or predict"
    )


