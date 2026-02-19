import os
import joblib
import sys
import types
import marshal
import io
import json
import zipfile
import tempfile
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


def _sanitize_layer_configs(node):
    if isinstance(node, dict):
        class_name = node.get("class_name")
        config = node.get("config")
        if isinstance(config, dict) and class_name not in {"InputLayer"}:
            config.pop("batch_input_shape", None)
            config.pop("batch_shape", None)
            config.pop("input_shape", None)

        for value in node.values():
            _sanitize_layer_configs(value)
    elif isinstance(node, list):
        for item in node:
            _sanitize_layer_configs(item)


def _load_model_from_keras_archive_bytes(config_bytes):
    with tempfile.TemporaryDirectory(prefix="keras_pickle_fix_") as tmp_dir:
        extracted_dir = os.path.join(tmp_dir, "extracted")
        os.makedirs(extracted_dir, exist_ok=True)

        # Extract incoming .keras zip bytes.
        with zipfile.ZipFile(io.BytesIO(config_bytes), "r") as zf:
            zf.extractall(extracted_dir)

        config_json_path = os.path.join(extracted_dir, "config.json")
        if os.path.exists(config_json_path):
            with open(config_json_path, "r", encoding="utf-8") as f:
                config_payload = json.load(f)

            _sanitize_layer_configs(config_payload)

            with open(config_json_path, "w", encoding="utf-8") as f:
                json.dump(config_payload, f)

        patched_path = os.path.join(tmp_dir, "patched.keras")
        with zipfile.ZipFile(patched_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(extracted_dir):
                for file_name in files:
                    abs_path = os.path.join(root, file_name)
                    rel_path = os.path.relpath(abs_path, extracted_dir)
                    zf.write(abs_path, rel_path)

        try:
            return load_model(patched_path, compile=False)
        except TypeError:
            return load_model(patched_path, compile=False, safe_mode=False)


def _load_model_from_keras_bytes_with_fallback(config_bytes):
    primary_error = None

    try:
        return _load_model_from_keras_archive_bytes(config_bytes)
    except Exception as error:
        primary_error = error

    # Fallback: try loading raw .keras bytes as-is.
    with tempfile.TemporaryDirectory(prefix="keras_pickle_raw_") as tmp_dir:
        raw_path = os.path.join(tmp_dir, "raw.keras")
        with open(raw_path, "wb") as f:
            f.write(config_bytes)
        try:
            return load_model(raw_path, compile=False)
        except TypeError:
            try:
                return load_model(raw_path, compile=False, safe_mode=False)
            except Exception as fallback_error:
                raise ValueError(
                    "Unable to deserialize Keras model from pickle bytecode. "
                    f"patched-load error: {primary_error}; raw-load error: {fallback_error}"
                ) from fallback_error
        except Exception as fallback_error:
            raise ValueError(
                "Unable to deserialize Keras model from pickle bytecode. "
                f"patched-load error: {primary_error}; raw-load error: {fallback_error}"
            ) from fallback_error


def load_image_model():
    configured = os.getenv("IMAGE_MODEL_PATH", "").strip()
    service_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if configured:
        configured_path = _resolve_model_path(configured)
        candidates = [configured_path]
        if not os.path.isabs(configured):
            root_alt = os.path.join(
                os.path.dirname(service_root), os.path.basename(configured_path)
            )
            if root_alt not in candidates:
                candidates.append(root_alt)
    else:
        candidates = [
            os.path.join(service_root, "models", "deepfake_image_model.pkl"),
            os.path.join(service_root, "models", "deepfake_image_model.h5"),
            os.path.join(os.path.dirname(service_root), "deepfake_image_model.pkl"),
        ]

    errors = []

    for model_path in candidates:
        if not os.path.exists(model_path):
            errors.append(f"{model_path}: not found")
            continue

        try:
            if model_path.lower().endswith(".pkl"):
                try:
                    return joblib.load(model_path)
                except ModuleNotFoundError as error:
                    missing = getattr(error, "name", "") or ""
                    if missing != "keras.src.saving.pickle_utils":
                        raise

                    # Compatibility shim for Keras pickle payloads that reference
                    # keras.src.saving.pickle_utils.deserialize_model_from_bytecode.
                    module_name = "keras.src.saving.pickle_utils"
                    if module_name not in sys.modules:
                        shim = types.ModuleType(module_name)

                        def deserialize_model_from_bytecode(config, metadata=None):
                            try:
                                # Keras pickles may store a full .keras archive as bytes.
                                if isinstance(config, (bytes, bytearray)):
                                    # If this isn't zip bytes, attempt marshal decode fallback.
                                    if bytes(config[:2]) != b"PK":
                                        decoded = marshal.loads(config)
                                        if isinstance(decoded, (bytes, bytearray)):
                                            decoded = decoded.decode("utf-8")
                                        if isinstance(decoded, str):
                                            return model_from_json(decoded)
                                    return _load_model_from_keras_bytes_with_fallback(config)

                                if isinstance(config, str):
                                    return model_from_json(config)
                            except Exception as deserialize_error:
                                raise ValueError(
                                    "Unable to deserialize Keras model from pickle bytecode. "
                                    f"reason: {deserialize_error}"
                                ) from deserialize_error
                            raise ValueError(
                                "Unable to deserialize Keras model from pickle bytecode"
                            )

                        shim.deserialize_model_from_bytecode = (
                            deserialize_model_from_bytecode
                        )
                        sys.modules[module_name] = shim

                    return joblib.load(model_path)

            model = load_model(model_path, compile=False)
            return model
        except Exception as error:
            message = str(error)
            if "No model config found" in message and model_path.lower().endswith(".h5"):
                arch_path = _get_image_arch_path()
                if arch_path and os.path.exists(arch_path):
                    with open(arch_path, "r", encoding="utf-8") as f:
                        model_json = f.read()
                    model = model_from_json(model_json)
                    model.load_weights(model_path)
                    return model

            errors.append(f"{model_path}: {message}")
            continue

    raise ValueError("Image model unavailable. Tried: " + " | ".join(errors))

image_model = None
image_model_loaded = False
image_model_error = None

try:
    image_model = load_image_model()
    image_model_loaded = True
except Exception as error:
    image_model_error = str(error)
