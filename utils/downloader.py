import os
import tempfile
from urllib.parse import urlparse

import requests


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _media_type_from_ext(ext):
    ext = (ext or "").lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    return None


def infer_media_type(filename="", content_type=""):
    ext = os.path.splitext((filename or "").lower())[1]
    media_type = _media_type_from_ext(ext)
    if media_type:
        return media_type

    ctype = (content_type or "").lower()
    if ctype.startswith("image/"):
        return "image"
    if ctype.startswith("audio/"):
        return "audio"
    if ctype.startswith("video/"):
        return "video"
    return None


def _default_suffix_for_media_type(media_type):
    if media_type == "image":
        return ".jpg"
    if media_type == "audio":
        return ".wav"
    if media_type == "video":
        return ".mp4"
    return ""


def _download_youtube_media(url):
    try:
        import yt_dlp
    except Exception as error:
        raise ValueError(
            "YouTube URL support requires yt-dlp. Install with: python -m pip install yt-dlp"
        ) from error

    temp_dir = tempfile.mkdtemp(prefix="yt_media_")
    outtmpl = os.path.join(temp_dir, "%(id)s.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "format": "best",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info)

    media_type = infer_media_type(file_path, "")
    if media_type is None:
        # Youtube can occasionally give unknown extensions.
        media_type = "video"

    return file_path, media_type


def download_media(url):
    lowered_url = url.lower()
    if "youtube.com" in lowered_url or "youtu.be" in lowered_url:
        return _download_youtube_media(url)

    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path.lower())[1]
    initial_media_type = _media_type_from_ext(ext)

    response = requests.get(url, timeout=20, stream=True)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    media_type = initial_media_type or infer_media_type("", content_type)
    if media_type is None:
        raise ValueError("Unsupported media type from URL/content-type")

    suffix = ext if _media_type_from_ext(ext) else _default_suffix_for_media_type(media_type)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
        file_name = f.name

    return file_name, media_type
