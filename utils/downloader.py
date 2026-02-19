import os
import tempfile
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB


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


def _build_session():
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "TrustLens-AI/1.0 (+media-fetch)",
            "Accept": "*/*",
        }
    )
    return session


def download_media(url):
    lowered_url = url.lower()
    if "youtube.com" in lowered_url or "youtu.be" in lowered_url:
        return _download_youtube_media(url)

    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path.lower())[1]
    initial_media_type = _media_type_from_ext(ext)

    session = _build_session()
    response = None

    try:
        # Fast preflight to reject non-media links early when possible.
        head = session.head(url, timeout=(8, 12), allow_redirects=True)
        head_content_type = head.headers.get("Content-Type", "")
        if not initial_media_type:
            hinted = infer_media_type("", head_content_type)
            if hinted is None and head_content_type:
                raise ValueError(
                    "Unsupported media type or indirect link. Provide a direct media URL or upload."
                )

        response = session.get(url, timeout=(8, 45), stream=True, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        media_type = initial_media_type or infer_media_type("", content_type)
        if media_type is None:
            raise ValueError(
                "Unsupported media type or indirect link. Provide a direct media URL or upload."
            )

        suffix = (
            ext
            if _media_type_from_ext(ext)
            else _default_suffix_for_media_type(media_type)
        )
        total_bytes = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    raise ValueError("File too large. Maximum allowed size is 200MB.")
                f.write(chunk)
            file_name = f.name

        return file_name, media_type
    except requests.Timeout as error:
        raise ValueError(
            "URL fetch timed out. Try a direct media link, another host, or upload the file."
        ) from error
    except requests.RequestException as error:
        raise ValueError(f"Failed to fetch media URL: {error}") from error
    finally:
        if response is not None:
            response.close()
        session.close()
