
# import requests

# def download_media(url):
#     response = requests.get(url, timeout=10)

#     if url.lower().endswith((".wav", ".mp3")):
#         file_name = "temp_audio.wav"
#         media_type = "audio"

#     elif url.lower().endswith((".jpg", ".jpeg", ".png")):
#         file_name = "temp_image.jpg"
#         media_type = "image"

#     else:
#         raise ValueError("Unsupported file type")

#     with open(file_name, "wb") as f:
#         f.write(response.content)

#     return file_name, media_type



import requests

def download_media(url):
    url = url.lower()

    if url.endswith((".wav", ".mp3")):
        file_name = "temp_audio.wav"
        media_type = "audio"

    elif url.endswith((".jpg", ".jpeg", ".png", ".webp")):
        file_name = "temp_image.jpg"
        media_type = "image"

    else:
        raise ValueError("Unsupported file type")

    response = requests.get(url, timeout=10)

    with open(file_name, "wb") as f:
        f.write(response.content)

    return file_name, media_type
