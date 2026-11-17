import base64
import io
import string
import subprocess
from typing import Any

from PIL import Image

ALPHANUMERIC_WHITELIST = string.ascii_letters + string.digits


def preprocess_image(image_bytes: bytes) -> bytes:
    """
    Enhances low-res CAPTCHA images in-memory for Tesseract:
    1. Converts to grayscale.
    2. Upscales 3x with LANCZOS interpolation (simulates 300 DPI text).
    3. Applies point-thresholding to strip background noise lines.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        # Grayscale
        gray = img.convert("L")
        # Upscale 3x
        w, h = gray.size
        scaled = gray.resize((w * 3, h * 3), Image.Resampling.LANCZOS)
        # Threshold: push dark text to 0, wash out light noise to 255
        thresholded = scaled.point(lambda p: 0 if p < 145 else 255, mode="1")

        out_buffer = io.BytesIO()
        thresholded.save(out_buffer, format="PNG")
        return out_buffer.getvalue()


def solve_captcha_from_base64(b64_image_str: str) -> str:
    """
    Solves an Agmarknet Base64 CAPTCHA in-memory using native Tesseract.
    Zero disk writes; executes via kernel stdin/stdout pipes.
    """
    if not b64_image_str or not isinstance(b64_image_str, str):
        raise ValueError(
            f"Expected valid base64 string, received: {type(b64_image_str)}"
        )

    if "," in b64_image_str:
        b64_image_str = b64_image_str.split(",", 1)[1]

    raw_bytes = base64.b64decode(b64_image_str)
    processed_bytes = preprocess_image(raw_bytes)

    cmd = [
        "tesseract",
        "stdin",
        "stdout",
        "--psm",
        "8",
        "-c",
        f"tessedit_char_whitelist={ALPHANUMERIC_WHITELIST}",
    ]

    result = subprocess.run(
        cmd,
        input=processed_bytes,
        capture_output=True,
        check=True,
    )

    return result.stdout.decode("utf-8").strip()


def extract_captcha_fields(payload: dict[str, Any]) -> tuple[str, str]:
    """
    Extracts (captcha_key, b64_image) from flat or nested responses.
    """
    data_block = payload.get("data")
    source: dict[str, Any] = data_block if isinstance(data_block, dict) else payload

    key = source.get("captcha_key")
    image = source.get("captcha_image") or source.get("image")

    if not key or not image:
        raise KeyError(
            f"Unable to locate CAPTCHA fields. Keys found: {list(payload.keys())}"
        )

    return str(key), str(image)


if __name__ == "__main__":
    import httpx

    print("[*] Fetching live CAPTCHA from Agmarknet 2.0 API...")
    url = "https://api.agmarknet.gov.in/v1/captcha/generator"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Origin": "https://agmarknet.gov.in",
        "Referer": "https://agmarknet.gov.in/",
    }

    resp = httpx.post(url, headers=headers, timeout=10.0)
    resp.raise_for_status()
    payload = resp.json()

    captcha_key, captcha_b64 = extract_captcha_fields(payload)
    print(f"[+] CAPTCHA Key: {captcha_key}")

    solved = solve_captcha_from_base64(captcha_b64)
    print(f"[+] Preprocessed OCR Solution: '{solved}' (Length: {len(solved)})")
