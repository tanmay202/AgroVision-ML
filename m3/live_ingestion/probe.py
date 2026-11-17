import httpx

from live_ingestion.ocr import extract_captcha_fields, solve_captcha_from_base64

BASE_URL = "https://api.agmarknet.gov.in/v1"
CAPTCHA_URL = f"{BASE_URL}/captcha/generator"
REPORT_URL = f"{BASE_URL}/daily-price-arrival/report"

HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Origin": "https://agmarknet.gov.in",
    "Referer": "https://agmarknet.gov.in/",
    "Accept": "application/json, text/plain, */*",
}

base_payload = {
    "state_name": "West Bengal",
    "district_name": "Hooghly",
    "cmdt_name": "Potato",
    "from_date": "20-09-2026",
    "to_date": "27-09-2026",
    "page": 1,
    "limit": 10,
}

client = httpx.Client(headers=HEADERS_BASE, timeout=15.0)

# Fetch and solve fresh challenge
resp = client.post(CAPTCHA_URL)
resp.raise_for_status()
key, b64_img = extract_captcha_fields(resp.json())
solution = solve_captcha_from_base64(b64_img)
print(f"[*] Live challenge solved: key={key[:8]}..., value='{solution}'\n")

test_cases = [
    # 1. Form-encoded payload instead of JSON
    (
        "1. Form-Urlencoded (data=payload)",
        {"data": {**base_payload, "captcha_key": key, "captcha_value": solution}},
        {},
    ),
    # 2. Field variations in JSON
    (
        "2. JSON with 'captcha_code'",
        {"json": {**base_payload, "captcha_key": key, "captcha_code": solution}},
        {},
    ),
    (
        "3. JSON with 'captcha' (short)",
        {"json": {**base_payload, "captcha_key": key, "captcha": solution}},
        {},
    ),
    (
        "4. JSON with 'captcha_val'",
        {"json": {**base_payload, "captcha_key": key, "captcha_val": solution}},
        {},
    ),
    (
        "5. JSON with 'captcha_text'",
        {"json": {**base_payload, "captcha_key": key, "captcha_text": solution}},
        {},
    ),
    # 3. Nested object structures
    (
        "6. Nested 'data' envelope",
        {
            "json": {
                "data": {**base_payload, "captcha_key": key, "captcha_value": solution}
            }
        },
        {},
    ),
    (
        "7. Nested 'captcha' object: {'key': ..., 'value': ...}",
        {"json": {**base_payload, "captcha": {"key": key, "value": solution}}},
        {},
    ),
    (
        "8. Nested 'captcha' object: {'captcha_key': ..., 'captcha_value': ...}",
        {
            "json": {
                **base_payload,
                "captcha": {"captcha_key": key, "captcha_value": solution},
            }
        },
        {},
    ),
    # 4. Standard X- headers
    (
        "9. Standard X- headers (x-captcha-key / x-captcha-value)",
        {"json": base_payload},
        {"x-captcha-key": key, "x-captcha-value": solution},
    ),
]

for name, req_kwargs, extra_headers in test_cases:
    h = {**HEADERS_BASE, **extra_headers}
    try:
        r = client.post(REPORT_URL, headers=h, **req_kwargs)
        print(f"--- {name} ---")
        print(f"Status: {r.status_code}")
        print(f"Body: {r.text[:180]}")
        if r.status_code == 200:
            print("[+] MATCH FOUND!")
            break
        elif "TOKEN_OR_CAPTCHA_REQUIRED" not in r.text:
            print("[*] Progress: Error changed (fields recognized)!")
    except Exception as e:
        print(f"Error testing {name}: {e}")
