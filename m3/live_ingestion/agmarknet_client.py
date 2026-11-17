import logging
import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from live_ingestion.ocr import extract_captcha_fields, solve_captcha_from_base64

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("AgmarknetClient")

BASE_URL = "https://api.agmarknet.gov.in/v1"
CAPTCHA_URL = f"{BASE_URL}/captcha/generator"
REPORT_URL = f"{BASE_URL}/daily-price-arrival/report"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Origin": "https://agmarknet.gov.in",
    "Referer": "https://agmarknet.gov.in/",
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
}

PILOT_IDS = {
    "state_wb": "[36]",
    "district_hooghly": "[681]",
    "districts_wb_4": "[675,681,690,694]",  # Bankura(675), Hooghly(681), Murshidabad(690), Purba Bardhaman(694)
    "group_vegetables": "6",
    "commodity_potato": "24",
    "group_cereals": "1",
    "commodity_rice": "3",
    "commodity_wheat": "1",
    "data_type_daily": "100006",
    "market_all": "[100002]",
    "grade_all": "[100003]",
    "variety_all": "[100007]",
}

COMMODITY_SPECS: dict[str, tuple[str, str]] = {
    "rice": (PILOT_IDS["group_cereals"], PILOT_IDS["commodity_rice"]),
    "wheat": (PILOT_IDS["group_cereals"], PILOT_IDS["commodity_wheat"]),
    "potato": (PILOT_IDS["group_vegetables"], PILOT_IDS["commodity_potato"]),
}


class AgmarknetAPIError(Exception):
    pass


class AgmarknetClient:
    def __init__(self, timeout: float = 20.0):
        self.client = httpx.Client(headers=DEFAULT_HEADERS, timeout=timeout)
        self.report_ticket: str | None = None
        self.current_filter_hash: tuple | None = None

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5) + wait_random(0, 1),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    )
    def _fetch_and_solve_captcha(self) -> tuple[str, str]:
        resp = self.client.post(CAPTCHA_URL)
        resp.raise_for_status()
        payload = resp.json()

        key, b64_img = extract_captcha_fields(payload)
        solution = solve_captcha_from_base64(b64_img)
        logger.info("Fetched CAPTCHA (key=%s...), solved: '%s'", key[:8], solution)
        return key, solution

    def _post_report_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            if not self.report_ticket:
                cap_key, cap_val = self._fetch_and_solve_captcha()
                payload["captcha_key"] = cap_key
                payload["captcha"] = cap_val
                payload.pop("report_ticket", None)
                time.sleep(1.2)
            else:
                payload["report_ticket"] = self.report_ticket
                payload.pop("captcha_key", None)
                payload.pop("captcha", None)

            resp = self.client.post(REPORT_URL, json=payload)

            if resp.status_code == 200:
                data = resp.json()
                access = data.get("report_access") or {}
                if "report_ticket" in access:
                    self.report_ticket = access["report_ticket"]
                    logger.info(
                        "Acquired/Refreshed session ticket: %s... (remaining: %s)",
                        self.report_ticket[:8],
                        access.get("remaining_calls"),
                    )
                return data

            logger.warning(
                "Agmarknet request failed (status=%d, attempt %d/%d). Server message: %s",
                resp.status_code,
                attempt,
                max_attempts,
                resp.text[:200],
            )

            if resp.status_code in (400, 401):
                self.report_ticket = None
                time.sleep(1.0)
                continue

            resp.raise_for_status()

        raise AgmarknetAPIError(
            f"Failed to query report after {max_attempts} attempts."
        )

    def fetch_commodity_records(
        self,
        commodity: str,
        from_date: str,
        to_date: str,
        district_ids: str = PILOT_IDS["districts_wb_4"],
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        comm_key = commodity.strip().lower()
        if comm_key not in COMMODITY_SPECS:
            raise ValueError(f"Unsupported commodity '{commodity}'")
        group_id, cmdt_id = COMMODITY_SPECS[comm_key]

        filter_identity = ("WB", district_ids, comm_key, from_date, to_date, str(limit))
        if self.current_filter_hash != filter_identity:
            self.report_ticket = None
            self.current_filter_hash = filter_identity

        all_records: list[dict[str, Any]] = []
        page = 1

        while True:
            payload: dict[str, Any] = {
                "from_date": from_date,
                "to_date": to_date,
                "data_type": PILOT_IDS["data_type_daily"],
                "group": group_id,
                "commodity": cmdt_id,
                "state": PILOT_IDS["state_wb"],
                "district": district_ids,
                "market": PILOT_IDS["market_all"],
                "grade": PILOT_IDS["grade_all"],
                "variety": PILOT_IDS["variety_all"],
                "page": str(page),
                "limit": str(limit),
            }

            logger.info(
                "Querying page %d (limit=%d) for WB %s (%s -> %s)...",
                page,
                limit,
                comm_key.upper(),
                from_date,
                to_date,
            )
            response_json = self._post_report_with_retry(payload)

            raw_records = response_json.get("data", {}).get("records", [])
            page_extracted_rows = []

            for block in raw_records:
                if isinstance(block, dict) and "data" in block:
                    inner_rows = block["data"]
                    if isinstance(inner_rows, list):
                        page_extracted_rows.extend(inner_rows)
                elif isinstance(block, dict):
                    page_extracted_rows.append(block)

            if not page_extracted_rows:
                logger.info("Page %d returned 0 records. Drainage complete.", page)
                break

            all_records.extend(page_extracted_rows)
            logger.info(
                "Extracted %d rows from page %d (total accumulated: %d)",
                len(page_extracted_rows),
                page,
                len(all_records),
            )

            if len(page_extracted_rows) < limit:
                break

            page += 1

        return all_records

    def fetch_hooghly_potatoes(
        self,
        from_date: str = "2026-10-01",
        to_date: str = "2026-10-23",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self.fetch_commodity_records(
            "potato", from_date, to_date, PILOT_IDS["district_hooghly"], limit
        )

    def fetch_hooghly_rice(
        self,
        from_date: str = "2026-08-01",
        to_date: str = "2026-11-01",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self.fetch_commodity_records(
            "rice", from_date, to_date, PILOT_IDS["district_hooghly"], limit
        )
