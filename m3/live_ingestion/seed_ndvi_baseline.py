from collections import defaultdict
from datetime import date
import os
import time
import httpx
import psycopg
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

LAT, LON = 22.90, 88.39
DISTRICT_ID = "WB_HG_01"
PRODUCTS = ("MOD13Q1", "MYD13Q1")  # Terra + Aqua = 8-day staggered composites

DB = dict(
    host=os.getenv("PGHOST", "127.0.0.1"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "agrovision"),
    user=os.getenv("PGUSER", "agrovision"),
    password=os.getenv("PGPASSWORD", "postgres"),
)

TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=16),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    reraise=True,
)
def fetch_subset_chunk(client: httpx.Client, product: str, start_code: str, end_code: str) -> list[dict]:
    resp = client.get(
        f"https://modis.ornl.gov/rst/api/v1/{product}/subset",
        params={
            "latitude": LAT,
            "longitude": LON,
            "band": "250m_16_days_NDVI",
            "startDate": start_code,
            "endDate": end_code,
            "kmAboveBelow": 1,
            "kmLeftRight": 1,
        },
    )
    resp.raise_for_status()
    return resp.json().get("subset", [])


def fetch_climatology() -> dict[int, float]:
    doy_samples: dict[int, list[float]] = defaultdict(list)

    with httpx.Client(timeout=TIMEOUT) as client:
        for product in PRODUCTS:
            r_dates = client.get(
                f"https://modis.ornl.gov/rst/api/v1/{product}/dates",
                params={"latitude": LAT, "longitude": LON},
            )
            r_dates.raise_for_status()
            dates = [
                d for d in r_dates.json().get("dates", [])
                if "2021-01-01" <= d["calendar_date"] <= "2025-12-31"
            ]
            print(f"{product}: fetching {len(dates)} composite dates (2021–2025) in 4-date chunks...")

            for i in range(0, len(dates), 4):
                chunk = dates[i : i + 4]
                subset_rows = fetch_subset_chunk(
                    client, product, chunk[0]["modis_date"], chunk[-1]["modis_date"]
                )
                for entry in subset_rows:
                    cal = date.fromisoformat(entry["calendar_date"])
                    doy = min(cal.timetuple().tm_yday, 365)
                    valid = [px * 0.0001 for px in entry["data"] if -2000 <= px <= 10000]
                    if valid:
                        doy_samples[doy].append(sum(valid) / len(valid))
                time.sleep(0.4)  # Polite pacing to avoid ORNL DAAC IP throttling

    anchors = sorted((doy, sum(vals) / len(vals)) for doy, vals in doy_samples.items())
    print(f"Computed {len(anchors)} real multi-year satellite DOY anchors.")

    daily_baseline: dict[int, float] = {}
    for idx in range(len(anchors)):
        d0, v0 = anchors[idx]
        d1, v1 = anchors[(idx + 1) % len(anchors)]
        span = (d1 - d0) if d1 > d0 else (366 - d0 + d1)
        for step in range(span):
            doy = ((d0 - 1 + step) % 366) + 1
            val = v0 + (v1 - v0) * (step / span)
            daily_baseline[doy] = round(val, 4)

    return daily_baseline


def main() -> None:
    baseline = fetch_climatology()
    rows = [(DISTRICT_ID, doy, baseline[doy]) for doy in sorted(baseline)]
    with psycopg.connect(**DB) as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO features.ndvi_baseline (district_id, day_of_year, multi_year_mean_ndvi)
            VALUES (%s, %s, %s)
            ON CONFLICT (district_id, day_of_year) DO UPDATE
            SET multi_year_mean_ndvi = EXCLUDED.multi_year_mean_ndvi;
            """,
            rows,
        )
        cur.execute(
            "SELECT COUNT(*), MIN(multi_year_mean_ndvi), MAX(multi_year_mean_ndvi) "
            "FROM features.ndvi_baseline WHERE district_id = %s;",
            (DISTRICT_ID,),
        )
        print("Saved WB_HG_01 ndvi_baseline (count, min, max):", cur.fetchone())


if __name__ == "__main__":
    main()
