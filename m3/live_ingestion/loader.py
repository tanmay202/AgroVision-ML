import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("AgmarknetLoader")

IST = timezone(timedelta(hours=5, minutes=30))

DB_CONFIG = {
    "host": os.getenv("PGHOST", "127.0.0.1"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "agrovision"),
    "user": os.getenv("PGUSER", "agrovision"),
    "password": os.getenv("PGPASSWORD", "postgres"),
}

DISTRICT_NAME_TO_ID: dict[str, str] = {
    "hooghly": "WB_HG_01",
    "purba bardhaman": "WB_BR_01",
    "burdwan": "WB_BR_01",
    "bankura": "WB_BK_01",
    "nadia": "WB_ND_01",
    "birbhum": "WB_BI_01",
    "paschim medinipur": "WB_PM_01",
    "murshidabad": "WB_MU_01",
    "jalpaiguri": "WB_JP_01",
}

MANDI_MAP: dict[str, tuple[str, str]] = {
    "Champadanga APMC": ("WB_MND_CHAMPADANGA", "WB_HG_01"),
    "Kalipur(Arambagh Krishak Bazar)APMC": ("WB_MND_KALIPUR", "WB_HG_01"),
    "Pandua APMC": ("WB_MND_PANDUA", "WB_HG_01"),
    "Sheoraphuly APMC": ("WB_MND_SHEORAPHULY", "WB_HG_01"),
    "Arambagh": ("WB_MND_ARAMBAGH", "WB_HG_01"),
}

CROP_MAP: dict[tuple[str, str], str] = {
    # Potato
    ("Potato", "Jyoti"): "POTATO_JYOTI",
    ("Potato", "Kufri Jyoti"): "POTATO_JYOTI",
    ("Potato", "Chandramukhi"): "POTATO_CHANDRAMUKHI",
    ("Potato", "Chandermukhi"): "POTATO_CHANDRAMUKHI",
    ("Potato", "Other"): "POTATO_OTHER",
    ("Potato", "Local"): "POTATO_LOCAL",
    # Rice / Paddy
    ("Rice", "Common"): "RICE_COMMON",
    ("Rice", "Fine"): "RICE_FINE",
    ("Rice", "Other"): "RICE_OTHER",
    ("Rice", "Sona Mansoori Non Basmati"): "RICE_SONA_MANSOORI",
    ("Rice", "Masuri"): "RICE_MASURI",
    ("Rice", "Masoori-Non Basmti"): "RICE_MASOORI",
    ("Rice", "Ratnachudi (718 5-749)"): "RICE_RATNACHUDI",
    ("Rice", "Jaya"): "RICE_JAYA",
    # Wheat
    ("Wheat", "Sonalika"): "WHEAT_SONALIKA",
    ("Wheat", "Kalyan"): "WHEAT_KALYAN",
    ("Wheat", "Sona"): "WHEAT_SONA",
    ("Wheat", "Other"): "WHEAT_OTHER",
    ("Wheat", "Local"): "WHEAT_LOCAL",
    ("Wheat", "PBW-343"): "WHEAT_PBW343",
    ("Wheat", "HD-2967"): "WHEAT_HD2967",
}

UPSERT_MARKET_FEATURES_SQL = """
INSERT INTO features.market_features (
    time,
    mandi_id,
    district_id,
    crop_id,
    modal_price_inr,
    arrivals_mt
)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (time, mandi_id, crop_id)
DO UPDATE SET
    modal_price_inr = EXCLUDED.modal_price_inr,
    arrivals_mt     = EXCLUDED.arrivals_mt,
    district_id     = EXCLUDED.district_id;
"""


def parse_ist_date(date_str: str) -> datetime:
    dt = datetime.strptime(date_str.strip(), "%d-%m-%Y")
    return dt.replace(tzinfo=IST)


def clean_float(value: Any) -> float:
    if value is None:
        return 0.0
    val_str = str(value).replace(",", "").strip()
    return float(val_str) if val_str else 0.0


def _slugify_mandi_id(market_name: str) -> str:
    clean = re.sub(r"\bAPMC\b", "", market_name, flags=re.IGNORECASE)
    slug = re.sub(r"[^A-Z0-9]+", "_", clean.upper()).strip("_")
    return f"WB_MND_{slug}"[:32]


def _resolve_crop_id(cmdt: str, variety: str) -> str | None:
    if (cmdt, variety) in CROP_MAP:
        return CROP_MAP[(cmdt, variety)]
    c_up = re.sub(r"[^A-Z0-9]+", "_", cmdt.upper()).strip("_")
    v_up = re.sub(r"[^A-Z0-9]+", "_", variety.upper()).strip("_")
    if c_up in ("RICE", "POTATO", "WHEAT") and v_up:
        return f"{c_up}_{v_up}"[:32]
    return None


def load_market_features(records: list[dict[str, Any]]) -> int:
    if not records:
        logger.info("No records provided to load.")
        return 0

    new_mandis: dict[str, tuple[str, str]] = {}
    new_crops: dict[str, tuple[str, str, float]] = {}
    tuples_to_insert: list[tuple] = []
    skipped_count = 0

    for row in records:
        raw_market = row.get("market_name", "").strip()
        raw_district = row.get("district_name", "").strip()
        cmdt = row.get("cmdt_name", "").strip()
        variety = row.get("variety_name", "").strip()
        raw_date = row.get("arrival_date", "").strip()

        if raw_market in MANDI_MAP:
            mandi_id, district_id = MANDI_MAP[raw_market]
        else:
            district_id = DISTRICT_NAME_TO_ID.get(raw_district.lower())
            if not district_id or not raw_market:
                skipped_count += 1
                continue
            mandi_id = _slugify_mandi_id(raw_market)
            new_mandis[mandi_id] = (raw_market, district_id)

        crop_id = _resolve_crop_id(cmdt, variety)
        if not crop_id:
            skipped_count += 1
            continue
        season = "Kharif" if cmdt.lower() == "rice" else "Rabi"
        gdd = 10.0 if cmdt.lower() == "rice" else (7.0 if cmdt.lower() == "potato" else 5.0)
        new_crops[crop_id] = (f"{cmdt} ({variety})", season, gdd)

        try:
            obs_time = parse_ist_date(raw_date)
            raw_price = row.get("model_price") or row.get("modal_price") or 0.0
            modal_price = clean_float(raw_price)
            arrivals = clean_float(row.get("arrival_qty", 0.0))
        except (ValueError, TypeError):
            skipped_count += 1
            continue

        tuples_to_insert.append(
            (obs_time, mandi_id, district_id, crop_id, modal_price, arrivals)
        )

    if not tuples_to_insert:
        logger.warning("No records survived transformation (skipped=%d).", skipped_count)
        return 0

    with psycopg.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        for cid, (cname, season, gdd) in new_crops.items():
            cur.execute(
                """
                INSERT INTO features.crop_dim (crop_id, canonical_name, season_type, base_temperature_gdd)
                VALUES (%s, %s, %s, %s) ON CONFLICT (crop_id) DO NOTHING;
                """,
                (cid, cname, season, gdd),
            )
        for mid, (mname, did) in new_mandis.items():
            cur.execute(
                """
                INSERT INTO features.mandi_dim (mandi_id, mandi_name, district_id)
                VALUES (%s, %s, %s) ON CONFLICT (mandi_id) DO NOTHING;
                """,
                (mid, mname, did),
            )
        cur.executemany(UPSERT_MARKET_FEATURES_SQL, tuples_to_insert)

    logger.info(
        "[+] Committed %d live Agmarknet rows (skipped=%d).",
        len(tuples_to_insert),
        skipped_count,
    )
    return len(tuples_to_insert)
