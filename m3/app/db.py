import logging
import os
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, Request

logger = logging.getLogger("AgroVisionAPI.DB")

DB_CONFIG = {
    "host": os.getenv("PGHOST", "127.0.0.1"),
    "port": int(os.getenv("PGPORT", "5432")),
    "database": os.getenv("PGDATABASE", "agrovision"),
    "user": os.getenv("PGUSER", "agrovision"),
    "password": os.getenv("PGPASSWORD", "postgres"),
    "min_size": 2,
    "max_size": 10,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages connection pool creation on API startup and clean teardown on shutdown.
    """
    logger.info(
        "Initializing asyncpg connection pool to %s:%s...",
        DB_CONFIG["host"],
        DB_CONFIG["port"],
    )
    try:
        pool = await asyncpg.create_pool(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            database=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            min_size=DB_CONFIG["min_size"],
            max_size=DB_CONFIG["max_size"],
        )
        app.state.db_pool = pool
        logger.info("[+] asyncpg connection pool established.")
    except Exception as exc:
        logger.error("Failed to connect to PostgreSQL: %s", exc)
        raise exc

    yield  # Application serves requests here

    logger.info("Closing asyncpg connection pool...")
    await app.state.db_pool.close()
    logger.info("[+] asyncpg pool closed cleanly.")


async def get_db(request: Request):
    """
    FastAPI dependency yielding a single connection from the pool per request.
    """
    pool: asyncpg.Pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        yield conn
