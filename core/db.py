"""Postgres connection helper. Reads DATABASE_URL from .env."""
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from core.config import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

DEFAULT_URL = "postgresql://pitchquery:pitchquery@localhost:5433/pitchquery"


def database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_URL)


def connect(autocommit: bool = False) -> psycopg.Connection:
    try:
        return psycopg.connect(database_url(), autocommit=autocommit)
    except psycopg.OperationalError as e:
        print(
            f"cannot reach Postgres at {database_url()}\n"
            f"  {e}\n"
            f"  start it with:  docker compose up -d db",
            file=sys.stderr,
        )
        raise SystemExit(1)


def apply_sql_file(conn: psycopg.Connection, path: Path) -> None:
    with conn.cursor() as cur:
        cur.execute(path.read_text(encoding="utf-8"))
    conn.commit()
