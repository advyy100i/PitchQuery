"""Postgres connection helper. Reads DATABASE_URL from .env."""
import functools
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from core.config import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

DEFAULT_URL = "postgresql://pitchquery:pitchquery@localhost:5433/pitchquery"


class Unavailable(RuntimeError):
    """Postgres is not reachable.

    A plain exception rather than SystemExit on purpose. SystemExit derives from
    BaseException, so a Prefect task that hit it would abort the flow run
    instead of being retried — and "the database blinked" is the textbook case
    for `@task(retries=3)`. The friendly message lives in the exception text,
    and `cli()` below turns it back into a clean exit for command-line callers.
    """


def database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_URL)


def connect(autocommit: bool = False) -> psycopg.Connection:
    try:
        return psycopg.connect(database_url(), autocommit=autocommit)
    except psycopg.OperationalError as e:
        raise Unavailable(
            f"cannot reach Postgres at {database_url()}\n"
            f"  {e}\n"
            f"  start it with:  docker compose up -d db"
        ) from e


def cli(fn):
    """Wrap a command-line entry point so an unreachable database prints one
    line and exits 1. Library callers — the Prefect tasks in pipeline/flows.py
    above all — still see `Unavailable` and can retry it."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Unavailable as e:
            print(e, file=sys.stderr)
            raise SystemExit(1)
    return wrapper


def apply_sql_file(conn: psycopg.Connection, path: Path) -> None:
    with conn.cursor() as cur:
        cur.execute(path.read_text(encoding="utf-8"))
    conn.commit()
