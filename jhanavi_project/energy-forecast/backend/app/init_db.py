# backend/app/init_db.py
import os
from typing import TYPE_CHECKING, Any, cast

# Import sqlalchemy only at runtime; provide TYPE_CHECKING imports for type checkers.
if TYPE_CHECKING:
    # Provide type hints to tools when available
    try:
        from sqlalchemy.engine import Engine, Connection  # type: ignore
    except Exception:  # pragma: no cover - typing fallback for editors
        Engine = Any  # type: ignore
    from typing import Any, cast

    # Import SQLAlchemy types when available so static analyzers (Pylance) know the
    # types of Engine/Connection. If SQLAlchemy isn't installed in the environment
    # we fall back to Any which keeps runtime behavior safe.
    try:
        from sqlalchemy.engine import Engine, Connection  # type: ignore
    except Exception:  # pragma: no cover - fallback for editors without SQLAlchemy
        Engine = Any  # type: ignore
        Connection = Any  # type: ignore
    create_engine = None  # type: ignore
    text = None  # type: ignore

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://energy:energy_pass@db:5432/energydb",
)


def init_db() -> None:
    """Initialize the database using the SQL in `sql/init.sql`.

    If sqlalchemy is not installed in the current environment (for example the
    editor language server uses a different interpreter), this function will
    safely no-op and print a helpful message.
    """
    if create_engine is None or text is None:
        print("sqlalchemy not available in this environment; skipping DB init.")
        return

    # Tell static checkers that this is a SQLAlchemy Engine so members like
    # `connect()` and `connect().execute()` are recognized. We still guard at
    # runtime for missing `create_engine`.
    engine = cast("Engine", create_engine(DATABASE_URL))  # type: ignore[arg-type]
    # Use context manager to open the SQL file and execute it on the connection
    with engine.connect() as conn:
        with open("sql/init.sql", "r") as fh:
            sql = fh.read()
        conn.execute(text(sql))

    print("DB initialized (if not exists).")


if __name__ == "__main__":
    init_db()
