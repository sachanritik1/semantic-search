# TODO: rewrite with Postgres integration tests (testcontainers).
# SQLite in-memory fixtures were removed in the PostgreSQL migration.

import pytest

pytestmark = pytest.mark.skip(
    reason="DB-layer tests pending Postgres rewrite (see tests/conftest.py)"
)


def test_placeholder():
    pass
