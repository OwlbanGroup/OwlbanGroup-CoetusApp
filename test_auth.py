"""
Tests for the env-gated API authentication layer.

Run with: python -m pytest test_auth.py -v
"""

import pytest
from fastapi.testclient import TestClient

# Import fresh_store from test_payroll but ensure it doesn't conflict with auth module
try:
    from test_payroll import fresh_store
except ImportError:
    # Fallback if test_payroll isn't available
    def fresh_store():
        """No-op fresh store for testing."""
        pass


TOKEN_A = "secret-token-abc123"
TOKEN_B = "other-secret-456"


def get_client():
    """Return a fresh TestClient bound to the FastAPI app.

    Uses a factory function instead of lru_cache to ensure test isolation
    and proper environment variable handling between tests.
    """
    from app import app  # pylint: disable=import-outside-toplevel
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure token configuration never leaks between tests."""
    monkeypatch.delenv("API_AUTH_TOKENS", raising=False)
    monkeypatch.delenv("AUTH_PROTECTED_PREFIXES", raising=False)
    fresh_store()


def test_auth_disabled_by_default(monkeypatch):
    """Without API_AUTH_TOKENS, payroll routes need no credentials."""
    monkeypatch.delenv("API_AUTH_TOKENS", raising=False)
    resp = get_client().get("/payroll/employees")
    assert resp.status_code == 200


def test_payroll_requires_credential_when_enabled(monkeypatch):
    """With tokens configured, an unauthenticated call gets 401."""
    monkeypatch.setenv("API_AUTH_TOKENS", TOKEN_A)
    resp = get_client().get("/payroll/employees")
    assert resp.status_code == 401
    assert resp.json() == {"error": "Missing API key or bearer token"}
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_payroll_wrong_credential_403(monkeypatch):
    """A presented-but-unknown token is rejected with 403 (no 401 hint)."""
    monkeypatch.setenv("API_AUTH_TOKENS", TOKEN_A)
    resp = get_client().get("/payroll/employees",
                            headers={"X-API-Key": "nope"})
    assert resp.status_code == 403
    assert resp.json() == {"error": "Invalid API key or bearer token"}


def test_payroll_valid_api_key_accepted(monkeypatch):
    """A configured X-API-Key grants access."""
    monkeypatch.setenv("API_AUTH_TOKENS", f"{TOKEN_A},{TOKEN_B}")
    resp = get_client().get("/payroll/employees",
                            headers={"X-API-Key": TOKEN_A})
    assert resp.status_code == 200
    resp = get_client().get("/payroll/employees",
                            headers={"X-API-Key": TOKEN_B})
    assert resp.status_code == 200


def test_payroll_valid_bearer_accepted(monkeypatch):
    """A configured Authorization: Bearer token grants access."""
    monkeypatch.setenv("API_AUTH_TOKENS", TOKEN_A)
    resp = get_client().get(
        "/payroll/employees",
        headers={"Authorization": f"Bearer {TOKEN_B}"})
    assert resp.status_code == 200


def test_unprotected_routes_stay_open(monkeypatch):
    """/health and /synthetic are outside the default protected prefix."""
    monkeypatch.setenv("API_AUTH_TOKENS", TOKEN_A)
    c = get_client()
    assert c.get("/health").status_code == 200
    assert c.get("/synthetic/capabilities").status_code == 200


def test_options_preflight_passes_through(monkeypatch):
    """CORS preflight OPTIONS requests are not blocked by the auth gate."""
    monkeypatch.setenv("API_AUTH_TOKENS", TOKEN_A)
    resp = get_client().options("/payroll/employees", headers={
        "Origin": "https://hr.example.com",
        "Access-Control-Request-Method": "POST",
    })
    # Whatever the router answers, the auth layer must not 401/403 it.
    assert resp.status_code not in (401, 403)


def test_custom_protected_prefixes(monkeypatch):
    """AUTH_PROTECTED_PREFIXES extends the guarded surface."""
    monkeypatch.setenv("API_AUTH_TOKENS", TOKEN_A)
    monkeypatch.setenv("AUTH_PROTECTED_PREFIXES", "/payroll,/synthetic")
    c = get_client()
    denied = c.get("/synthetic/capabilities")
    assert denied.status_code == 401
    allowed = c.get("/synthetic/capabilities",
                    headers={"X-API-Key": TOKEN_A})
    assert allowed.status_code == 200


def test_openapi_includes_security_schemes_when_enabled(monkeypatch):
    """The OpenAPI schema advertises both credential styles."""
    c = get_client()
    monkeypatch.delenv("API_AUTH_TOKENS", raising=False)
    c.app.openapi_schema = None  # bust the cached schema
    schema_open = c.get("/openapi.json").json()
    assert "securitySchemes" not in schema_open.get("components", {})
    monkeypatch.setenv("API_AUTH_TOKENS", TOKEN_A)
    c.app.openapi_schema = None
    schema_secured = c.get("/openapi.json").json()
    names = schema_secured["components"]["securitySchemes"]
    assert set(names) == {"ApiKeyAuth", "BearerAuth"}
    c.app.openapi_schema = None  # leave the cache empty for other tests
