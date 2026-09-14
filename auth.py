"""
API authentication for the Coetus REST API.

Auth is env-gated and off by default, so local development and the test
suite are unaffected:

- Set ``API_AUTH_TOKENS`` to a comma-separated list of shared secrets to
  protect the configured route prefixes (default: ``/payroll``).
- Accepted credentials: an ``X-API-Key`` header or an
  ``Authorization: Bearer <token>`` header matching any configured token.
- Unset (or empty) ``API_AUTH_TOKENS`` leaves the API fully open.

Tokens are read from the environment per request so configuration changes
and tests do not require a process restart. Comparison uses
``hmac.compare_digest`` to avoid leaking timing information.
"""

import hmac
import os
from typing import Optional, Tuple

DEFAULT_PROTECTED_PREFIXES = "/payroll"


def configured_tokens() -> Tuple[str, ...]:
    """Return configured shared secrets; empty tuple means auth is off."""
    raw = os.environ.get("API_AUTH_TOKENS", "")
    return tuple(token.strip() for token in raw.split(",") if token.strip())


def protected_prefixes() -> Tuple[str, ...]:
    """Path prefixes guarded while auth is enabled (normalized to /x form)."""
    raw = os.environ.get("AUTH_PROTECTED_PREFIXES", DEFAULT_PROTECTED_PREFIXES)
    return tuple("/" + part.strip().strip("/") for part in raw.split(",")
                 if part.strip())


def is_protected_path(path: str) -> bool:
    """True when ``path`` falls under one of the protected prefixes."""
    return any(path == prefix or path.startswith(prefix + "/")
               for prefix in protected_prefixes())


def check_credential(headers) -> Optional[Tuple[int, str]]:
    """
    Validate request headers against the configured tokens.

    Returns ``None`` when the request may proceed, otherwise a
    ``(status_code, message)`` denial:
    - 401 when no credential is presented (with ``WWW-Authenticate``),
    - 403 when the presented credential matches no configured token.
    """
    tokens = configured_tokens()
    if not tokens:
        return None  # auth disabled — open API
    presented = headers.get("x-api-key")
    if not presented:
        authz = headers.get("authorization", "")
        if authz.lower().startswith("bearer "):
            presented = authz[7:].strip()
    if not presented:
        return (401, "Missing API key or bearer token")
    for token in tokens:
        if hmac.compare_digest(presented, token):
            return None
    return (403, "Invalid API key or bearer token")
