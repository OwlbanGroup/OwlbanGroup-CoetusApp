"""Smoke test for the /classify AI endpoint.

Run standalone: python _classify_smoke.py
"""
import io

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

with open("test_image.jpg", "rb") as fh:
    data = fh.read()

resp = client.post(
    "/classify",
    files={"file": ("test_image.jpg", io.BytesIO(data), "image/jpeg")},
)
print(resp.status_code, resp.json())
