"""
Unit tests for the synthetic human training-data generators and their
FastAPI endpoints.

Run with: python -m pytest test_synthetic_data.py -v
"""

import csv
import io
import json
import wave
from functools import lru_cache

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import synthetic_data as sd
from synthetic_data import (
    GENERATOR_VERSION,
    generate_dialogue,
    generate_face_image,
    generate_profiles,
    generate_voice_clip,
    dataset_manifest,
    profiles_to_csv,
    profiles_to_jsonl,
    render_face_png,
    render_voice_wav,
)


# ---------------------------------------------------------------------------
# Profile generator tests
# ---------------------------------------------------------------------------

def test_profiles_count_and_schema():
    """Each generated profile has the full schema and synthetic flag."""
    profiles = generate_profiles(count=7, seed=1)
    assert len(profiles) == 7
    required = {"id", "first_name", "last_name", "full_name", "age", "gender",
                "email", "phone", "street", "city", "state", "occupation",
                "synthetic"}
    for i, p in enumerate(profiles):
        assert required.issubset(p.keys())
        assert p["id"] == i + 1
        assert p["synthetic"] is True
        assert 18 <= p["age"] <= 85
        assert p["gender"] in ("male", "female")
        assert p["email"].endswith((".com", ".org", ".net"))
        assert "@" in p["email"]
        # Phone format ddd-ddd-dddd
        parts = p["phone"].split("-")
        assert len(parts) == 3 and all(part.isdigit() for part in parts)
        assert p["full_name"] == f"{p['first_name']} {p['last_name']}"


def test_profiles_seeded_reproducibility():
    """The same seed produces byte-identical datasets."""
    a = generate_profiles(count=25, seed=123)
    b = generate_profiles(count=25, seed=123)
    assert a == b
    c = generate_profiles(count=25, seed=124)
    assert a != c


def test_profile_emails_unique():
    """No two profiles in a large batch share an email."""
    profiles = generate_profiles(count=500, seed=9)
    emails = [p["email"] for p in profiles]
    assert len(emails) == len(set(emails))


def test_profiles_invalid_count():
    """Counts outside 1..10000 raise ValueError."""
    with pytest.raises(ValueError):
        generate_profiles(count=0)
    with pytest.raises(ValueError):
        generate_profiles(count=10001)


def test_profiles_unsupported_locale():
    """Unknown locales are rejected."""
    with pytest.raises(ValueError):
        generate_profiles(count=1, locale="fr")


def test_profiles_csv_export():
    """CSV export emits a header plus one row per profile."""
    profiles = generate_profiles(count=3, seed=5)
    text = profiles_to_csv(profiles)
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][0] == "id"
    assert len(rows) == 4  # header + 3
    assert rows[1][12] == "true"  # synthetic flag serialized


def test_profiles_jsonl_export():
    """JSONL export emits one JSON object per line."""
    profiles = generate_profiles(count=4, seed=6)
    text = profiles_to_jsonl(profiles)
    lines = [line for line in text.splitlines() if line]
    assert len(lines) == 4
    parsed = json.loads(lines[0])
    assert parsed["id"] == 1
    assert parsed["synthetic"] is True


# ---------------------------------------------------------------------------
# Dialogue generator tests
# ---------------------------------------------------------------------------

def test_dialogue_schema_and_alternating_speakers():
    """Turns alternate user/assistant and start with the user."""
    dialogues = generate_dialogue(count=5, seed=2)
    assert len(dialogues) == 5
    for d in dialogues:
        assert d["intent"] in ("greeting", "order_status", "booking",
                               "faq", "smalltalk")
        assert d["synthetic"] is True
        turns = d["turns"]
        assert len(turns) >= 2
        assert turns[0]["speaker"] == "user"
        assert turns[-1]["speaker"] == "assistant"
        for prev, cur in zip(turns, turns[1:]):
            assert prev["speaker"] != cur["speaker"]
        for turn in turns:
            assert turn["text"].strip()
            # All template slots must be filled.
            assert "{" not in turn["text"] and "}" not in turn["text"]


def test_dialogue_turn_bounds():
    """min/max turn bounds are respected (after even-turn alignment)."""
    dialogues = generate_dialogue(count=30, seed=3, min_turns=2, max_turns=4)
    for d in dialogues:
        assert len(d["turns"]) in (2, 4)


def test_dialogue_seeded_reproducibility():
    """The same seed reproduces the same dialogues."""
    assert generate_dialogue(count=10, seed=77) == generate_dialogue(count=10,
                                                                     seed=77)


def test_dialogue_invalid_bounds():
    """Bad turn bounds raise ValueError."""
    with pytest.raises(ValueError):
        generate_dialogue(count=1, min_turns=1)
    with pytest.raises(ValueError):
        generate_dialogue(count=1, min_turns=4, max_turns=2)


def test_dialogue_invalid_count():
    """Counts outside 1..10000 raise ValueError."""
    with pytest.raises(ValueError):
        generate_dialogue(count=-3)


def test_dialogue_jsonl_export():
    """Dialogue JSONL export parses line by line."""
    text = sd.dialogue_to_jsonl(generate_dialogue(count=3, seed=8))
    lines = [line for line in text.splitlines() if line]
    assert len(lines) == 3
    assert json.loads(lines[1])["id"] == 2


# ---------------------------------------------------------------------------
# Face image generator tests
# ---------------------------------------------------------------------------

def test_face_image_size_and_mode():
    """Faces render as RGB images at the requested size."""
    img = generate_face_image(seed=11, size=96)
    assert img.size == (96, 96)
    assert img.mode == "RGB"


def test_face_image_seeded_reproducibility():
    """Same seed yields identical pixels."""
    a = np.asarray(generate_face_image(seed=12, size=64))
    b = np.asarray(generate_face_image(seed=12, size=64))
    assert np.array_equal(a, b)


def test_face_image_varies_across_seeds():
    """Different seeds produce visibly different images."""
    a = np.asarray(generate_face_image(seed=13, size=64))
    b = np.asarray(generate_face_image(seed=14, size=64))
    assert not np.array_equal(a, b)


def test_face_image_invalid_size():
    """Sizes outside 32..1024 raise ValueError."""
    with pytest.raises(ValueError):
        generate_face_image(size=16)
    with pytest.raises(ValueError):
        generate_face_image(size=2048)


def test_render_face_png_bytes():
    """PNG bytes decode back to a valid image."""
    png = render_face_png(seed=15, size=48)
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG"
    assert img.size == (48, 48)


# ---------------------------------------------------------------------------
# Voice clip generator tests
# ---------------------------------------------------------------------------


def test_voice_clip_shape_and_dtype():
    """Waveform length matches duration * sample_rate, stored as int16."""
    clip = generate_voice_clip(seed=16, duration_seconds=0.8,
                               sample_rate=16000)
    assert clip["waveform"].dtype == np.int16
    assert clip["waveform"].shape == (12800,)
    assert clip["sample_rate"] == 16000
    assert clip["duration_seconds"] == 0.8


def test_voice_clip_not_silent():
    """The clip carries real signal, not just zeros."""
    clip = generate_voice_clip(seed=17, duration_seconds=1.0)
    assert int(np.max(np.abs(clip["waveform"]))) > 1000


def test_voice_clip_seeded_reproducibility():
    """Same seed yields identical samples."""
    a = generate_voice_clip(seed=18, duration_seconds=0.5)
    b = generate_voice_clip(seed=18, duration_seconds=0.5)
    assert np.array_equal(a["waveform"], b["waveform"])


def test_voice_clip_invalid_params():
    """Bad sample rates and durations raise ValueError."""
    with pytest.raises(ValueError):
        generate_voice_clip(sample_rate=12345)
    with pytest.raises(ValueError):
        generate_voice_clip(duration_seconds=0.01)
    with pytest.raises(ValueError):
        generate_voice_clip(duration_seconds=11.0)


def test_render_voice_wav_bytes():
    """WAV bytes parse back with matching rate and frame count."""
    wav_bytes = render_voice_wav(seed=19, duration_seconds=0.5,
                                 sample_rate=22050)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 22050
        assert wav.getnframes() == 11025


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------

def test_manifest_fields():
    """Manifests record kind, count, seed, version, and extras."""
    m = dataset_manifest("profiles", 42, 99, extra={"locale": "en"})
    assert m["kind"] == "profiles"
    assert m["count"] == 42
    assert m["seed"] == 99
    assert m["generator_version"] == GENERATOR_VERSION
    assert m["synthetic"] is True
    assert m["locale"] == "en"


def test_manifest_none_seed_is_explicit():
    """A None seed stays None so nondeterminism is visible in exports."""
    m = dataset_manifest("dialogue", 3, None)
    assert m["seed"] is None


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_client():
    """Return a cached TestClient bound to the FastAPI app."""
    from app import app  # pylint: disable=import-outside-toplevel
    return TestClient(app)


def test_api_synthetic_profiles_json():
    """POST /synthetic/profiles returns manifest plus records."""
    c = get_client()
    resp = c.post("/synthetic/profiles", json={"count": 4, "seed": 21})
    assert resp.status_code == 200
    body = resp.json()
    assert body["manifest"]["count"] == 4
    assert body["manifest"]["seed"] == 21
    assert len(body["profiles"]) == 4
    assert body["profiles"][0]["synthetic"] is True


def test_api_synthetic_profiles_csv():
    """csv format streams CSV text with a header row."""
    c = get_client()
    resp = c.post("/synthetic/profiles",
                  json={"count": 2, "format": "csv"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    rows = list(csv.reader(io.StringIO(resp.text)))
    assert len(rows) == 3  # header + 2


def test_api_synthetic_profiles_jsonl():
    """jsonl format streams newline-delimited JSON."""
    c = get_client()
    resp = c.post("/synthetic/profiles",
                  json={"count": 2, "format": "jsonl"})
    assert resp.status_code == 200
    lines = [line for line in resp.text.splitlines() if line]
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == 1


def test_api_synthetic_profiles_invalid_format_400():
    """Unknown export formats return 400."""
    c = get_client()
    resp = c.post("/synthetic/profiles", json={"count": 1, "format": "xml"})
    assert resp.status_code == 400


def test_api_synthetic_profiles_invalid_count_400():
    """Counts below 1 return 400."""
    c = get_client()
    resp = c.post("/synthetic/profiles", json={"count": 0})
    assert resp.status_code == 400


def test_api_synthetic_profiles_bad_locale_400():
    """Unsupported locales return 400."""
    c = get_client()
    resp = c.post("/synthetic/profiles", json={"count": 1, "locale": "zz"})
    assert resp.status_code == 400


def test_api_synthetic_dialogue_json():
    """POST /synthetic/dialogue returns manifest plus conversations."""
    c = get_client()
    resp = c.post("/synthetic/dialogue",
                  json={"count": 3, "seed": 22, "min_turns": 2,
                        "max_turns": 4})
    assert resp.status_code == 200
    body = resp.json()
    assert body["manifest"]["kind"] == "dialogue"
    assert len(body["dialogues"]) == 3
    assert body["dialogues"][0]["turns"][0]["speaker"] == "user"


def test_api_synthetic_dialogue_jsonl():
    """jsonl format streams newline-delimited dialogue JSON."""
    c = get_client()
    resp = c.post("/synthetic/dialogue",
                  json={"count": 2, "format": "jsonl"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    assert len([line for line in resp.text.splitlines() if line]) == 2


def test_api_synthetic_dialogue_bad_bounds_400():
    """min_turns < 2 returns 400."""
    c = get_client()
    resp = c.post("/synthetic/dialogue",
                  json={"count": 1, "min_turns": 1, "max_turns": 4})
    assert resp.status_code == 400


def test_api_synthetic_face_png():
    """GET /synthetic/face serves a valid PNG of the requested size."""
    c = get_client()
    resp = c.get("/synthetic/face", params={"seed": 23, "size": 64})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(resp.content))
    assert img.size == (64, 64)


def test_api_synthetic_face_bad_size_400():
    """Out-of-range sizes return 400."""
    c = get_client()
    resp = c.get("/synthetic/face", params={"size": 10})
    assert resp.status_code == 400


def test_api_synthetic_voice_wav():
    """GET /synthetic/voice serves a valid WAV at the requested rate."""
    c = get_client()
    resp = c.get("/synthetic/voice",
                 params={"seed": 24, "duration_seconds": 0.4,
                         "sample_rate": 16000})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(resp.content), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 6400


def test_api_synthetic_voice_bad_rate_400():
    """Unsupported sample rates return 400."""
    c = get_client()
    resp = c.get("/synthetic/voice", params={"sample_rate": 1234})
    assert resp.status_code == 400


def test_api_synthetic_voice_bad_duration_400():
    """Durations outside 0.1..10.0 return 400."""
    c = get_client()
    resp = c.get("/synthetic/voice", params={"duration_seconds": 30})
    assert resp.status_code == 400


def test_api_synthetic_capabilities():
    """GET /synthetic/capabilities describes all four generators."""
    c = get_client()
    resp = c.get("/synthetic/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["generator_version"] == GENERATOR_VERSION
    assert set(body["generators"].keys()) == {
        "profiles", "dialogue", "faces", "voice"
    }
