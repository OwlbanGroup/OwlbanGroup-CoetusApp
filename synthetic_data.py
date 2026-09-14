"""
Synthetic human training-data generation for the Coetus AI system.

Produces four kinds of fully synthetic, seeded (reproducible) datasets
used to bootstrap and smoke-test ML pipelines without touching real
personal data:

- **Profiles**  — synthetic human tabular records (name, age, contact
  details, occupation) for tabular-model training and schema testing.
- **Dialogue**  — template-based conversational samples (user/assistant
  turns with intent labels) for NLP fine-tuning pipelines.
- **Faces**     — procedurally rendered face-like images for vision
  pipeline smoke tests (placeholder data, not photorealistic).
- **Voice**     — formant-style synthesized speech-like audio clips for
  audio pipeline smoke tests (placeholder data, not intelligible speech).

Everything is driven by :class:`numpy.random.default_rng`, so a given
seed always reproduces the same dataset. No component here uses real
personally identifiable information: all names, emails, phone numbers,
and addresses are fabricated from fixed word pools.
"""

import io
import wave
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw

# Bumped whenever generator behaviour changes, so dataset manifests can
# record exactly which revision produced a given dataset.
GENERATOR_VERSION = "1.0.0"

SUPPORTED_LOCALES = ("en",)

MAX_PROFILE_COUNT = 10000
MAX_DIALOGUE_COUNT = 10000

FIRST_NAMES_MALE = (
    "James", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Chris", "Daniel", "Matthew", "Anthony", "Mark", "Steven",
    "Andrew", "Kevin", "Brian", "George", "Edward", "Ronald", "Timothy",
    "Jason", "Jeffrey", "Ryan", "Jacob", "Nicholas", "Eric", "Jonathan",
    "Stephen", "Larry", "Justin", "Scott", "Brandon", "Benjamin", "Samuel",
    "Gregory", "Frank", "Alexander", "Raymond", "Patrick", "Jack", "Dennis",
)

FIRST_NAMES_FEMALE = (
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara",
    "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
    "Margaret", "Sandra", "Ashley", "Kimberly", "Emily", "Donna",
    "Michelle", "Carol", "Amanda", "Dorothy", "Melissa", "Deborah",
    "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia", "Kathleen",
    "Amy", "Angela", "Shirley", "Anna", "Brenda", "Pamela", "Emma",
    "Nicole", "Helen", "Samantha", "Katherine",
)

LAST_NAMES = (
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz",
    "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris",
)

OCCUPATIONS = (
    "Software Engineer", "Registered Nurse", "Teacher", "Accountant",
    "Electrician", "Marketing Manager", "Truck Driver", "Chef",
    "Data Analyst", "Physical Therapist", "Sales Representative",
    "Mechanical Engineer", "Graphic Designer", "Pharmacist", "Plumber",
    "Human Resources Specialist", "Financial Advisor", "Dental Hygienist",
    "Logistics Coordinator", "Customer Service Manager", "Architect",
    "Paralegal", "Radiologic Technologist", "Carpenter", "Insurance Agent",
    "Lab Technician", "Project Manager", "Real Estate Agent",
    "Occupational Therapist", "Network Administrator",
)

# (city, state) pairs — fabricated residents, real place names only as
# coarse geographic labels for stratified sampling.
CITIES = (
    ("Seattle", "WA"), ("Austin", "TX"), ("Denver", "CO"), ("Chicago", "IL"),
    ("Boston", "MA"), ("Atlanta", "GA"), ("Phoenix", "AZ"), ("Portland", "OR"),
    ("Raleigh", "NC"), ("Columbus", "OH"), ("Nashville", "TN"),
    ("Minneapolis", "MN"), ("San Diego", "CA"), ("Tampa", "FL"),
    ("Pittsburgh", "PA"), ("Salt Lake City", "UT"),
)

EMAIL_DOMAINS = (
    "example.com", "example.org", "example.net", "mail.example.com",
    "corp.example.net",
)

STREET_NAMES = (
    "Maple St", "Oak Ave", "Cedar Ln", "Birch Dr", "Elm Ct", "Pine Rd",
    "Willow Way", "Juniper Pl", "Aspen Blvd", "Chestnut St",
)


def _rng(seed):
    """Return a numpy Generator; ``seed=None`` means nondeterministic."""
    return np.random.default_rng(seed)


def _validate_count(count, upper):
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("count must be an integer")
    if count < 1 or count > upper:
        raise ValueError(f"count must be between 1 and {upper}")


# ---------------------------------------------------------------------------
# Synthetic human profiles
# ---------------------------------------------------------------------------


def generate_profiles(count=10, seed=None, locale="en"):
    """
    Generate ``count`` synthetic human profile records.

    Returns a list of dicts with fabricated names, ages, contact details,
    and occupations. Records carry ``"synthetic": True`` so downstream
    consumers can never confuse them with real personal data. A fixed
    ``seed`` reproduces the same dataset exactly.
    """
    if locale not in SUPPORTED_LOCALES:
        raise ValueError(
            f"unsupported locale '{locale}'; supported: {SUPPORTED_LOCALES}"
        )
    _validate_count(count, MAX_PROFILE_COUNT)
    rng = _rng(seed)

    used_emails = set()
    profiles = []
    for idx in range(count):
        gender = str(rng.choice(["male", "female"]))
        first_pool = FIRST_NAMES_MALE if gender == "male" else FIRST_NAMES_FEMALE
        first_name = str(rng.choice(first_pool))
        last_name = str(rng.choice(LAST_NAMES))
        city, state = CITIES[int(rng.integers(len(CITIES)))]
        age = int(rng.integers(18, 86))

        email = _unique_email(rng, first_name, last_name, used_emails)
        area = int(rng.integers(200, 999))
        phone = (f"{area:03d}-{int(rng.integers(200, 999)):03d}-"
                 f"{int(rng.integers(0, 10000)):04d}")
        street_no = int(rng.integers(10, 9900))

        profiles.append(OrderedDict([
            ("id", idx + 1),
            ("first_name", first_name),
            ("last_name", last_name),
            ("full_name", f"{first_name} {last_name}"),
            ("age", age),
            ("gender", gender),
            ("email", email),
            ("phone", phone),
            ("street",
             f"{street_no} {STREET_NAMES[int(rng.integers(len(STREET_NAMES)))]}"),
            ("city", city),
            ("state", state),
            ("occupation", str(rng.choice(OCCUPATIONS))),
            ("synthetic", True),
        ]))
    return profiles


def _unique_email(rng, first_name, last_name, used):
    """Build a fabricated, collision-free email from a name pool."""
    patterns = (
        f"{first_name.lower()}.{last_name.lower()}",
        f"{first_name.lower()}{last_name.lower()}",
        f"{first_name[0].lower()}{last_name.lower()}",
        f"{first_name.lower()}_{last_name.lower()}",
    )
    domain = EMAIL_DOMAINS[int(rng.integers(len(EMAIL_DOMAINS)))]
    local = patterns[int(rng.integers(len(patterns)))]
    email = f"{local}@{domain}"
    suffix = 2
    while email in used:
        email = f"{local}{suffix}@{domain}"
        suffix += 1
    used.add(email)
    return email


PROFILE_CSV_FIELDS = (
    "id", "first_name", "last_name", "full_name", "age", "gender",
    "email", "phone", "street", "city", "state", "occupation", "synthetic",
)


def profiles_to_csv(profiles):
    """Serialize profiles to RFC-4180 CSV text (with header)."""
    import csv

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(PROFILE_CSV_FIELDS),
                            extrasaction="ignore")
    writer.writeheader()
    for profile in profiles:
        row = dict(profile)
        row["synthetic"] = "true" if profile.get("synthetic") else "false"
        writer.writerow(row)
    return buf.getvalue()


def profiles_to_jsonl(profiles):
    """Serialize profiles to JSON Lines text (one JSON object per line)."""
    import json

    return "".join(json.dumps(p) + "\n" for p in profiles)


# ---------------------------------------------------------------------------
# Synthetic dialogue text
# ---------------------------------------------------------------------------

_DIALOGUE_INTENTS = ("greeting", "order_status", "booking", "faq", "smalltalk")

_USER_TEMPLATES = {
    "greeting": (
        "Hi there!", "Hello, good {time_of_day}!", "Hey, how are you?",
        "Good {time_of_day}, nice to meet you.",
    ),
    "order_status": (
        "Where is my order #{order_id}?",
        "Can you track package {order_id} for me?",
        "My order #{order_id} hasn't arrived yet.",
        "When will order #{order_id} ship?",
    ),
    "booking": (
        "I'd like to book a table for {party} people.",
        "Can I schedule an appointment for {party} people next week?",
        "Please reserve a room from {weekday} to Sunday.",
        "I need a {service} appointment this {weekday}.",
    ),
    "faq": (
        "What are your business hours?",
        "Do you offer {service}?",
        "How do I reset my password?",
        "Is there a fee for {service}?",
    ),
    "smalltalk": (
        "Nice weather today, isn't it?",
        "Did you watch the game last night?",
        "Any recommendations for lunch around here?",
        "How has your week been?",
    ),
}

_ASSISTANT_TEMPLATES = {
    "greeting": (
        "Hello! How can I help you today?",
        "Hi! What can I do for you?",
        "Good {time_of_day}! I'm happy to assist.",
    ),
    "order_status": (
        "Order #{order_id} is currently {order_state} and should arrive by {weekday}.",
        "Let me check... package {order_id} was {order_state} this morning.",
    ),
    "booking": (
        "Sure, I can book that for {party} people on {weekday}.",
        "A table for {party} on {weekday} — let me confirm availability.",
    ),
    "faq": (
        "Yes, we offer {service} during standard business hours.",
        "You can find details about {service} on our support page.",
    ),
    "smalltalk": (
        "It really is a lovely day!",
        "Ha, I'm just an assistant, but I appreciate it!",
    ),
}

_SERVICES = ("delivery", "installation", "returns", "tech support", "tutoring")
_ORDER_STATES = ("in transit", "out for delivery", "processed", "delayed")
_TIME_OF_DAY = ("morning", "afternoon", "evening")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")


def _fill(rng, template):
    """Fill a dialogue template's placeholder slots with random values."""
    return template.format(
        time_of_day=str(rng.choice(_TIME_OF_DAY)),
        order_id=int(rng.integers(10000, 999999)),
        party=int(rng.integers(2, 9)),
        service=str(rng.choice(_SERVICES)),
        order_state=str(rng.choice(_ORDER_STATES)),
        weekday=str(rng.choice(_WEEKDAYS)),
    )


def generate_dialogue(count=10, seed=None, min_turns=2, max_turns=6):
    """
    Generate ``count`` synthetic user/assistant conversations.

    Each sample is a dict with an ``intent`` label and a list of
    ``turns`` (alternating ``user``/``assistant`` utterances drawn from
    template pools with randomized slot fillers). Turn count is sampled
    in ``[min_turns, max_turns]``; conversations are kept to an even
    number of turns so they always end on an assistant reply.
    """
    _validate_count(count, MAX_DIALOGUE_COUNT)
    if not isinstance(min_turns, int) or not isinstance(max_turns, int) \
            or isinstance(min_turns, bool) or isinstance(max_turns, bool) \
            or min_turns < 2 or max_turns < min_turns:
        raise ValueError(
            "turn bounds must be integers with 2 <= min_turns <= max_turns"
        )
    rng = _rng(seed)

    dialogues = []
    for idx in range(count):
        # Sample an even number of turns in the requested range so each
        # conversation ends with an assistant response.
        target = int(rng.integers(min_turns, max_turns + 1))
        turns_requested = target if target % 2 == 0 else target + 1

        intent = str(rng.choice(_DIALOGUE_INTENTS))
        turns = []
        for turn_no in range(turns_requested):
            speaker = "user" if turn_no % 2 == 0 else "assistant"
            pools = _USER_TEMPLATES if speaker == "user" else _ASSISTANT_TEMPLATES
            template = str(rng.choice(pools[intent]))
            turns.append(OrderedDict([
                ("speaker", speaker),
                ("text", _fill(rng, template)),
            ]))
        dialogues.append(OrderedDict([
            ("id", idx + 1),
            ("intent", intent),
            ("turns", turns),
            ("synthetic", True),
        ]))
    return dialogues


def dialogue_to_jsonl(dialogues):
    """Serialize dialogue samples to JSON Lines text."""
    import json

    return "".join(json.dumps(d) + "\n" for d in dialogues)


# ---------------------------------------------------------------------------
# Synthetic face images
# ---------------------------------------------------------------------------


_SKIN_TONES = (
    (241, 194, 150), (224, 172, 132), (198, 134, 96),
    (166, 106, 70), (128, 78, 52), (92, 54, 36),
)
_HAIR_COLORS = (
    (24, 18, 12), (52, 34, 18), (86, 56, 28), (120, 84, 44),
    (170, 130, 70), (210, 190, 150), (90, 90, 95),
)
_EYE_COLORS = ((58, 40, 24), (40, 76, 40), (44, 88, 130), (20, 20, 20))
_BG_COLORS = ((228, 232, 238), (222, 230, 224), (236, 226, 214))


def _shade(color, delta):
    """Darken/lighten an RGB tuple by ``delta`` per channel."""
    return tuple(max(0, min(255, int(c) + delta)) for c in color)


def generate_face_image(seed=None, size=128):
    """
    Render a deterministic, procedurally drawn face-like image.

    Produces a PIL RGB ``Image`` of ``size`` x ``size`` pixels. This is
    structured placeholder data for pipeline smoke tests — geometry,
    palette, and placement all derive from the seed, so the same seed
    always yields the same image. It is explicitly *not* photorealistic
    and not a GAN sample.
    """
    if not isinstance(size, int) or isinstance(size, bool) \
            or size < 32 or size > 1024:
        raise ValueError("size must be an integer between 32 and 1024")
    rng = _rng(seed)

    skin = _SKIN_TONES[int(rng.integers(len(_SKIN_TONES)))]
    hair = _HAIR_COLORS[int(rng.integers(len(_HAIR_COLORS)))]
    eye = _EYE_COLORS[int(rng.integers(len(_EYE_COLORS)))]
    bg = _BG_COLORS[int(rng.integers(len(_BG_COLORS)))]

    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    cx = size / 2

    head_w = size * float(rng.uniform(0.52, 0.62))
    head_h = size * float(rng.uniform(0.66, 0.76))
    head_top = size * 0.12
    head_box = (cx - head_w / 2, head_top, cx + head_w / 2, head_top + head_h)

    # Hair behind the head, then the head, then fringe over the forehead.
    hair_pad = size * float(rng.uniform(0.02, 0.06))
    draw.ellipse((head_box[0] - hair_pad, head_box[1] - hair_pad,
                  head_box[2] + hair_pad, head_box[1] + head_h * 0.7), fill=hair)
    draw.ellipse(head_box, fill=skin)
    fringe_h = head_h * float(rng.uniform(0.22, 0.32))
    draw.ellipse((head_box[0], head_box[1], head_box[2],
                  head_box[1] + fringe_h), fill=hair)

    eye_y = head_top + head_h * float(rng.uniform(0.42, 0.50))
    eye_dx = head_w * float(rng.uniform(0.20, 0.26))
    eye_r = size * float(rng.uniform(0.035, 0.045))
    brow_w = max(1, int(size * 0.012))
    for sign in (-1, 1):
        ex, ey = cx + sign * eye_dx, eye_y
        draw.ellipse((ex - eye_r, ey - eye_r * 0.6, ex + eye_r, ey + eye_r * 0.6),
                     fill=(255, 255, 255))
        pupil_r = eye_r * float(rng.uniform(0.35, 0.5))
        draw.ellipse((ex - pupil_r, ey - pupil_r, ex + pupil_r, ey + pupil_r),
                     fill=eye)
        brow_y = ey - eye_r * float(rng.uniform(1.8, 2.6))
        draw.line((ex - eye_r * 1.4, brow_y, ex + eye_r * 1.4, brow_y),
                  fill=hair, width=brow_w)

    nose_y = eye_y + head_h * float(rng.uniform(0.12, 0.18))
    draw.line((cx, eye_y + eye_r * 0.8, cx, nose_y), fill=_shade(skin, -30),
              width=max(1, int(size * 0.01)))

    mouth_y = head_top + head_h * float(rng.uniform(0.78, 0.86))
    mouth_w = head_w * float(rng.uniform(0.22, 0.34))
    draw.arc((cx - mouth_w, mouth_y - size * 0.05,
              cx + mouth_w, mouth_y + size * 0.05),
             start=10, end=170, fill=_shade(skin, -60),
             width=max(1, int(size * 0.012)))
    return img


def render_face_png(seed=None, size=128):
    """Render a synthetic face and return the PNG bytes."""
    img = generate_face_image(seed=seed, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Synthetic voice clips
# ---------------------------------------------------------------------------


SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 44100)

# Rough vowel formant pairs (F1, F2) in Hz.
_FORMANTS = {
    "ah": (700, 1220), "eh": (530, 1840), "ee": (270, 2290),
    "oh": (570, 840), "oo": (300, 870),
}


def generate_voice_clip(seed=None, duration_seconds=1.5, sample_rate=16000):
    """
    Synthesize a short, speech-like audio clip.

    The clip is built from a sequence of "phoneme" segments: each
    segment is a harmonic-rich glottal source with a drifting fundamental
    frequency (prosody), spectrally shaped toward one of the vowel
    formant targets, with amplitude envelopes and short silences between
    segments. The result sounds voice-like and has realistic spectral
    structure for audio-pipeline smoke tests, but is not intelligible
    speech.

    Returns ``{"waveform": np.int16 array, "sample_rate": int,
    "duration_seconds": float}``.
    """
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) \
            or sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}")
    duration = float(duration_seconds)
    if duration < 0.1 or duration > 10.0:
        raise ValueError("duration_seconds must be between 0.1 and 10.0")
    rng = _rng(seed)

    total = int(round(duration * sample_rate))
    waveform = np.zeros(total, dtype=np.float64)

    # Plan segments until the duration is covered.
    pos = 0
    base_f0 = float(rng.uniform(95, 180))
    while pos < total:
        seg_len = min(int(rng.uniform(0.12, 0.35) * sample_rate), total - pos)
        if seg_len <= 0:
            break

        if rng.random() < 0.15:  # inter-segment pause
            pos += seg_len
            continue

        f1, f2 = _FORMANTS[str(rng.choice(list(_FORMANTS)))]
        waveform[pos:pos + seg_len] = _synth_segment(
            rng, seg_len, sample_rate, base_f0, f1, f2)
        pos += seg_len

    # Gentle global fade-in/out to avoid clicks at the boundaries.
    fade = min(int(0.01 * sample_rate), total // 4)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        waveform[:fade] *= ramp
        waveform[-fade:] *= ramp[::-1]

    peak = float(np.max(np.abs(waveform))) or 1.0
    pcm = np.clip(waveform / peak * 0.85, -1.0, 1.0)
    return {
        "waveform": np.asarray(pcm * 32767.0, dtype=np.int16),
        "sample_rate": sample_rate,
        "duration_seconds": total / sample_rate,
    }


def _synth_segment(rng, n, sample_rate, base_f0, f1, f2):
    """
    Render one voiced segment: harmonic stack with per-harmonic gains
    shaped by two formant resonance envelopes and a drifting f0.
    """
    t = np.arange(n, dtype=np.float64) / sample_rate
    # Prosody: f0 drifts smoothly around the speaker's base pitch.
    f0 = base_f0 * (1.0 + 0.12 * np.sin(2 * np.pi * rng.uniform(1.5, 4.0) * t)
                    + 0.05 * np.sin(2 * np.pi * rng.uniform(0.4, 1.0) * t))
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate

    out = np.zeros(n, dtype=np.float64)
    max_harmonic = int(sample_rate / (2 * base_f0))
    for h in range(1, max(2, min(max_harmonic, 60)) + 1):
        freq = h * base_f0
        # Formant gains: gaussian resonance bumps at F1 and F2, plus a
        # faint residual source spectrum so not every harmonic vanishes.
        gain = (
            np.exp(-((freq - f1) ** 2) / (2 * 110.0 ** 2))
            + 0.8 * np.exp(-((freq - f2) ** 2) / (2 * 160.0 ** 2))
            + 0.02 / (h ** 0.5)
        )
        if float(gain) < 1e-4:
            continue
        jitter = 1.0 + float(rng.uniform(-0.05, 0.05))
        out += gain * jitter * np.sin(h * phase + float(rng.uniform(0, 2 * np.pi)))

    # Syllabic amplitude envelope with a soft attack/decay.
    env = 0.55 + 0.45 * np.sin(np.pi * np.linspace(0.0, 1.0, n)) ** 2
    voiced = out * env
    breath = rng.normal(0.0, 0.01, n)  # slight aspiration noise
    return (voiced + breath) / max(1.0, float(np.max(np.abs(voiced))))


def render_voice_wav(seed=None, duration_seconds=1.5, sample_rate=16000):
    """Synthesize a voice clip and return WAV file bytes (16-bit PCM mono)."""
    clip = generate_voice_clip(seed=seed, duration_seconds=duration_seconds,
                               sample_rate=sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(clip["sample_rate"])
        wav.writeframes(clip["waveform"].tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Dataset manifests
# ---------------------------------------------------------------------------

def dataset_manifest(kind, count, seed, extra=None):
    """
    Build a reproducibility manifest describing a generated dataset.

    Records the generator version, dataset kind, sample count, and the
    effective seed (``None`` means nondeterministic), plus any
    generator-specific extras. Embed this alongside exported data.
    """
    manifest = OrderedDict([
        ("kind", kind),
        ("count", count),
        ("seed", seed),
        ("generator_version", GENERATOR_VERSION),
        ("synthetic", True),
    ])
    if extra:
        manifest.update(extra)
    return manifest
