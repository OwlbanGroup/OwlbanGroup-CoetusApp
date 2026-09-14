"""
FastAPI application: NVIDIA Blackwell AI classifier plus the Coetus
payroll REST API (employees, payslips, runs, liabilities, exports).
"""

import io
import logging
import os
from decimal import Decimal
from typing import Annotated, Optional

import torch
import auth
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from dashboard import get_dashboard_html
from data_utils import load_image, preprocess_image, get_image_classes
from model import load_model, predict
from payroll import (
    Employee,
    add_employee,
    get_employee,
    update_employee,
    list_employees,
    delete_employee,
    generate_payslip,
    record_payslip,
    get_pay_history,
    get_payslip_record,
    delete_payslip_record,
    get_liabilities_report,
    get_ytd_summary,
    get_tax_config,
    export_history_csv,
    run_payroll,
    BRACKET_PRESETS,
)
from payslip_pdf import render_payslip_pdf
from synthetic_data import (
    MAX_DIALOGUE_COUNT,
    MAX_PROFILE_COUNT,
    SUPPORTED_SAMPLE_RATES,
    GENERATOR_VERSION,
    dataset_manifest,
    dialogue_to_jsonl,
    generate_dialogue,
    generate_profiles,
    profiles_to_csv,
    profiles_to_jsonl,
    render_face_png,
    render_voice_wav,
)

logger = logging.getLogger("coetus.app")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())

app = FastAPI(
    title="NVIDIA Blackwell AI Classifier",
    description="End-to-end AI system using NVIDIA Blackwell GPUs",
)

# Cross-origin browser clients (internal tools, hosted dashboards) need
# CORS. Set CORS_ORIGINS to a comma-separated allowlist, e.g.
#   CORS_ORIGINS="https://hr.example.com,https://ops.example.com"
# When unset, cross-origin browser calls are disallowed entirely.
_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",")
            if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# API authentication (env-gated, off by default)
# ---------------------------------------------------------------------------
# Set API_AUTH_TOKENS (comma-separated) to require an "X-API-Key" header or
# an "Authorization: Bearer <token>" on the protected route prefixes
# (default /payroll; override with AUTH_PROTECTED_PREFIXES). Unset means
# the API is fully open — local development and CI are unaffected.

@app.middleware("http")
async def api_auth_middleware(request: Request, call_next):
    """
    Credential gate for protected API prefixes.

    No-ops unless API_AUTH_TOKENS is configured. OPTIONS requests (CORS
    preflights) always pass so browser clients can negotiate access.
    """
    if request.method != "OPTIONS" and auth.configured_tokens():
        if auth.is_protected_path(request.url.path):
            denial = auth.check_credential(request.headers)
            if denial is not None:
                status_code, message = denial
                headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 \
                    else None
                return JSONResponse(content={
                    "error": message},
                    status_code=status_code, headers=headers)
    return await call_next(request)


def _custom_openapi():
    """Augment the schema with security schemes while auth is enabled."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, description=app.description,
                         version=app.version, routes=app.routes)
    if auth.configured_tokens():
        schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {"type": "apiKey", "in": "header",
                           "name": "X-API-Key"},
            "BearerAuth": {"type": "http", "scheme": "bearer"},
        }
        schema["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi  # type: ignore[method-assign]


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Last-resort handler for anything not caught by an endpoint: log the
    full traceback server-side and return a generic error so internals
    never leak to live clients.
    """
    logger.exception("Unhandled error on %s %s",
                     request.method, request.url.path)
    return JSONResponse(content={"error": "Internal server error"},
                        status_code=500)


# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

# Load model and classes
model = load_model(device)
class_names = get_image_classes()

# Reject oversized uploads before they consume memory / model time.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@app.post("/classify")
async def classify_image(file: Annotated[UploadFile, File()]):
    """
    Classify an uploaded image using the AI model.
    """
    try:
        # Read the uploaded file, enforcing the size limit first
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            return JSONResponse(content={
                "error": "Image exceeds the 10 MB upload limit"
            }, status_code=413)
        image = load_image(io.BytesIO(contents))

        # Preprocess the image
        input_batch = preprocess_image(image, device)

        # Perform prediction
        predicted_class = predict(model, input_batch, class_names)

        return JSONResponse(content={"predicted_class": predicted_class},
                            status_code=200)
    # Any failure is surfaced to the client as a 400 error payload;
    # unexpected ones are logged server-side for diagnosis.
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Classification failed: %s", e)
        return JSONResponse(content={"error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# Payroll API
# ---------------------------------------------------------------------------

class EmployeeCreate(BaseModel):
    """Request body for POST /payroll/employees."""
    name: str
    pay_type: str  # "salaried" or "hourly"
    annual_salary: Optional[float] = None   # required when salaried
    hourly_rate: Optional[float] = None     # required when hourly
    benefits_deduction_per_period: float = 0.0
    pay_periods_per_year: int = 26
    # Federal tax table preset: "single", "married_joint",
    # "head_of_household", or null to use the module's active table.
    filing_status: Optional[str] = "single"
    # State income tax preset key (see payroll.STATE_TAX_RATES), or null.
    state: Optional[str] = None
    # Optional company tag for multi-company setups.
    company_id: Optional[str] = None
    # Pre-tax 401(k) contribution as a percent of gross pay (0-100).
    retirement_401k_percent: float = 0.0


class EmployeeUpdate(BaseModel):
    """Partial-update body for PUT /payroll/employees/{id}."""
    model_config = {"extra": "forbid"}  # reject unknown/immutable fields outright

    name: Optional[str] = None
    annual_salary: Optional[float] = None
    hourly_rate: Optional[float] = None
    benefits_deduction_per_period: Optional[float] = None
    pay_periods_per_year: Optional[int] = None
    filing_status: Optional[str] = None
    state: Optional[str] = None
    company_id: Optional[str] = None
    retirement_401k_percent: Optional[float] = None


class PayslipRequest(BaseModel):
    """Body for the per-employee payslip endpoint."""
    hours_worked: Optional[float] = None  # required for hourly employees
    pay_period_index: int = 0             # period label stored with the payslip


class PayrollRunRequest(BaseModel):
    """Body for the batch payroll-run endpoint."""
    # Optional mapping of employee_id -> hours worked for this period.
    # Required entries for hourly employees; salaried employees ignore it.
    hours: Optional[dict] = None
    # Optional tax table override for this run:
    # "single", "married_joint", or "head_of_household".
    filing_status: Optional[str] = None
    # Optionally restrict the run to one company's employees.
    company_id: Optional[str] = None
    # Period label stored with every payslip recorded by this run.
    pay_period_index: int = 0


def _employee_payload(emp):
    """Serialize an Employee into the JSON shape used by the API."""
    return {
        "id": emp.id,
        "name": emp.name,
        "pay_type": emp.pay_type,
        "annual_salary": (float(emp.annual_salary)
                          if emp.annual_salary is not None else None),
        "hourly_rate": (float(emp.hourly_rate)
                        if emp.hourly_rate is not None else None),
        "filing_status": emp.filing_status,
        "state": emp.state,
        "company_id": emp.company_id,
        "retirement_401k_percent": float(emp.retirement_401k_percent),
    }


@app.post("/payroll/employees")
async def create_employee(payload: EmployeeCreate):
    """Register a new employee in the payroll system."""
    try:
        employee = Employee(
            name=payload.name,
            pay_type=payload.pay_type,
            annual_salary=(Decimal(str(payload.annual_salary))
                           if payload.annual_salary is not None else None),
            hourly_rate=(Decimal(str(payload.hourly_rate))
                         if payload.hourly_rate is not None else None),
            benefits_deduction_per_period=Decimal(
                str(payload.benefits_deduction_per_period)),
            pay_periods_per_year=payload.pay_periods_per_year,
            filing_status=payload.filing_status,
            state=payload.state,
            company_id=payload.company_id,
            retirement_401k_percent=Decimal(
                str(payload.retirement_401k_percent)),
        )
        stored = add_employee(employee)
        return JSONResponse(content=_employee_payload(stored), status_code=200)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/payroll/employees")
async def get_all_employees(company_id: Optional[str] = None):
    """List employees, optionally filtered by company_id."""
    return JSONResponse(content=[
        _employee_payload(e)
        for e in list_employees(company_id=company_id)
    ], status_code=200)


@app.get("/payroll/employees/{employee_id}")
async def read_employee(employee_id: int):
    """Fetch a single employee by ID."""
    employee = get_employee(employee_id)
    if employee is None:
        return JSONResponse(
            content={"error": f"Employee {employee_id} not found"},
            status_code=404)
    return JSONResponse(content=_employee_payload(employee), status_code=200)


@app.delete("/payroll/employees/{employee_id}")
async def remove_employee(employee_id: int):
    """Remove an employee from the payroll system."""
    if delete_employee(employee_id):
        return JSONResponse(content={"deleted": employee_id}, status_code=200)
    return JSONResponse(
        content={"error": f"Employee {employee_id} not found"}, status_code=404)


@app.post("/payroll/employees/{employee_id}/payslip")
async def create_payslip(employee_id: int, payload: PayslipRequest):
    """Generate a payslip for one pay period (recorded to pay history)."""
    employee = get_employee(employee_id)
    if employee is None:
        return JSONResponse(
            content={"error": f"Employee {employee_id} not found"},
            status_code=404)
    try:
        hours = (Decimal(str(payload.hours_worked))
                 if payload.hours_worked is not None else None)
        slip = generate_payslip(employee, hours_worked=hours,
                                pay_period_index=payload.pay_period_index)
        slip["history_id"] = record_payslip(
            slip, pay_period_index=payload.pay_period_index)
        return JSONResponse(content=slip, status_code=200)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/payroll/employees/{employee_id}/payslips")
async def read_pay_history(employee_id: int):
    """Return the recorded payslip history for an employee, newest first."""
    if get_employee(employee_id) is None:
        return JSONResponse(
            content={"error": f"Employee {employee_id} not found"},
            status_code=404)
    return JSONResponse(content=get_pay_history(employee_id), status_code=200)


@app.put("/payroll/employees/{employee_id}")
async def modify_employee(employee_id: int, payload: EmployeeUpdate):
    """
    Partially update an employee's record. Only provided fields are changed;
    `pay_type` is immutable. Returns 404 for unknown IDs and 400 for
    invalid field names or values.
    """
    # Only apply fields the client explicitly sent; omitted fields stay
    # unchanged, and an explicit null is allowed to clear optional values.
    provided = payload.model_fields_set
    updates = {k: v for k, v in payload.model_dump().items() if k in provided}
    try:
        updated = update_employee(employee_id, updates)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    if updated is None:
        return JSONResponse(
            content={"error": f"Employee {employee_id} not found"},
            status_code=404)
    body = _employee_payload(updated)
    body["benefits_deduction_per_period"] = float(
        updated.benefits_deduction_per_period)
    body["pay_periods_per_year"] = updated.pay_periods_per_year
    return JSONResponse(content=body, status_code=200)


@app.get("/payroll/liabilities")
async def read_liabilities(company_id: Optional[str] = None):
    """
    Aggregate employer payroll tax liabilities (FICA match + FUTA/SUTA)
    across all recorded payslips, optionally filtered by company_id.
    """
    return JSONResponse(
        content=get_liabilities_report(company_id=company_id), status_code=200)


@app.get("/payroll/employees/{employee_id}/ytd")
async def read_ytd_summary(employee_id: int):
    """
    Year-to-date (all recorded periods) summary for one employee: wages,
    401(k), income-tax wages, every withholding, deductions, and net pay.
    """
    if get_employee(employee_id) is None:
        return JSONResponse(
            content={"error": f"Employee {employee_id} not found"},
            status_code=404)
    return JSONResponse(content=get_ytd_summary(employee_id), status_code=200)


@app.get("/payroll/export.csv")
async def export_payroll_csv(company_id: Optional[str] = None,
                             employee_id: Optional[int] = None):
    """
    Export recorded payslips as a CSV payroll journal, optionally scoped
    to a company and/or a single employee.
    """
    csv_text = export_history_csv(company_id=company_id, employee_id=employee_id)
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=payroll_journal.csv"},
    )


@app.get("/payroll/employees/{employee_id}/payslips/{history_id}/pdf")
async def download_payslip_pdf(employee_id: int, history_id: int):
    """Download one recorded payslip as a PDF document."""
    record = get_payslip_record(history_id)
    if record is None or record["employee_id"] != employee_id:
        return JSONResponse(content={
            "error": f"Payslip {history_id} not found for employee {employee_id}"
        }, status_code=404)
    pdf_bytes = render_payslip_pdf(record)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                f"attachment; filename=payslip_{employee_id}_{history_id}.pdf"
        },
    )


@app.post("/payroll/run")
async def run_payroll_for_period(payload: PayrollRunRequest):
    """
    Run payroll for all employees for one pay period. Generates a payslip
    per employee plus period totals. Hourly employees require an entry in
    the `hours` mapping; missing entries are reported in `errors`.
    """
    if (payload.filing_status is not None
            and payload.filing_status not in BRACKET_PRESETS):
        return JSONResponse(content={
            "error": f"filing_status must be one of {sorted(BRACKET_PRESETS)}"
        }, status_code=400)

    hours_map = None
    if payload.hours:
        try:
            hours_map = {int(k): Decimal(str(v)) for k, v in payload.hours.items()}
            for employee_id, hrs in hours_map.items():
                if get_employee(employee_id) is None:
                    return JSONResponse(content={
                        "error": f"Employee {employee_id} not found"
                    }, status_code=404)
                if hrs < 0:
                    return JSONResponse(content={
                        "error": f"hours for employee {employee_id} must be >= 0"
                    }, status_code=400)
        except ValueError as e:
            return JSONResponse(content={"error": str(e)}, status_code=400)

    brackets = BRACKET_PRESETS[payload.filing_status] if payload.filing_status else None
    result = run_payroll(hours_by_employee=hours_map, tax_brackets=brackets,
                         pay_period_index=payload.pay_period_index,
                         company_id=payload.company_id)
    return JSONResponse(content=result, status_code=200)


@app.get("/payroll/employees/{employee_id}/payslips/{history_id}")
async def read_payslip_record(employee_id: int, history_id: int):
    """Fetch one recorded payslip as JSON by its history row id."""
    record = get_payslip_record(history_id)
    if record is None or record["employee_id"] != employee_id:
        return JSONResponse(content={
            "error": f"Payslip {history_id} not found for employee {employee_id}"
        }, status_code=404)
    return JSONResponse(content=record, status_code=200)


@app.delete("/payroll/employees/{employee_id}/payslips/{history_id}")
async def remove_payslip_record(employee_id: int, history_id: int):
    """
    Void (delete) one recorded payslip. Returns 404 when the payslip does
    not exist or belongs to a different employee. Liabilities, YTD
    summaries, and CSV exports reflect the removal immediately.
    """
    deleted = delete_payslip_record(history_id, employee_id=employee_id)
    if not deleted:
        return JSONResponse(content={
            "error": f"Payslip {history_id} not found for employee {employee_id}"
        }, status_code=404)
    return JSONResponse(content={"deleted": history_id,
                                 "employee_id": employee_id}, status_code=200)


@app.get("/payroll/rates")
async def read_tax_config():
    """
    Current payroll tax configuration: FICA rates and wage bases, the
    employer SUTA rate, overtime rules, available filing statuses and
    state presets, and all bracket tables.
    """
    return JSONResponse(content=get_tax_config(), status_code=200)


# ---------------------------------------------------------------------------
# Synthetic human training-data endpoints
# ---------------------------------------------------------------------------


class SyntheticProfilesRequest(BaseModel):
    """Body for POST /synthetic/profiles."""
    count: int = 10
    seed: Optional[int] = None
    locale: str = "en"
    format: str = "json"  # json | csv | jsonl


class SyntheticDialogueRequest(BaseModel):
    """Body for POST /synthetic/dialogue."""
    count: int = 10
    seed: Optional[int] = None
    min_turns: int = 2
    max_turns: int = 6
    format: str = "json"  # json | jsonl


@app.post("/synthetic/profiles")
async def create_synthetic_profiles(payload: SyntheticProfilesRequest):
    """
    Generate synthetic human profile records (fabricated names, contact
    details, occupations) for tabular-model training and schema testing.

    ``format`` selects the response body: ``json`` (default) returns a
    dataset manifest plus the records; ``csv`` and ``jsonl`` stream the
    records as downloadable text. All records are marked ``synthetic``.
    """
    if payload.format not in ("json", "csv", "jsonl"):
        return JSONResponse(content={
            "error": "format must be one of: json, csv, jsonl"
        }, status_code=400)
    if payload.count < 1 or payload.count > MAX_PROFILE_COUNT:
        return JSONResponse(content={
            "error": f"count must be between 1 and {MAX_PROFILE_COUNT}"
        }, status_code=400)

    try:
        profiles = generate_profiles(count=payload.count, seed=payload.seed,
                                     locale=payload.locale)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    manifest = dataset_manifest("profiles", payload.count, payload.seed,
                                extra={"locale": payload.locale})
    if payload.format == "csv":
        return Response(content=profiles_to_csv(profiles),
                        media_type="text/csv",
                        headers={"Content-Disposition":
                                 'attachment; filename="synthetic_profiles.csv"'})
    if payload.format == "jsonl":
        return Response(content=profiles_to_jsonl(profiles),
                        media_type="application/x-ndjson",
                        headers={"Content-Disposition":
                                 'attachment; filename="synthetic_profiles.jsonl"'})
    return JSONResponse(content={"manifest": manifest, "profiles": profiles},
                        status_code=200)


@app.post("/synthetic/dialogue")
async def create_synthetic_dialogue(payload: SyntheticDialogueRequest):
    """
    Generate synthetic user/assistant dialogue samples with intent
    labels for NLP fine-tuning pipelines. ``format`` selects ``json``
    (manifest + samples) or ``jsonl`` (streamed JSON Lines).
    """
    if payload.format not in ("json", "jsonl"):
        return JSONResponse(content={
            "error": "format must be one of: json, jsonl"
        }, status_code=400)
    if payload.count < 1 or payload.count > MAX_DIALOGUE_COUNT:
        return JSONResponse(content={
            "error": f"count must be between 1 and {MAX_DIALOGUE_COUNT}"
        }, status_code=400)

    try:
        dialogues = generate_dialogue(
            count=payload.count, seed=payload.seed,
            min_turns=payload.min_turns, max_turns=payload.max_turns)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    manifest = dataset_manifest("dialogue", payload.count, payload.seed,
                                extra={"min_turns": payload.min_turns,
                                       "max_turns": payload.max_turns})
    if payload.format == "jsonl":
        return Response(content=dialogue_to_jsonl(dialogues),
                        media_type="application/x-ndjson",
                        headers={"Content-Disposition":
                                 'attachment; filename="synthetic_dialogue.jsonl"'})
    return JSONResponse(content={"manifest": manifest, "dialogues": dialogues},
                        status_code=200)


@app.get("/synthetic/face")
async def get_synthetic_face(seed: Optional[int] = None, size: int = 128):
    """
    Download a procedurally rendered synthetic face image as PNG.

    Structured placeholder imagery for vision-pipeline smoke tests —
    the same ``seed`` always yields the same image.
    """
    if size < 32 or size > 1024:
        return JSONResponse(content={
            "error": "size must be between 32 and 1024"
        }, status_code=400)
    png = render_face_png(seed=seed, size=size)
    return Response(content=png, media_type="image/png",
                    headers={"Content-Disposition":
                             'attachment; filename="synthetic_face.png"'})


@app.get("/synthetic/voice")
async def get_synthetic_voice(seed: Optional[int] = None,
                              duration_seconds: float = 1.5,
                              sample_rate: int = 16000):
    """
    Download a synthesized speech-like audio clip as WAV (16-bit PCM
    mono). Formant-style placeholder audio for audio-pipeline smoke
    tests — not intelligible speech.
    """
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        return JSONResponse(content={
            "error": f"sample_rate must be one of {list(SUPPORTED_SAMPLE_RATES)}"
        }, status_code=400)
    if duration_seconds < 0.1 or duration_seconds > 10.0:
        return JSONResponse(content={
            "error": "duration_seconds must be between 0.1 and 10.0"
        }, status_code=400)
    try:
        wav = render_voice_wav(seed=seed, duration_seconds=duration_seconds,
                               sample_rate=sample_rate)
    except (TypeError, ValueError) as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    return Response(content=wav, media_type="audio/wav",
                    headers={"Content-Disposition":
                             'attachment; filename="synthetic_voice.wav"'})


@app.get("/synthetic/capabilities")
async def get_synthetic_capabilities():
    """Describe the synthetic data generators and their parameter ranges."""
    return JSONResponse(content={
        "generator_version": GENERATOR_VERSION,
        "generators": {
            "profiles": {
                "endpoint": "POST /synthetic/profiles",
                "formats": ["json", "csv", "jsonl"],
                "max_count": MAX_PROFILE_COUNT,
            },
            "dialogue": {
                "endpoint": "POST /synthetic/dialogue",
                "formats": ["json", "jsonl"],
                "max_count": MAX_DIALOGUE_COUNT,
            },
            "faces": {
                "endpoint": "GET /synthetic/face",
                "format": "png",
                "size_range": [32, 1024],
            },
            "voice": {
                "endpoint": "GET /synthetic/voice",
                "format": "wav",
                "sample_rates": list(SUPPORTED_SAMPLE_RATES),
                "duration_range_seconds": [0.1, 10.0],
            },
        },
        "note": "All generated records are synthetic; none contain real "
                "personal data. Seeds make every dataset reproducible.",
    }, status_code=200)


@app.get("/health", include_in_schema=False)
async def health():
    """Liveness probe for container orchestrators and load balancers."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Service banner with links to the dashboard and API docs."""
    return {
        "message": "NVIDIA Blackwell AI Classifier API",
        "dashboard": "/dashboard",
        "payroll_docs": "/docs",
    }


@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    """Mobile-friendly payroll console."""
    return Response(content=get_dashboard_html(), media_type="text/html")


if __name__ == "__main__":
    import uvicorn

    # Allow the bind host to be overridden via the HOST environment variable.
    # Defaults to 127.0.0.1 for local-development safety; set HOST=0.0.0.0
    # for container deployments (see Dockerfile ENV directive).
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=8000)
