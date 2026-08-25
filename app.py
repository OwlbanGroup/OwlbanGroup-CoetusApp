from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
import torch
from data_utils import load_image, preprocess_image, get_image_classes
from model import load_model
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
    get_liabilities_report,
    get_ytd_summary,
    export_history_csv,
    run_payroll,
    BRACKET_PRESETS,
)
from payslip_pdf import render_payslip_pdf
from dashboard import get_dashboard_html
from decimal import Decimal
from pydantic import BaseModel
from typing import Optional
import io

app = FastAPI(title="NVIDIA Blackwell AI Classifier", description="End-to-end AI system using NVIDIA Blackwell GPUs")

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and classes
model = load_model(device)
class_names = get_image_classes()

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify an uploaded image using the AI model.
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        image = load_image(io.BytesIO(contents))

        # Preprocess the image
        input_batch = preprocess_image(image, device)

        # Perform prediction
        from model import predict
        predicted_class = predict(model, input_batch, class_names)

        return JSONResponse(content={"predicted_class": predicted_class}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# Payroll API
# ---------------------------------------------------------------------------

class EmployeeCreate(BaseModel):
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
    hours_worked: Optional[float] = None  # required for hourly employees
    pay_period_index: int = 0             # period label stored with the payslip


class PayrollRunRequest(BaseModel):
    # Optional mapping of employee_id -> hours worked for this period.
    # Required entries for hourly employees; salaried employees ignore it.
    hours: Optional[dict] = None
    # Optional tax table override for this run:
    # "single", "married_joint", or "head_of_household".
    filing_status: Optional[str] = None
    # Optionally restrict the run to one company's employees.
    company_id: Optional[str] = None


@app.post("/payroll/employees")
async def create_employee(payload: EmployeeCreate):
    """Register a new employee in the payroll system."""
    try:
        employee = Employee(
            name=payload.name,
            pay_type=payload.pay_type,
            annual_salary=Decimal(str(payload.annual_salary)) if payload.annual_salary is not None else None,
            hourly_rate=Decimal(str(payload.hourly_rate)) if payload.hourly_rate is not None else None,
            benefits_deduction_per_period=Decimal(str(payload.benefits_deduction_per_period)),
            pay_periods_per_year=payload.pay_periods_per_year,
            filing_status=payload.filing_status,
            state=payload.state,
            company_id=payload.company_id,
            retirement_401k_percent=Decimal(str(payload.retirement_401k_percent)),
        )
        stored = add_employee(employee)
        return JSONResponse(content={
            "id": stored.id,
            "name": stored.name,
            "pay_type": stored.pay_type,
            "annual_salary": float(stored.annual_salary) if stored.annual_salary is not None else None,
            "hourly_rate": float(stored.hourly_rate) if stored.hourly_rate is not None else None,
            "filing_status": stored.filing_status,
            "state": stored.state,
            "company_id": stored.company_id,
            "retirement_401k_percent": float(stored.retirement_401k_percent),
        }, status_code=200)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/payroll/employees")
async def get_all_employees(company_id: Optional[str] = None):
    """List employees, optionally filtered by company_id."""
    return JSONResponse(content=[
        {
            "id": e.id,
            "name": e.name,
            "pay_type": e.pay_type,
            "annual_salary": float(e.annual_salary) if e.annual_salary is not None else None,
            "hourly_rate": float(e.hourly_rate) if e.hourly_rate is not None else None,
            "filing_status": e.filing_status,
            "state": e.state,
            "company_id": e.company_id,
            "retirement_401k_percent": float(e.retirement_401k_percent),
        }
        for e in list_employees(company_id=company_id)
    ], status_code=200)


@app.get("/payroll/employees/{employee_id}")
async def read_employee(employee_id: int):
    """Fetch a single employee by ID."""
    employee = get_employee(employee_id)
    if employee is None:
        return JSONResponse(content={"error": f"Employee {employee_id} not found"}, status_code=404)
    return JSONResponse(content={
        "id": employee.id,
        "name": employee.name,
        "pay_type": employee.pay_type,
        "annual_salary": float(employee.annual_salary) if employee.annual_salary is not None else None,
        "hourly_rate": float(employee.hourly_rate) if employee.hourly_rate is not None else None,
        "filing_status": employee.filing_status,
        "state": employee.state,
        "company_id": employee.company_id,
        "retirement_401k_percent": float(employee.retirement_401k_percent),
    }, status_code=200)


@app.delete("/payroll/employees/{employee_id}")
async def remove_employee(employee_id: int):
    """Remove an employee from the payroll system."""
    if delete_employee(employee_id):
        return JSONResponse(content={"deleted": employee_id}, status_code=200)
    return JSONResponse(content={"error": f"Employee {employee_id} not found"}, status_code=404)


@app.post("/payroll/employees/{employee_id}/payslip")
async def create_payslip(employee_id: int, payload: PayslipRequest):
    """Generate a payslip for one pay period (recorded to pay history)."""
    employee = get_employee(employee_id)
    if employee is None:
        return JSONResponse(content={"error": f"Employee {employee_id} not found"}, status_code=404)
    try:
        hours = Decimal(str(payload.hours_worked)) if payload.hours_worked is not None else None
        slip = generate_payslip(employee, hours_worked=hours,
                                pay_period_index=payload.pay_period_index)
        slip["history_id"] = record_payslip(slip, pay_period_index=payload.pay_period_index)
        return JSONResponse(content=slip, status_code=200)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/payroll/employees/{employee_id}/payslips")
async def read_pay_history(employee_id: int):
    """Return the recorded payslip history for an employee, newest first."""
    if get_employee(employee_id) is None:
        return JSONResponse(content={"error": f"Employee {employee_id} not found"}, status_code=404)
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
        return JSONResponse(content={"error": f"Employee {employee_id} not found"}, status_code=404)
    return JSONResponse(content={
        "id": updated.id,
        "name": updated.name,
        "pay_type": updated.pay_type,
        "annual_salary": float(updated.annual_salary) if updated.annual_salary is not None else None,
        "hourly_rate": float(updated.hourly_rate) if updated.hourly_rate is not None else None,
        "benefits_deduction_per_period": float(updated.benefits_deduction_per_period),
        "pay_periods_per_year": updated.pay_periods_per_year,
        "filing_status": updated.filing_status,
        "state": updated.state,
        "company_id": updated.company_id,
        "retirement_401k_percent": float(updated.retirement_401k_percent),
    }, status_code=200)


@app.get("/payroll/liabilities")
async def read_liabilities(company_id: Optional[str] = None):
    """
    Aggregate employer payroll tax liabilities (FICA match + FUTA/SUTA)
    across all recorded payslips, optionally filtered by company_id.
    """
    return JSONResponse(content=get_liabilities_report(company_id=company_id), status_code=200)


@app.get("/payroll/employees/{employee_id}/ytd")
async def read_ytd_summary(employee_id: int):
    """
    Year-to-date (all recorded periods) summary for one employee: wages,
    401(k), income-tax wages, every withholding, deductions, and net pay.
    """
    if get_employee(employee_id) is None:
        return JSONResponse(content={"error": f"Employee {employee_id} not found"}, status_code=404)
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
        return JSONResponse(content={"error": f"Payslip {history_id} not found"
                                            f" for employee {employee_id}"}, status_code=404)
    pdf_bytes = render_payslip_pdf(record)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=payslip_{employee_id}_{history_id}.pdf"
        },
    )


@app.post("/payroll/run")
async def run_payroll_for_period(payload: PayrollRunRequest):
    """
    Run payroll for all employees for one pay period. Generates a payslip
    per employee plus period totals. Hourly employees require an entry in
    the `hours` mapping; missing entries are reported in `errors`.
    """
    if payload.filing_status is not None and payload.filing_status not in BRACKET_PRESETS:
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
                         company_id=payload.company_id)
    return JSONResponse(content=result, status_code=200)

@app.get("/")
async def root():
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
