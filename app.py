from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from data_utils import load_image, preprocess_image, get_image_classes
from model import load_model
from payroll import (
    Employee,
    add_employee,
    get_employee,
    list_employees,
    delete_employee,
    generate_payslip,
    run_payroll,
    BRACKET_PRESETS,
)
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


class PayslipRequest(BaseModel):
    hours_worked: Optional[float] = None  # required for hourly employees


class PayrollRunRequest(BaseModel):
    # Optional mapping of employee_id -> hours worked for this period.
    # Required entries for hourly employees; salaried employees ignore it.
    hours: Optional[dict] = None
    # Optional tax table override for this run:
    # "single", "married_joint", or "head_of_household".
    filing_status: Optional[str] = None


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
        )
        stored = add_employee(employee)
        return JSONResponse(content={
            "id": stored.id,
            "name": stored.name,
            "pay_type": stored.pay_type,
            "annual_salary": float(stored.annual_salary) if stored.annual_salary is not None else None,
            "hourly_rate": float(stored.hourly_rate) if stored.hourly_rate is not None else None,
        }, status_code=200)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/payroll/employees")
async def get_all_employees():
    """List all employees registered for payroll."""
    return JSONResponse(content=[
        {
            "id": e.id,
            "name": e.name,
            "pay_type": e.pay_type,
            "annual_salary": float(e.annual_salary) if e.annual_salary is not None else None,
            "hourly_rate": float(e.hourly_rate) if e.hourly_rate is not None else None,
        }
        for e in list_employees()
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
    }, status_code=200)


@app.delete("/payroll/employees/{employee_id}")
async def remove_employee(employee_id: int):
    """Remove an employee from the payroll system."""
    if delete_employee(employee_id):
        return JSONResponse(content={"deleted": employee_id}, status_code=200)
    return JSONResponse(content={"error": f"Employee {employee_id} not found"}, status_code=404)


@app.post("/payroll/employees/{employee_id}/payslip")
async def create_payslip(employee_id: int, payload: PayslipRequest):
    """Generate a payslip for one pay period, including tax and deductions."""
    employee = get_employee(employee_id)
    if employee is None:
        return JSONResponse(content={"error": f"Employee {employee_id} not found"}, status_code=404)
    try:
        hours = Decimal(str(payload.hours_worked)) if payload.hours_worked is not None else None
        slip = generate_payslip(employee, hours_worked=hours)
        return JSONResponse(content=slip, status_code=200)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


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
    result = run_payroll(hours_by_employee=hours_map, tax_brackets=brackets)
    return JSONResponse(content=result, status_code=200)

@app.get("/")
async def root():
    return {"message": "NVIDIA Blackwell AI Classifier API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
