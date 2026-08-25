# OwlbanGroup-CoetusApp: E2E NVIDIA Blackwell AI System

This project implements an end-to-end AI system optimized for NVIDIA Blackwell GPUs, featuring image classification using pre-trained models, data preprocessing, and a FastAPI-based API for inference.

## Features

- **GPU Acceleration**: Leverages NVIDIA Blackwell GPUs for high-performance AI computations.
- **Image Classification**: Uses ResNet50 pre-trained model for ImageNet classification.
- **API Interface**: FastAPI-based REST API for easy integration.
- **Containerized Deployment**: Docker support for GPU-enabled containers.
- **Modular Design**: Separated utilities for data handling, model management, and inference.

## Prerequisites

- NVIDIA Blackwell GPU with CUDA 12.4+ support
- Docker (for containerized deployment)
- Python 3.8+ (for local development)

## Installation

### Option 1: Docker Deployment (Recommended)

1. Build the Docker image:
   ```bash
   docker build -t blackwell-ai-system .
   ```

2. Run the container with GPU support:
   ```bash
   docker run --gpus all -p 8000:8000 blackwell-ai-system
   ```

### Option 2: Local Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure CUDA is available:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Usage

### API Usage

Start the API server:
```bash
python app.py
```

The API will be available at `http://localhost:8000`.

#### Classify an Image

Upload an image file to classify:

```bash
curl -X POST "http://localhost:8000/classify" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/image.jpg"
```

Response:
```json
{
  "predicted_class": "golden retriever"
}
```

### Command-Line Usage

Classify an image directly:

```bash
python main.py path/to/image.jpg
```

## Project Structure

- `main.py`: Command-line interface for image classification
- `app.py`: FastAPI application for API-based inference (image classification + payroll)
- `model.py`: Model loading and prediction utilities
- `data_utils.py`: Data preprocessing and utility functions
- `payroll.py`: Payroll module — employee records (SQLite-backed), gross pay, tax withholding, payslips
- `test_payroll.py`: Unit and API tests for the payroll module (`python -m pytest test_payroll.py -v`)
- `requirements.txt`: Python dependencies
- `Dockerfile`: Containerization configuration

## Payroll API

The service includes payroll management endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/payroll/employees` | Register an employee (`name`, `pay_type` of `salaried`/`hourly`, `annual_salary` or `hourly_rate`, optional `filing_status`) |
| GET | `/payroll/employees` | List all employees |
| GET | `/payroll/employees/{id}` | Fetch one employee |
| PUT | `/payroll/employees/{id}` | Partially update an employee (rate, benefits, filing status, state, etc.) |
| DELETE | `/payroll/employees/{id}` | Remove an employee |
| POST | `/payroll/employees/{id}/payslip` | Generate a payslip (pass `hours_worked` for hourly employees); recorded to pay history |
| GET | `/payroll/employees/{id}/payslips` | Employee's recorded payslip history, newest first |
| GET | `/payroll/employees/{id}/payslips/{history_id}/pdf` | Download one recorded payslip as PDF |
| GET | `/payroll/liabilities` | Employer payroll tax liabilities aggregated from pay history |
| POST | `/payroll/run` | Batch-run payroll for all employees for one pay period; every slip is recorded to pay history |

Example — create a salaried employee and generate a payslip:

```bash
curl -X POST http://localhost:8000/payroll/employees \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "pay_type": "salaried", "annual_salary": 52000}'

curl -X POST http://localhost:8000/payroll/employees/1/payslip \
  -H "Content-Type: application/json" -d '{}'
```

Batch-run payroll for all employees in one call:

```bash
curl -X POST http://localhost:8000/payroll/run \
  -H "Content-Type: application/json" \
  -d '{"hours": {"2": 45}, "filing_status": "single"}'
```

The response contains a payslip per employee, period totals (`gross_pay`, `tax_withheld`, `benefits_deduction`, `net_pay`), and an `errors` list for hourly employees who were missing from `hours`.

Payslip calculations use `Decimal` precision: salaried gross pay is `annual_salary / pay_periods_per_year`; hourly pay includes overtime at 1.5× above 40 hours; benefit deductions are subtracted per period.

### Withholdings

Each payslip includes these withholding components:

- **Federal income tax** — progressive brackets on annualized gross (see Tax tables below)
- **State income tax** — per-employee preset (`state` field). Flat-rate states (`az`, `co`, `il`, `in`, `nc`, `pa`) and progressive schedules (`ca`, `ny` — simplified 2024 single-filer tables), or `none`/null
- **Social Security** — 6.2% up to the $168,600 annual wage base
- **Medicare** — 1.45% of all wages, plus the 0.9% Additional Medicare surtax above $200,000

`net_pay = gross_pay − tax_withheld − social_security − medicare − state_tax_withheld − benefits_deduction`.

### Employer liabilities

The employer also owes taxes on every payslip (recorded alongside each history row):

- **FICA match** — Social Security 6.2% (same wage base) + Medicare 1.45%, without the employee-only 0.9% surtax
- **FUTA** — effective 0.6% on the first $7,000 of wages per employee per year
- **SUTA** — configurable rate (`payroll.set_suta_rate()`, default 2.7%) applied to each state's taxable wage base

`GET /payroll/liabilities` aggregates all recorded payslips into a report of gross wages, employee withholdings, employer taxes, total FICA liability, and total employer liability. Pass `?company_id=` to scope it.

### Multi-company support

Every employee can carry an optional `company_id` tag. It filters `GET /payroll/employees?company_id=`, scopes `POST /payroll/run`, is snapshotted into every recorded payslip, and scopes `GET /payroll/liabilities?company_id=` — so several companies can share one payroll database while keeping reporting separate.

### Payslip PDF export

Any recorded payslip can be downloaded as a formatted PDF:

```bash
curl -O -J http://localhost:8000/payroll/employees/1/payslips/1/pdf
```

### Tax tables

Withholding uses progressive marginal brackets on annualized gross pay. The module ships with **2024 US Federal bracket presets** (IRS Rev. Proc. 2023-34) for three filing statuses — `single` (the default), `married_joint`, and `head_of_household`. These are for estimation only and exclude state/local taxes and standard-deduction effects.

Each employee stores their own `filing_status` (set at creation, defaults to `single`), which drives their federal withholding. Resolution order: explicit `tax_brackets` argument → the employee's `filing_status` preset → the module's active table. `/payroll/run` still accepts a per-run `filing_status` that overrides everyone for that run.

- Per-run override: pass `filing_status` to `/payroll/run`
- Programmatic: `payroll.set_tax_brackets_by_status("married_joint")` or fully custom tables via `payroll.set_tax_brackets([...])`

### Pay history

Every payslip generated via `POST /payroll/employees/{id}/payslip` or `POST /payroll/run` is persisted to a `pay_history` SQLite table (with a timestamp and period index) and retrievable via `GET /payroll/employees/{id}/payslips`.

### Payroll persistence

Employees are stored in a SQLite database (`payroll.db` in the working directory by default). Override the location with the `PAYROLL_DB_PATH` environment variable or call `payroll.configure_store(path)` programmatically. Data survives application restarts; monetary values are stored as TEXT to preserve decimal precision.

Run the payroll tests with:

```bash
python -m pytest test_payroll.py -v
```

## GPU Verification

To verify GPU usage:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.get_device_name(0)}")
```

## Contributing

Contributions are welcome! Please ensure all changes are tested with GPU acceleration.

## License

This project is licensed under the MIT License.
