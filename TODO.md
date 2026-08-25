# TODO: E2E NVIDIA Blackwell AI System

- [x] Create requirements.txt with AI/ML dependencies (PyTorch with CUDA, NumPy, etc.)
- [x] Create main.py: Core AI script for a simple image classifier using pre-trained model on Blackwell GPU
- [x] Create data_utils.py: For data loading and preprocessing
- [x] Create model.py: Define and load AI models
- [x] Create app.py: FastAPI-based API for inference
- [x] Create Dockerfile: For containerized deployment with GPU support
- [x] Update README.md: With setup and usage instructions
- [x] Install dependencies
- [x] Test the system (run training/inference)
- [x] Verify GPU usage

## Payroll Module

- [x] Create payroll.py: Employee records, gross pay (salaried/hourly + overtime), progressive tax, payslip generation
- [x] Add /payroll REST endpoints to app.py
- [x] Add test_payroll.py unit + API tests
- [x] SQLite persistence for employees (PAYROLL_DB_PATH env var / configure_store())
- [x] Real 2024 US Federal tax bracket presets (single / married_joint / head_of_household) with per-run filing_status override
- [x] Batch "run payroll" endpoint (/payroll/run) with hours input and period totals
- [x] FICA withholding: Social Security (6.2%, $168,600 wage base), Medicare (1.45% + 0.9% surtax over $200k)
- [x] Per-employee filing status stored in SQLite with automatic schema migration
- [x] Persisted pay history table + GET /payroll/employees/{id}/payslips endpoint
- [x] Employee update endpoint (PUT /payroll/employees/{id}, partial updates)
- [x] Flat-rate state income tax presets (STATE_TAX_RATES) with per-employee state
- [x] Employer FICA match + GET /payroll/liabilities aggregate report
- [x] Payslip PDF export via GET /payroll/employees/{id}/payslips/{history_id}/pdf (fpdf2)
- [x] Progressive state income tax tables (CA, NY) alongside flat-rate presets
- [x] Employer unemployment taxes: FUTA (0.6% on first $7k) + SUTA (configurable rate, per-state wage bases)
- [x] Multi-company support via company_id with filtering for list/run/liabilities
