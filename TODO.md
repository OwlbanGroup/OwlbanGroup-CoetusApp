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
- [ ] Replace illustrative default tax brackets with real jurisdiction tables
- [ ] Optional: batch "run payroll" endpoint across all employees per period
