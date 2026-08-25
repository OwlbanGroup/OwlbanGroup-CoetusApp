"""
Unit tests for the payroll module and its FastAPI endpoints.

Run with: python -m pytest test_payroll.py -v
"""

import os
import tempfile

from decimal import Decimal
from fastapi.testclient import TestClient

import payroll
from payroll import Employee, calculate_gross_pay, calculate_tax, generate_payslip


def fresh_store():
    """Point payroll at a brand-new temporary SQLite database."""
    payroll.configure_store(os.path.join(tempfile.mkdtemp(), "payroll_test.db"))


# ---------------------------------------------------------------------------
# Core calculation tests
# ---------------------------------------------------------------------------

def make_salaried(annual="52000", benefits="0"):
    return Employee(
        name="Alice",
        pay_type="salaried",
        annual_salary=Decimal(annual),
        benefits_deduction_per_period=Decimal(benefits),
        pay_periods_per_year=26,
    )


def make_hourly(rate="20", benefits="0"):
    return Employee(
        name="Bob",
        pay_type="hourly",
        hourly_rate=Decimal(rate),
        benefits_deduction_per_period=Decimal(benefits),
    )


def test_salaried_gross_pay():
    emp = make_salaried("52000")
    # 52000 / 26 = 2000.00 per period
    assert calculate_gross_pay(emp) == Decimal("2000.00")


def test_hourly_regular_hours():
    emp = make_hourly("20")
    assert calculate_gross_pay(emp, hours_worked=Decimal("40")) == Decimal("800.00")


def test_hourly_overtime_paid_at_time_and_a_half():
    emp = make_hourly("20")
    # 40 regular * 20 + 10 overtime * 30 = 1100.00
    assert calculate_gross_pay(emp, hours_worked=Decimal("50")) == Decimal("1100.00")


def test_hourly_requires_hours():
    try:
        calculate_gross_pay(make_hourly())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_progressive_tax_brackets():
    brackets = [
        (Decimal("10000"), Decimal("0.10")),
        (Decimal("20000"), Decimal("0.20")),
        (None, Decimal("0.30")),
    ]
    payroll.set_tax_brackets(brackets)
    try:
        # 10000*0.10 + 10000*0.20 + 5000*0.30 = 1000 + 2000 + 1500 = 4500
        assert calculate_tax(Decimal("25000")) == Decimal("4500.00")
        # Below first bracket boundary only
        assert calculate_tax(Decimal("5000")) == Decimal("500.00")
    finally:
        payroll.set_tax_brackets(payroll.DEFAULT_TAX_BRACKETS)


def test_payslip_net_pay_math():
    payroll.set_tax_brackets([(None, Decimal("0.10"))])
    try:
        emp = make_salaried("52000", benefits="100")
        slip = generate_payslip(emp)
        assert slip["gross_pay"] == 2000.00
        # annual gross 52000 * 10% = 5200 tax; per period = 200.00
        assert slip["tax_withheld"] == 200.00
        assert slip["benefits_deduction"] == 100.00
        assert slip["net_pay"] == 1700.00
    finally:
        payroll.set_tax_brackets(payroll.DEFAULT_TAX_BRACKETS)


def test_employee_validation_rejects_bad_pay_type():
    emp = Employee(name="X", pay_type="commission")
    try:
        emp.validate()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_employee_validation_requires_salary_for_salaried():
    emp = Employee(name="X", pay_type="salaried")
    try:
        emp.validate()
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Tax bracket preset tests
# ---------------------------------------------------------------------------

def test_bracket_presets_married_lower_than_single():
    income = Decimal("100000")
    payroll.set_tax_brackets(payroll.BRACKET_PRESETS["single"])
    try:
        single_tax = calculate_tax(income)
        payroll.set_tax_brackets(payroll.BRACKET_PRESETS["married_joint"])
        married_tax = calculate_tax(income)
        assert married_tax < single_tax
    finally:
        payroll.set_tax_brackets(payroll.DEFAULT_TAX_BRACKETS)


def test_calculate_tax_single_bracket_expected_amount():
    # Single filer, $60,000 annual:
    # 11600*.10 + 35550*.12 + (60000-47150)*.22
    # = 1160 + 4266 + 2827 = 8253
    payroll.set_tax_brackets(payroll.BRACKET_PRESETS["single"])
    try:
        assert calculate_tax(Decimal("60000")) == Decimal("8253.00")
    finally:
        pass


def test_calculate_tax_explicit_brackets_override_active():
    payroll.set_tax_brackets(payroll.BRACKET_PRESETS["single"])
    custom = [(None, Decimal("0.05"))]
    # Explicit table wins over the active single-filer table.
    assert calculate_tax(Decimal("100000"), brackets=custom) == Decimal("5000.00")


def test_set_tax_brackets_by_status_invalid_raises():
    try:
        payroll.set_tax_brackets_by_status("nonexistent")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Batch payroll run tests
# ---------------------------------------------------------------------------

def test_run_payroll_mixed_employees_reports_errors_and_totals():
    fresh_store()
    salaried = payroll.add_employee(make_salaried("52000"))
    hourly = payroll.add_employee(make_hourly("20"))
    missing_hours = payroll.add_employee(make_hourly("25"))

    result = payroll.run_payroll(hours_by_employee={hourly.id: Decimal("40")})

    assert result["employees_paid"] == 2
    assert result["employees_errored"] == 1
    assert result["errors"][0]["employee_id"] == missing_hours.id

    slip_ids = {s["employee_id"] for s in result["payslips"]}
    assert slip_ids == {salaried.id, hourly.id}

    summed_gross = sum(s["gross_pay"] for s in result["payslips"])
    assert abs(result["totals"]["gross_pay"] - summed_gross) < 0.01
    # Salaried: 2000/period; hourly: 40 * 20 = 800 -> gross total 2800
    assert result["totals"]["gross_pay"] == 2800.00


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------

def setup_function(function):
    fresh_store()


def test_add_get_delete_employee():
    added = payroll.add_employee(make_salaried())
    assert added.id > 0
    fetched = payroll.get_employee(added.id)
    assert fetched is not None and fetched.name == "Alice"
    assert payroll.delete_employee(added.id) is True
    assert payroll.get_employee(added.id) is None
    assert payroll.delete_employee(added.id) is False


def test_decimal_precision_survives_roundtrip():
    emp = make_salaried("52000.55", benefits="123.45")
    stored = payroll.add_employee(emp)
    fetched = payroll.get_employee(stored.id)
    assert fetched.annual_salary == Decimal("52000.55")
    assert fetched.benefits_deduction_per_period == Decimal("123.45")
    assert fetched.pay_periods_per_year == 26


def test_persistence_across_store_recreation(tmp_path):
    db_file = os.path.join(str(tmp_path), "persist.db")
    payroll.configure_store(db_file)
    stored = payroll.add_employee(make_hourly("25.50"))

    # Simulate a restart: re-open the same database file fresh
    payroll.configure_store(db_file)
    fetched = payroll.get_employee(stored.id)
    assert fetched is not None
    assert fetched.name == "Bob"
    assert fetched.hourly_rate == Decimal("25.50")


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

client = None


def get_client():
    global client
    if client is None:
        from app import app
        client = TestClient(app)
    return client


def test_api_create_and_list_employees():
    c = get_client()
    resp = c.post("/payroll/employees", json={
        "name": "Carol", "pay_type": "hourly", "hourly_rate": 25.0,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Carol"

    listing = c.get("/payroll/employees").json()
    assert any(e["id"] == body["id"] for e in listing)


def test_api_payslip_endpoint():
    c = get_client()
    created = c.post("/payroll/employees", json={
        "name": "Dave", "pay_type": "salaried",
        "annual_salary": 104000, "pay_periods_per_year": 52,
    }).json()

    resp = c.post(f"/payroll/employees/{created['id']}/payslip", json={"hours_worked": None})
    assert resp.status_code == 200
    slip = resp.json()
    assert slip["gross_pay"] == 2000.00
    assert abs(slip["net_pay"] + slip["tax_withheld"] - 2000.00) < 0.01


def test_api_payslip_unknown_employee_returns_404():
    c = get_client()
    resp = c.post("/payroll/employees/999999/payslip", json={})
    assert resp.status_code == 404


def test_api_invalid_employee_returns_400():
    c = get_client()
    resp = c.post("/payroll/employees", json={"name": "", "pay_type": "salaried"})
    assert resp.status_code == 400


def test_api_run_payroll_endpoint():
    c = get_client()
    salaried = c.post("/payroll/employees", json={
        "name": "Salaried Sam", "pay_type": "salaried", "annual_salary": 52000,
    }).json()
    hourly = c.post("/payroll/employees", json={
        "name": "Hourly Hank", "pay_type": "hourly", "hourly_rate": 20.0,
    }).json()

    resp = c.post("/payroll/run", json={"hours": {str(hourly["id"]): 40}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["employees_paid"] == 2
    assert body["employees_errored"] == 0
    assert body["totals"]["gross_pay"] == 2800.00  # 2000 + 800

    slip_by_id = {s["employee_id"]: s for s in body["payslips"]}
    assert slip_by_id[salaried["id"]]["gross_pay"] == 2000.00
    assert slip_by_id[hourly["id"]]["gross_pay"] == 800.00


def test_api_run_payroll_missing_hourly_hours_reported_in_errors():
    c = get_client()
    hourly = c.post("/payroll/employees", json={
        "name": "No Hours Nora", "pay_type": "hourly", "hourly_rate": 30.0,
    }).json()

    resp = c.post("/payroll/run", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["employees_paid"] == 0
    assert body["employees_errored"] == 1
    assert body["errors"][0]["employee_id"] == hourly["id"]


def test_api_run_payroll_invalid_filing_status_returns_400():
    c = get_client()
    resp = c.post("/payroll/run", json={"filing_status": "bogus"})
    assert resp.status_code == 400


def test_api_run_payroll_unknown_hours_employee_returns_404():
    c = get_client()
    resp = c.post("/payroll/run", json={"hours": {"999999": 10}})
    assert resp.status_code == 404
