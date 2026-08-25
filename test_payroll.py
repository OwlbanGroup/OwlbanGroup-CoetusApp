"""
Unit tests for the payroll module and its FastAPI endpoints.

Run with: python -m pytest test_payroll.py -v
"""

import csv
import io
import os
import tempfile

from decimal import Decimal
from functools import lru_cache

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
    """Build a salaried Employee with sensible defaults."""
    return Employee(
        name="Alice",
        pay_type="salaried",
        annual_salary=Decimal(annual),
        benefits_deduction_per_period=Decimal(benefits),
        pay_periods_per_year=26,
    )


def make_hourly(rate="20", benefits="0"):
    """Build an hourly Employee with sensible defaults."""
    return Employee(
        name="Bob",
        pay_type="hourly",
        hourly_rate=Decimal(rate),
        benefits_deduction_per_period=Decimal(benefits),
    )


def test_salaried_gross_pay():
    """Annual salary divided by periods gives gross pay."""
    emp = make_salaried("52000")
    # 52000 / 26 = 2000.00 per period
    assert calculate_gross_pay(emp) == Decimal("2000.00")


def test_hourly_regular_hours():
    """Regular hours are paid at straight time."""
    emp = make_hourly("20")
    assert calculate_gross_pay(emp, hours_worked=Decimal("40")) == Decimal("800.00")


def test_hourly_overtime_paid_at_time_and_a_half():
    """Hours beyond 40 are paid at 1.5x."""
    emp = make_hourly("20")
    # 40 regular * 20 + 10 overtime * 30 = 1100.00
    assert calculate_gross_pay(emp, hours_worked=Decimal("50")) == Decimal("1100.00")


def test_hourly_requires_hours():
    """Hourly employees without hours raise ValueError."""
    try:
        calculate_gross_pay(make_hourly())
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_progressive_tax_brackets():
    """Tax is computed marginally across brackets."""
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
    """Full payslip math from gross down to net pay."""
    emp = make_salaried("52000", benefits="100")
    slip = generate_payslip(emp, tax_brackets=[(None, Decimal("0.10"))])
    assert slip["gross_pay"] == 2000.00
    # Federal: annual 52000 * 10% = 5200 -> 200.00 per period
    assert slip["tax_withheld"] == 200.00
    # FICA on 52000 annualized: SS 3224.00 -> 124.00; Medicare 754.00 -> 29.00
    assert slip["social_security"] == 124.00
    assert slip["medicare"] == 29.00
    assert slip["benefits_deduction"] == 100.00
    assert slip["total_deductions"] == 453.00
    assert slip["net_pay"] == 1547.00


def test_fica_amounts_for_hourly():
    """FICA is based on annualized hourly gross."""
    emp = make_hourly("20")
    slip = generate_payslip(emp, hours_worked=Decimal("40"))
    # annualized 800 * 26 = 20800: SS 1289.60 -> 49.60; Medicare 301.60 -> 11.60
    assert slip["social_security"] == 49.60
    assert slip["medicare"] == 11.60


def test_social_security_wage_base_cap():
    """Social Security stops at the annual wage base."""
    emp = Employee(name="High Earner", pay_type="salaried",
                   annual_salary=Decimal("400000"), pay_periods_per_year=12)
    slip = generate_payslip(emp)
    # SS capped at 168600 * 6.2% = 10453.20 per year -> 871.10 per month
    assert slip["social_security"] == 871.10


def test_calculate_fica_includes_additional_medicare():
    """High earners owe the Additional Medicare surtax."""
    ss, medicare = payroll.calculate_fica(Decimal("250000"))
    assert ss == Decimal("168600") * Decimal("0.062")
    expected = Decimal("250000") * Decimal("0.0145") + \
        Decimal("50000") * Decimal("0.009")
    assert medicare == expected


def test_employee_filing_status_drives_withholding():
    """Filing status changes federal withholding, not FICA."""
    single_emp = make_salaried("104000")
    married_emp = make_salaried("104000")
    married_emp.filing_status = "married_joint"
    s = generate_payslip(single_emp)
    m = generate_payslip(married_emp)
    assert s["filing_status"] == "single"
    assert m["filing_status"] == "married_joint"
    assert m["tax_withheld"] < s["tax_withheld"]
    # FICA does not depend on filing status
    assert s["social_security"] == m["social_security"]
    assert s["medicare"] == m["medicare"]


def test_employee_validation_rejects_bad_pay_type():
    """Unsupported pay types fail validation."""
    emp = Employee(name="X", pay_type="commission")
    try:
        emp.validate()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_employee_validation_requires_salary_for_salaried():
    """Salaried employees require an annual salary."""
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
    """Married-joint preset taxes less than single."""
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
    """Single-filer preset yields the exact expected tax."""
    # Single filer, $60,000 annual:
    # 11600*.10 + 35550*.12 + (60000-47150)*.22
    # = 1160 + 4266 + 2827 = 8253
    payroll.set_tax_brackets(payroll.BRACKET_PRESETS["single"])
    try:
        assert calculate_tax(Decimal("60000")) == Decimal("8253.00")
    finally:
        pass


def test_calculate_tax_explicit_brackets_override_active():
    """Explicit brackets win over the active preset."""
    payroll.set_tax_brackets(payroll.BRACKET_PRESETS["single"])
    custom = [(None, Decimal("0.05"))]
    # Explicit table wins over the active single-filer table.
    assert calculate_tax(Decimal("100000"), brackets=custom) == Decimal("5000.00")


def test_set_tax_brackets_by_status_invalid_raises():
    """Unknown filing status raises ValueError."""
    try:
        payroll.set_tax_brackets_by_status("nonexistent")
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Batch payroll run tests
# ---------------------------------------------------------------------------

def test_run_payroll_mixed_employees_reports_errors_and_totals():
    """Batch runs pay valid employees and report errors."""
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


def test_invalid_filing_status_rejected():
    """Bad filing status blocks persistence."""
    emp = make_salaried()
    emp.filing_status = "bogus"
    fresh_store()
    try:
        payroll.add_employee(emp)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# State tax & employer tax tests
# ---------------------------------------------------------------------------

def test_state_tax_flat_rate_withheld():
    """Flat-rate states withhold a fixed percentage."""
    emp = make_salaried("52000")
    emp.state = "pa"  # 3.07%
    slip = generate_payslip(emp)
    # annualized 52000 * 0.0307 = 1596.40 -> 61.40 per period
    assert slip["state"] == "pa"
    assert slip["state_tax_withheld"] == 61.40


def test_no_state_means_zero_state_tax():
    """No state means zero state tax."""
    emp = make_salaried("52000")  # state defaults to None
    assert generate_payslip(emp)["state_tax_withheld"] == 0.0
    emp_none = make_salaried("52000")
    emp_none.state = "none"
    assert generate_payslip(emp_none)["state_tax_withheld"] == 0.0


def test_invalid_state_rejected():
    """Unknown state codes fail validation."""
    emp = make_salaried()
    emp.state = "zz"
    try:
        emp.validate()
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Pre-tax 401(k) tests
# ---------------------------------------------------------------------------

def test_401k_reduces_income_tax_but_not_fica():
    """401(k) lowers income-tax wages but not FICA wages."""
    payroll.set_tax_brackets(payroll.BRACKET_PRESETS["single"])
    try:
        emp = make_salaried("52000")
        emp.retirement_401k_percent = Decimal("10")
        slip = generate_payslip(emp)
        assert slip["retirement_401k"] == 200.00  # 10% of 2000
        # Income-tax wages drop to 46,800: fed = 1160 + 35200*.12 = 5384 -> 207.08
        assert slip["tax_withheld"] == 207.00 or slip["tax_withheld"] == 207.08
        # FICA still on full gross
        assert slip["social_security"] == 124.00
        assert slip["medicare"] == 29.00
        assert slip["net_pay"] == round(2000 - slip["tax_withheld"] - 124 - 29 - 200, 2)
    finally:
        payroll.set_tax_brackets(payroll.DEFAULT_TAX_BRACKETS)


def test_401k_percent_validation_bounds():
    """401(k) percent must be between 0 and 100."""
    emp = make_salaried()
    emp.retirement_401k_percent = Decimal("101")
    try:
        emp.validate()
        assert False, "expected ValueError"
    except ValueError:
        pass
    emp.retirement_401k_percent = Decimal("-1")
    try:
        emp.validate()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ytd_summary_aggregates_history():
    """YTD summary aggregates recorded payslips."""
    fresh_store()
    emp = payroll.add_employee(make_hourly("20"))
    for hours in ("40", "45"):
        payroll.record_payslip(
            generate_payslip(emp, hours_worked=Decimal(hours)))
    ytd = payroll.get_ytd_summary(emp.id)
    assert ytd["pay_periods_paid"] == 2
    assert ytd["hours_worked"] == 85.0
    assert ytd["gross_wages"] == 1750.00  # 800 + 1100
    assert ytd["withholdings"]["social_security"] > 0
    assert abs(ytd["taxable_wages_income_tax"]
               - (ytd["gross_wages"] - ytd["retirement_401k"])) < 0.01


def test_export_history_csv_content():
    """CSV export emits header plus joined data rows."""
    fresh_store()
    emp = payroll.add_employee(make_hourly("20"))
    payroll.record_payslip(generate_payslip(emp, hours_worked=Decimal("40")),
                           pay_period_index=4)
    csv_text = payroll.export_history_csv()
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("history_id,employee_id")
    assert len(lines) == 2
    row = lines[1].split(",")
    assert row[2] == "Bob"          # employee_name via join
    assert float(row[6]) == 800.0   # gross_pay


def test_export_history_csv_company_filter():
    """CSV export can filter by company."""
    fresh_store()
    a = make_salaried("52000")
    a.company_id = "acme"
    g = make_salaried("52000")
    g.company_id = "globex"
    ea = payroll.add_employee(a)
    eg = payroll.add_employee(g)
    payroll.run_payroll(company_id="acme")
    payroll.run_payroll(company_id="globex")
    acme_rows = list(csv.DictReader(io.StringIO(
        payroll.export_history_csv(company_id="acme"))))
    assert len(acme_rows) == 1
    assert acme_rows[0]["employee_id"] == str(ea.id)
    all_ids = {r["employee_id"] for r in csv.DictReader(io.StringIO(
        payroll.export_history_csv()))}
    assert all_ids == {str(ea.id), str(eg.id)}


def test_compute_employer_taxes_match_without_surtax():
    """Employer FICA mirrors employee rates sans surtax."""
    ss_match, medicare = payroll.compute_employer_taxes(Decimal("250000"))
    # Employer SS capped at wage base; no Additional Medicare surtax for employer
    assert ss_match == Decimal("168600") * Decimal("0.062")
    assert medicare == Decimal("250000") * Decimal("0.0145")


def test_payslip_includes_employer_amounts():
    """Payslips include employer-side FICA amounts."""
    emp = make_salaried("52000")
    slip = generate_payslip(emp)
    # Employee and employer FICA match on wages below the SS cap
    assert slip["employer_social_security"] == slip["social_security"]
    assert slip["employer_medicare"] == slip["medicare"]


# ---------------------------------------------------------------------------
# Progressive state tax tests
# ---------------------------------------------------------------------------

def test_california_progressive_state_tax():
    """CA applies progressive marginal rates."""
    # CA $60,000 annual: 10756*.01 + 14743*.02 + 14746*.04
    #   + 15621*.06 + (60000-55866)*.08 = 107.56+294.86+589.84+937.26+330.72
    emp = make_salaried("60000")
    emp.state = "ca"
    slip = generate_payslip(emp)
    annual_ca = Decimal("107.56") + Decimal("294.86") + Decimal("589.84") \
        + Decimal("937.26") + Decimal("330.72")
    expected_per_period = (annual_ca / Decimal("26")).quantize(Decimal("0.01"))
    assert abs(slip["state_tax_withheld"] - float(expected_per_period)) < 0.01


def test_new_york_progressive_higher_than_flat_pa_at_same_income():
    """NY progressive beats PA flat at high incomes."""
    income = Decimal("100000")
    ny = payroll.calculate_state_tax(income, "ny")
    pa = payroll.calculate_state_tax(income, "pa")
    assert ny > pa  # NY's top rates exceed PA's flat 3.07% at high incomes


def test_california_progressive_marginal_rates():
    """Only the first CA bracket applies below its boundary."""
    # Only the first bracket applies below its boundary: 10,000 * 1%
    assert payroll.calculate_state_tax(Decimal("10000"), "ca") == Decimal("100.00")


# ---------------------------------------------------------------------------
# Unemployment tax tests (FUTA/SUTA, employer-side)
# ---------------------------------------------------------------------------

def test_futa_capped_at_wage_base():
    """FUTA is capped at the federal wage base."""
    futa_low, _ = payroll.compute_employer_unemployment(Decimal("5000"), None)
    futa_high, _ = payroll.compute_employer_unemployment(Decimal("100000"), None)
    assert futa_low == Decimal("5000") * Decimal("0.006")
    assert futa_high == Decimal("7000") * Decimal("0.006")  # capped


def test_suta_uses_state_wage_base_and_configurable_rate():
    """SUTA honors state wage base and configured rate."""
    payroll.set_suta_rate(Decimal("0.031"))
    try:
        _, suta_co = payroll.compute_employer_unemployment(Decimal("50000"), "co")
        assert suta_co == Decimal("17000") * Decimal("0.031")
        # No SUTA without a state
        _, suta_none = payroll.compute_employer_unemployment(Decimal("50000"), None)
        assert suta_none == Decimal("0")
    finally:
        payroll.set_suta_rate(payroll.DEFAULT_SUTA_RATE)


def test_set_suta_rate_rejects_invalid():
    """Invalid SUTA rates raise ValueError."""
    try:
        payroll.set_suta_rate(Decimal("5"))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_payslip_includes_unemployment_amounts():
    """Payslips include employer FUTA/SUTA."""
    emp = make_salaried("52000")
    emp.state = "co"
    slip = generate_payslip(emp)
    # FUTA on annualized 52000 capped at 7000 * 0.6% = 42.00 -> 1.6154/period
    assert slip["employer_futa"] > 0
    assert slip["employer_suta"] > 0


# ---------------------------------------------------------------------------
# Multi-company tests
# ---------------------------------------------------------------------------

def test_list_employees_filters_by_company():
    """Listing filters employees by company."""
    fresh_store()
    acme = make_salaried("52000")
    acme.company_id = "acme"
    globex = make_salaried("52000")
    globex.company_id = "globex"
    a = payroll.add_employee(acme)
    g = payroll.add_employee(globex)
    ids = {e.id for e in payroll.list_employees(company_id="acme")}
    assert a.id in ids
    assert g.id not in ids


def test_run_payroll_respects_company_scope():
    """Runs only pay employees in scope."""
    fresh_store()
    acme = make_salaried("52000")
    acme.company_id = "acme"
    globex = make_hourly("20")
    globex.company_id = "globex"
    ea = payroll.add_employee(acme)
    eg = payroll.add_employee(globex)

    result = payroll.run_payroll(
        hours_by_employee={eg.id: Decimal("40")}, company_id="acme")
    assert result["employees_paid"] == 1
    assert result["payslips"][0]["employee_id"] == ea.id
    assert result["totals"]["gross_pay"] == 2000.00


def test_liabilities_report_filtered_by_company():
    """Liability reports can be company-scoped."""
    fresh_store()
    acme = make_salaried("52000")
    acme.company_id = "acme"
    globex = make_salaried("104000")
    globex.company_id = "globex"
    payroll.add_employee(acme)
    payroll.add_employee(globex)
    payroll.run_payroll()
    report_acme = payroll.get_liabilities_report(company_id="acme")
    assert report_acme["payslips_recorded"] == 1
    assert report_acme["company_id"] == "acme"
    full = payroll.get_liabilities_report()
    assert full["payslips_recorded"] == 2


# ---------------------------------------------------------------------------
# Employee update tests
# ---------------------------------------------------------------------------

def test_update_employee_partial():
    """Partial updates change only the supplied fields."""
    fresh_store()
    emp = payroll.add_employee(make_hourly("25"))
    emp.state = "co"
    payroll.update_employee(emp.id, {"state": "co"})  # persist the state first
    payroll.update_employee(
        emp.id, {"hourly_rate": "30", "name": "Renamed Bob"})
    fetched = payroll.get_employee(emp.id)
    assert fetched is not None
    assert fetched.hourly_rate == Decimal("30")
    assert fetched.name == "Renamed Bob"
    # untouched fields survive
    assert fetched.pay_type == "hourly"


def test_update_preserves_state_when_not_provided():
    """Omitted fields keep their stored values."""
    fresh_store()
    emp = payroll.add_employee(make_salaried("52000"))
    payroll.update_employee(emp.id, {"state": "il"})
    # Update something else without mentioning state -> it must be preserved
    payroll.update_employee(emp.id, {"annual_salary": "60000"})
    fetched = payroll.get_employee(emp.id)
    assert fetched is not None
    assert fetched.state == "il"


def test_update_employee_rejects_unknown_field():
    """Unknown update fields raise ValueError."""
    fresh_store()
    emp = payroll.add_employee(make_hourly("25"))
    try:
        payroll.update_employee(emp.id, {"pay_type": "salaried"})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_update_unknown_employee_returns_none():
    """Updating a ghost employee returns None."""
    fresh_store()
    assert payroll.update_employee(999999, {"name": "Ghost"}) is None


# ---------------------------------------------------------------------------
# Pay history tests
# ---------------------------------------------------------------------------

def test_payslip_history_roundtrip():
    """Payslips round-trip through history intact."""
    fresh_store()
    emp = payroll.add_employee(make_hourly("20"))
    slip = generate_payslip(emp, hours_worked=Decimal("40"))
    hid = payroll.record_payslip(slip, pay_period_index=3)
    hist = payroll.get_pay_history(emp.id)
    assert len(hist) == 1
    row = hist[0]
    assert row["id"] == hid
    assert row["employee_id"] == emp.id
    assert row["pay_period_index"] == 3
    assert row["gross_pay"] == 800.00
    assert row["net_pay"] == slip["net_pay"]
    assert row["recorded_at"]  # timestamp populated


def test_run_payroll_records_history():
    """Runs record one history row per employee."""
    fresh_store()
    e = payroll.add_employee(make_salaried("52000"))
    result = payroll.run_payroll(pay_period_index=1)
    assert result["payslips"][0]["history_id"] > 0
    assert len(payroll.get_pay_history(e.id)) == 1


def test_reset_store_clears_pay_history():
    """Reset wipes all pay history."""
    fresh_store()
    emp = payroll.add_employee(make_hourly("20"))
    payroll.record_payslip(generate_payslip(emp, hours_worked=Decimal("40")))
    payroll.reset_store()
    assert payroll.get_pay_history(emp.id) == []


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------

def setup_function(_function):
    """Give every test a brand-new temporary database."""
    fresh_store()


def test_add_get_delete_employee():
    """Employees can be added, fetched, and deleted."""
    added = payroll.add_employee(make_salaried())
    assert added.id > 0
    fetched = payroll.get_employee(added.id)
    assert fetched is not None
    assert fetched.name == "Alice"
    assert payroll.delete_employee(added.id) is True
    assert payroll.get_employee(added.id) is None
    assert payroll.delete_employee(added.id) is False


def test_decimal_precision_survives_roundtrip():
    """Decimal precision survives SQLite round-trip."""
    emp = make_salaried("52000.55", benefits="123.45")
    stored = payroll.add_employee(emp)
    fetched = payroll.get_employee(stored.id)
    assert fetched is not None
    assert fetched.annual_salary == Decimal("52000.55")
    assert fetched.benefits_deduction_per_period == Decimal("123.45")
    assert fetched.pay_periods_per_year == 26


def test_persistence_across_store_recreation(tmp_path):
    """Data survives reopening the same database."""
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

@lru_cache(maxsize=1)
def get_client():
    """Return a cached TestClient bound to the FastAPI app."""
    from app import app  # pylint: disable=import-outside-toplevel
    return TestClient(app)


def test_api_create_and_list_employees():
    """POST then GET lists the created employee."""
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
    """The payslip endpoint returns correct math."""
    c = get_client()
    created = c.post("/payroll/employees", json={
        "name": "Dave", "pay_type": "salaried",
        "annual_salary": 104000, "pay_periods_per_year": 52,
    }).json()

    resp = c.post(f"/payroll/employees/{created['id']}/payslip",
                  json={"hours_worked": None})
    assert resp.status_code == 200
    slip = resp.json()
    assert slip["gross_pay"] == 2000.00
    withheld = slip["tax_withheld"] + slip["social_security"] + slip["medicare"]
    assert abs(slip["net_pay"] + withheld - 2000.00) < 0.01
    assert "history_id" in slip


def test_api_payslip_unknown_employee_returns_404():
    """Payslip for unknown id returns 404."""
    c = get_client()
    resp = c.post("/payroll/employees/999999/payslip", json={})
    assert resp.status_code == 404


def test_api_invalid_employee_returns_400():
    """Blank names are rejected with 400."""
    c = get_client()
    resp = c.post("/payroll/employees", json={"name": "", "pay_type": "salaried"})
    assert resp.status_code == 400


def test_api_run_payroll_endpoint():
    """Run endpoint pays all employees correctly."""
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
    """Missing hourly hours surface as errors."""
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
    """Bogus filing status returns 400."""
    c = get_client()
    resp = c.post("/payroll/run", json={"filing_status": "bogus"})
    assert resp.status_code == 400


def test_api_run_payroll_unknown_hours_employee_returns_404():
    """Hours for unknown ids return 404."""
    c = get_client()
    resp = c.post("/payroll/run", json={"hours": {"999999": 10}})
    assert resp.status_code == 404


def test_api_create_employee_with_filing_status():
    """Filing status persists via the API."""
    c = get_client()
    resp = c.post("/payroll/employees", json={
        "name": "Married Mary", "pay_type": "salaried",
        "annual_salary": 104000, "filing_status": "married_joint",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["filing_status"] == "married_joint"
    # Persisted round-trip through SQLite
    refetched = c.get(f"/payroll/employees/{body['id']}").json()
    assert refetched["filing_status"] == "married_joint"


def test_api_create_employee_invalid_filing_status_returns_400():
    """Invalid filing status returns 400."""
    c = get_client()
    resp = c.post("/payroll/employees", json={
        "name": "Bad Status", "pay_type": "salaried",
        "annual_salary": 1000, "filing_status": "nope",
    })
    assert resp.status_code == 400


def test_api_payslip_history_endpoint():
    """History endpoint returns newest-first payslips."""
    c = get_client()
    e = c.post("/payroll/employees", json={
        "name": "History Hank", "pay_type": "salaried", "annual_salary": 52000,
    }).json()
    eid = e["id"]
    c.post(f"/payroll/employees/{eid}/payslip", json={"pay_period_index": 5})
    c.post(f"/payroll/employees/{eid}/payslip", json={"pay_period_index": 6})

    hist = c.get(f"/payroll/employees/{eid}/payslips").json()
    assert len(hist) == 2
    assert [row["pay_period_index"] for row in hist] == [6, 5]  # newest first
    for row in hist:
        assert row["gross_pay"] == 2000.00
        assert row["social_security"] > 0
        assert row["medicare"] > 0

    assert c.get("/payroll/employees/999999/payslips").status_code == 404


def test_api_update_employee():
    """PUT updates fields and future payslips reflect them."""
    c = get_client()
    e = c.post("/payroll/employees", json={
        "name": "Updatable Uma", "pay_type": "hourly", "hourly_rate": 25.0,
    }).json()

    resp = c.put(f"/payroll/employees/{e['id']}", json={
        "hourly_rate": 30, "state": "co",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["hourly_rate"] == 30.0
    assert body["state"] == "co"
    assert body["name"] == "Updatable Uma"  # untouched

    # New rate takes effect on the next payslip
    slip = c.post(f"/payroll/employees/{e['id']}/payslip",
                  json={"hours_worked": 40}).json()
    assert slip["gross_pay"] == 1200.00


def test_api_update_unknown_employee_returns_404():
    """Updating unknown id returns 404."""
    c = get_client()
    resp = c.put("/payroll/employees/999999", json={"name": "Ghost"})
    assert resp.status_code == 404


def test_api_update_invalid_field_returns_400():
    """Immutable fields are rejected."""
    c = get_client()
    e = c.post("/payroll/employees", json={
        "name": "X", "pay_type": "hourly", "hourly_rate": 10,
    }).json()
    resp = c.put(f"/payroll/employees/{e['id']}", json={"pay_type": "salaried"})
    # pay_type is immutable; forbidden by the model -> 422 validation error
    assert resp.status_code in (400, 422)


def test_api_liabilities_endpoint():
    """Liabilities endpoint reports FICA totals."""
    c = get_client()
    c.post("/payroll/employees", json={
        "name": "Liable Larry", "pay_type": "salaried", "annual_salary": 52000,
    })
    run = c.post("/payroll/run", json={}).json()
    report = c.get("/payroll/liabilities").json()

    assert report["payslips_recorded"] >= 1
    assert report["employer_taxes"]["social_security_match"] > 0
    # Employer SS matches employee SS below the wage base
    assert abs(report["employer_taxes"]["social_security_match"]
               - report["employee_withholdings"]["social_security"]) < 0.05
    expected_total = (report["employee_withholdings"]["social_security"]
                      + report["employer_taxes"]["social_security_match"]
                      + report["employee_withholdings"]["medicare"]
                      + report["employer_taxes"]["medicare"])
    assert abs(report["total_fica_liability"] - expected_total) < 0.05
    assert abs(report["gross_wages"] - run["totals"]["gross_pay"]) < 0.05


def test_api_payslip_pdf_download():
    """PDF download serves a valid PDF."""
    c = get_client()
    e = c.post("/payroll/employees", json={
        "name": "PDF Paula", "pay_type": "salaried", "annual_salary": 52000,
    }).json()
    eid = e["id"]
    slip = c.post(f"/payroll/employees/{eid}/payslip",
                  json={"pay_period_index": 7}).json()

    resp = c.get(f"/payroll/employees/{eid}/payslips/{slip['history_id']}/pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF")
    expected_pdf = f"payslip_{eid}_{slip['history_id']}.pdf"
    assert expected_pdf in resp.headers["content-disposition"]

    # Mismatched employee/payslip pair -> 404
    other = c.post("/payroll/employees", json={
        "name": "Other Otto", "pay_type": "hourly", "hourly_rate": 5,
    }).json()
    mismatch = c.get(
        f"/payroll/employees/{other['id']}"
        f"/payslips/{slip['history_id']}/pdf")
    assert mismatch.status_code == 404


def test_api_company_scoped_endpoints():
    """Company scoping works across endpoints."""
    c = get_client()
    acme_emp = c.post("/payroll/employees", json={
        "name": "Acme Annie", "pay_type": "salaried",
        "annual_salary": 52000, "company_id": "acme",
    }).json()
    c.post("/payroll/employees", json={
        "name": "Globex Gary", "pay_type": "salaried",
        "annual_salary": 52000, "company_id": "globex",
    }).json()

    listing = c.get("/payroll/employees", params={"company_id": "acme"}).json()
    assert len(listing) == 1
    assert listing[0]["id"] == acme_emp["id"]
    assert listing[0]["company_id"] == "acme"

    run = c.post("/payroll/run", json={"hours": {}, "company_id": "acme"}).json()
    assert run["employees_paid"] == 1

    report = c.get("/payroll/liabilities", params={"company_id": "acme"}).json()
    assert report["company_id"] == "acme"
    assert report["employer_taxes"]["suta"] >= 0  # no state -> suta 0, still reported


def test_api_liabilities_includes_unemployment():
    """Liability totals include unemployment taxes."""
    c = get_client()
    c.post("/payroll/employees", json={
        "name": "CO Carla", "pay_type": "salaried",
        "annual_salary": 52000, "state": "co",
    })
    c.post("/payroll/run", json={})
    report = c.get("/payroll/liabilities").json()
    assert report["employer_taxes"]["futa"] > 0
    assert report["employer_taxes"]["suta"] > 0
    expected = (report["total_fica_liability"]
                + report["employer_taxes"]["unemployment_total"])
    assert abs(report["total_employer_liability"] - expected) < 0.05


def test_api_401k_creation_update_and_ytd():
    """401(k) flows through create, slip, YTD, and update."""
    c = get_client()
    e = c.post("/payroll/employees", json={
        "name": "Saver Sam", "pay_type": "salaried",
        "annual_salary": 52000, "retirement_401k_percent": 10,
    }).json()
    assert e["retirement_401k_percent"] == 10.0

    slip = c.post(f"/payroll/employees/{e['id']}/payslip", json={}).json()
    assert slip["retirement_401k"] == 200.0

    ytd = c.get(f"/payroll/employees/{e['id']}/ytd").json()
    assert ytd["pay_periods_paid"] == 1
    assert ytd["gross_wages"] == 2000.0
    assert ytd["retirement_401k"] == 200.0
    assert abs(ytd["taxable_wages_income_tax"] - 1800.0) < 0.01

    # Update the contribution rate via PUT
    upd = c.put(f"/payroll/employees/{e['id']}",
                json={"retirement_401k_percent": 5}).json()
    assert upd["retirement_401k_percent"] == 5.0


def test_api_ytd_unknown_employee_returns_404():
    """YTD for unknown id returns 404."""
    c = get_client()
    resp = c.get("/payroll/employees/999999/ytd")
    assert resp.status_code == 404


def test_dashboard_endpoint():
    """Dashboard renders HTML containing the app title."""
    c = get_client()
    resp = c.get("/dashboard")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Coetus Payroll" in resp.text


def test_api_csv_export_endpoint():
    """Export endpoint streams CSV with header and rows."""
    c = get_client()
    e = c.post("/payroll/employees", json={
        "name": "CSV Celia", "pay_type": "hourly", "hourly_rate": 20,
    }).json()
    c.post(f"/payroll/employees/{e['id']}/payslip", json={"hours_worked": 40})

    resp = c.get("/payroll/export.csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    lines = resp.text.strip().splitlines()
    assert lines[0].startswith("history_id,employee_id")
    assert len(lines) == 2
    assert "CSV Celia" in lines[1]

    # Empty scope yields header only
    empty = c.get("/payroll/export.csv", params={"employee_id": 999999})
    assert len(empty.text.strip().splitlines()) == 1
