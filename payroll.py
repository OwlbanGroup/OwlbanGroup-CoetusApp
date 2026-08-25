"""
Payroll module for OwlbanGroup-CoetusApp.

Provides employee record management and payroll calculation:
gross pay (salaried or hourly with overtime), progressive tax
withholding, benefit deductions, and net pay.
"""

import os
import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional


TWO_PLACES = Decimal("0.01")
OVERTIME_THRESHOLD_HOURS = Decimal("40")
OVERTIME_MULTIPLIER = Decimal("1.5")

VALID_PAY_TYPES = ("salaried", "hourly")


@dataclass
class Employee:
    """An employee payroll record."""
    id: int = 0
    name: str = ""
    pay_type: str = "salaried"               # "salaried" or "hourly"
    annual_salary: Optional[Decimal] = None   # required for salaried
    hourly_rate: Optional[Decimal] = None     # required for hourly
    benefits_deduction_per_period: Decimal = field(default_factory=lambda: Decimal("0"))
    pay_periods_per_year: int = 26            # bi-weekly default

    def validate(self):
        if self.pay_type not in VALID_PAY_TYPES:
            raise ValueError(f"pay_type must be one of {VALID_PAY_TYPES}")
        if not self.name or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if self.pay_periods_per_year < 1:
            raise ValueError("pay_periods_per_year must be >= 1")
        if self.benefits_deduction_per_period < 0:
            raise ValueError("benefits_deduction_per_period cannot be negative")
        if self.pay_type == "salaried":
            if self.annual_salary is None or self.annual_salary < 0:
                raise ValueError("salaried employees require a non-negative annual_salary")
        else:
            if self.hourly_rate is None or self.hourly_rate < 0:
                raise ValueError("hourly employees require a non-negative hourly_rate")


# 2024 US Federal income tax brackets (marginal rates on annual taxable income).
# Source: IRS Rev. Proc. 2023-34. These are illustrative for payroll estimation
# and do not include FICA, state/local taxes, or standard-deduction effects.
BRACKET_PRESETS = {
    "single": [
        (Decimal("11600"), Decimal("0.10")),
        (Decimal("47150"), Decimal("0.12")),
        (Decimal("100525"), Decimal("0.22")),
        (Decimal("191950"), Decimal("0.24")),
        (Decimal("243725"), Decimal("0.32")),
        (Decimal("609350"), Decimal("0.35")),
        (None, Decimal("0.37")),
    ],
    "married_joint": [
        (Decimal("23200"), Decimal("0.10")),
        (Decimal("94300"), Decimal("0.12")),
        (Decimal("201050"), Decimal("0.22")),
        (Decimal("383900"), Decimal("0.24")),
        (Decimal("487450"), Decimal("0.32")),
        (Decimal("731200"), Decimal("0.35")),
        (None, Decimal("0.37")),
    ],
    "head_of_household": [
        (Decimal("16550"), Decimal("0.10")),
        (Decimal("63100"), Decimal("0.12")),
        (Decimal("100500"), Decimal("0.22")),
        (Decimal("211400"), Decimal("0.24")),
        (Decimal("267700"), Decimal("0.32")),
        (Decimal("630050"), Decimal("0.35")),
        (None, Decimal("0.37")),
    ],
}

DEFAULT_TAX_BRACKETS = BRACKET_PRESETS["single"]

_tax_brackets: List[tuple] = list(DEFAULT_TAX_BRACKETS)


# ---------------------------------------------------------------------------
# SQLite-backed employee store
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = os.environ.get("PAYROLL_DB_PATH", "payroll.db")
_db_path: str = DEFAULT_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    pay_type TEXT NOT NULL,
    annual_salary TEXT,
    hourly_rate TEXT,
    benefits_deduction_per_period TEXT NOT NULL DEFAULT '0',
    pay_periods_per_year INTEGER NOT NULL DEFAULT 26
)
"""


def _connect() -> sqlite3.Connection:
    """Open a connection to the payroll database, creating the schema."""
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def configure_store(path: str):
    """Point the payroll store at a different SQLite database file."""
    global _db_path
    _db_path = path
    _connect().close()


def set_tax_brackets(brackets: List[tuple]):
    """Replace the tax brackets. Each entry is (upper_bound_or_None, rate)."""
    global _tax_brackets
    _tax_brackets = list(brackets)


def set_tax_brackets_by_status(filing_status: str):
    """
    Select a preset tax table by filing status.
    One of: "single", "married_joint", "head_of_household".
    """
    if filing_status not in BRACKET_PRESETS:
        raise ValueError(f"filing_status must be one of {sorted(BRACKET_PRESETS)}")
    set_tax_brackets(BRACKET_PRESETS[filing_status])


def _row_to_employee(row: sqlite3.Row) -> Employee:
    return Employee(
        id=row["id"],
        name=row["name"],
        pay_type=row["pay_type"],
        annual_salary=Decimal(row["annual_salary"]) if row["annual_salary"] is not None else None,
        hourly_rate=Decimal(row["hourly_rate"]) if row["hourly_rate"] is not None else None,
        benefits_deduction_per_period=Decimal(row["benefits_deduction_per_period"] or "0"),
        pay_periods_per_year=row["pay_periods_per_year"],
    )


def get_tax_brackets() -> List[tuple]:
    return list(_tax_brackets)


# ---------------------------------------------------------------------------
# Employee store operations
# ---------------------------------------------------------------------------

def add_employee(employee: Employee) -> Employee:
    """Persist an employee and assign an ID. Returns the stored employee."""
    employee.validate()
    conn = _connect()
    try:
        cursor = conn.execute(
            "INSERT INTO employees (name, pay_type, annual_salary, hourly_rate,"
            " benefits_deduction_per_period, pay_periods_per_year)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                employee.name.strip(),
                employee.pay_type,
                str(employee.annual_salary) if employee.annual_salary is not None else None,
                str(employee.hourly_rate) if employee.hourly_rate is not None else None,
                str(employee.benefits_deduction_per_period),
                employee.pay_periods_per_year,
            ),
        )
        conn.commit()
        employee.id = cursor.lastrowid
        return employee
    finally:
        conn.close()


def get_employee(employee_id: int) -> Optional[Employee]:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
        return _row_to_employee(row) if row is not None else None
    finally:
        conn.close()


def list_employees() -> List[Employee]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM employees ORDER BY id").fetchall()
        return [_row_to_employee(row) for row in rows]
    finally:
        conn.close()


def delete_employee(employee_id: int) -> bool:
    conn = _connect()
    try:
        cursor = conn.execute("DELETE FROM employees WHERE id = ?", (employee_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def reset_store():
    """Delete all employees and reset IDs (useful for testing)."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM employees")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'employees'")
        conn.commit()
    finally:
        conn.close()



# ---------------------------------------------------------------------------
# Pay calculations
# ---------------------------------------------------------------------------

def calculate_gross_pay(employee: Employee, hours_worked: Optional[Decimal] = None,
                        pay_period_index: int = 0) -> Decimal:
    """
    Calculate gross pay for one pay period.

    - Salaried: annual_salary / pay_periods_per_year
    - Hourly: hourly_rate * hours_worked, with hours above
      OVERTIME_THRESHOLD_HOURS paid at OVERTIME_MULTIPLIER
    """
    if employee.pay_type == "salaried":
        gross = employee.annual_salary / Decimal(employee.pay_periods_per_year)
    else:
        if hours_worked is None or hours_worked < 0:
            raise ValueError("hours_worked is required and must be >= 0 for hourly employees")
        regular = min(hours_worked, OVERTIME_THRESHOLD_HOURS)
        overtime = max(hours_worked - OVERTIME_THRESHOLD_HOURS, Decimal("0"))
        gross = (employee.hourly_rate * regular) + \
                (employee.hourly_rate * OVERTIME_MULTIPLIER * overtime)
    return gross.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def calculate_tax(gross_annualized: Decimal,
                  brackets: Optional[List[tuple]] = None) -> Decimal:
    """
    Compute tax owed on an annualized amount using progressive brackets.
    Falls back to the module-level active brackets when not provided.
    """
    if brackets is None:
        brackets = _tax_brackets
    remaining = gross_annualized
    previous_bound = Decimal("0")
    tax = Decimal("0")
    for upper_bound, rate in brackets:
        bracket_width = (upper_bound - previous_bound) if upper_bound is not None else remaining
        taxable_in_bracket = min(remaining, bracket_width)
        if taxable_in_bracket <= 0:
            break
        tax += taxable_in_bracket * rate
        remaining -= taxable_in_bracket
        previous_bound = upper_bound if upper_bound is not None else previous_bound
        if remaining <= 0:
            break
    return tax.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def generate_payslip(employee: Employee, hours_worked: Optional[Decimal] = None,
                     pay_period_index: int = 0,
                     tax_brackets: Optional[List[tuple]] = None) -> dict:
    """
    Generate a full payslip for one pay period including tax withholding
    and benefit deductions. Optionally pass explicit tax_brackets to
    override the module-level active table.
    """
    gross = calculate_gross_pay(employee, hours_worked=hours_worked,
                                pay_period_index=pay_period_index)
    annual_gross = gross * Decimal(employee.pay_periods_per_year)
    annual_tax = calculate_tax(annual_gross, brackets=tax_brackets)
    tax_withheld = (annual_tax / Decimal(employee.pay_periods_per_year)).quantize(
        TWO_PLACES, rounding=ROUND_HALF_UP)
    benefits = employee.benefits_deduction_per_period.quantize(TWO_PLACES)
    net = (gross - tax_withheld - benefits).quantize(TWO_PLACES)

    return {
        "employee_id": employee.id,
        "employee_name": employee.name,
        "pay_type": employee.pay_type,
        "hours_worked": float(hours_worked) if hours_worked is not None else None,
        "gross_pay": float(gross),
        "tax_withheld": float(tax_withheld),
        "benefits_deduction": float(benefits),
        "net_pay": float(net),
    }


def run_payroll(hours_by_employee: Optional[Dict[int, Decimal]] = None,
                tax_brackets: Optional[List[tuple]] = None) -> dict:
    """
    Run payroll for all employees for one pay period.

    - hours_by_employee: mapping of employee_id -> hours worked. Entries are
      required for hourly employees; missing ones are reported as errors.
    - tax_brackets: optional explicit table overriding the active one.

    Returns per-employee payslips, per-employee errors, period totals,
    and counts.
    """
    hours_by_employee = hours_by_employee or {}
    slips = []
    errors = []
    total_gross = Decimal("0")
    total_tax = Decimal("0")
    total_benefits = Decimal("0")
    total_net = Decimal("0")

    for employee in list_employees():
        try:
            slip = generate_payslip(employee,
                                    hours_worked=hours_by_employee.get(employee.id),
                                    tax_brackets=tax_brackets)
            slips.append(slip)
            total_gross += Decimal(str(slip["gross_pay"]))
            total_tax += Decimal(str(slip["tax_withheld"]))
            total_benefits += Decimal(str(slip["benefits_deduction"]))
            total_net += Decimal(str(slip["net_pay"]))
        except ValueError as e:
            errors.append({
                "employee_id": employee.id,
                "employee_name": employee.name,
                "error": str(e),
            })

    def _f(value):
        return float(value.quantize(TWO_PLACES, rounding=ROUND_HALF_UP))

    return {
        "payslips": slips,
        "errors": errors,
        "totals": {
            "gross_pay": _f(total_gross),
            "tax_withheld": _f(total_tax),
            "benefits_deduction": _f(total_benefits),
            "net_pay": _f(total_net),
        },
        "employees_paid": len(slips),
        "employees_errored": len(errors),
    }

