"""
Payroll module for OwlbanGroup-CoetusApp.

Provides employee record management and payroll calculation:
gross pay (salaried or hourly with overtime), progressive tax
withholding, benefit deductions, and net pay.
"""

import csv
import io
import os
import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Union


TWO_PLACES = Decimal("0.01")
OVERTIME_THRESHOLD_HOURS = Decimal("40")
OVERTIME_MULTIPLIER = Decimal("1.5")

# 2024 FICA rates (employee side)
SOCIAL_SECURITY_RATE = Decimal("0.062")
SOCIAL_SECURITY_WAGE_BASE = Decimal("168600")   # annual cap for SS tax
MEDICARE_RATE = Decimal("0.0145")
ADDITIONAL_MEDICARE_RATE = Decimal("0.009")     # surtax on wages above threshold
ADDITIONAL_MEDICARE_THRESHOLD = Decimal("200000")

# Illustrative state income tax presets (2024 headline rates).
# A state's value is either a flat Decimal rate or a progressive bracket
# list [(upper_bound_or_None, marginal_rate)] like the federal tables.
# Employees with state=None pay no state tax through this module.
STATE_TAX_RATES: Dict[str, Union[Decimal, List[tuple]]] = {
    "none": Decimal("0"),
    "az": Decimal("0.025"),    # Arizona (flat)
    "co": Decimal("0.044"),    # Colorado (flat)
    "il": Decimal("0.0495"),   # Illinois (flat)
    "in": Decimal("0.030"),    # Indiana (flat)
    "nc": Decimal("0.0425"),   # North Carolina (flat)
    "pa": Decimal("0.0307"),   # Pennsylvania (flat)
    # California 2024 single-filer progressive schedule (simplified)
    "ca": [
        (Decimal("10756"), Decimal("0.01")),
        (Decimal("25499"), Decimal("0.02")),
        (Decimal("40245"), Decimal("0.04")),
        (Decimal("55866"), Decimal("0.06")),
        (Decimal("70606"), Decimal("0.08")),
        (Decimal("360659"), Decimal("0.093")),
        (Decimal("432787"), Decimal("0.103")),
        (Decimal("721314"), Decimal("0.113")),
        (None, Decimal("0.123")),
    ],
    # New York 2024 single-filer progressive schedule (simplified)
    "ny": [
        (Decimal("8500"), Decimal("0.04")),
        (Decimal("11700"), Decimal("0.045")),
        (Decimal("13900"), Decimal("0.0525")),
        (Decimal("80650"), Decimal("0.055")),
        (Decimal("215400"), Decimal("0.06")),
        (Decimal("1077550"), Decimal("0.0685")),
        (Decimal("5000000"), Decimal("0.0965")),
        (None, Decimal("0.109")),
    ],
}

# Employer unemployment insurance (2024 figures, illustrative)
FUTA_RATE = Decimal("0.006")              # 6.0% minus max 5.4% state credit
FUTA_WAGE_BASE = Decimal("7000")          # per employee, per year
DEFAULT_SUTA_RATE = Decimal("0.027")      # configurable via set_suta_rate()
SUTA_WAGE_BASES = {                       # sample taxable wage bases per state
    "none": Decimal("0"),
    "az": Decimal("7000"),
    "ca": Decimal("7000"),
    "co": Decimal("17000"),
    "il": Decimal("13270"),
    "in": Decimal("9500"),
    "nc": Decimal("28000"),
    "ny": Decimal("11900"),
    "pa": Decimal("10000"),
}
_suta_rate: Decimal = DEFAULT_SUTA_RATE

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
    # BRACKET_PRESETS key, or None to use the active table
    filing_status: Optional[str] = "single"
    # STATE_TAX_RATES key, or None for no state tax
    state: Optional[str] = None
    company_id: Optional[str] = None          # optional company/grouping tag
    retirement_401k_percent: Decimal = field(default_factory=lambda: Decimal("0"))

    def validate(self):
        """Raise ValueError when any field violates the model's rules."""
        self._validate_common()
        self._validate_pay_amounts()

    def _validate_common(self):
        """Validate fields shared by every pay type."""
        if self.pay_type not in VALID_PAY_TYPES:
            raise ValueError(f"pay_type must be one of {VALID_PAY_TYPES}")
        if not self.name or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if self.pay_periods_per_year < 1:
            raise ValueError("pay_periods_per_year must be >= 1")
        if self.benefits_deduction_per_period < 0:
            raise ValueError("benefits_deduction_per_period cannot be negative")
        if not Decimal("0") <= self.retirement_401k_percent <= Decimal("100"):
            raise ValueError("retirement_401k_percent must be between 0 and 100")
        if self.filing_status is not None and self.filing_status not in BRACKET_PRESETS:
            raise ValueError(
                f"filing_status must be one of {sorted(BRACKET_PRESETS)} or null")
        if self.state is not None and self.state not in STATE_TAX_RATES:
            raise ValueError(f"state must be one of {sorted(STATE_TAX_RATES)} or null")
        if self.company_id is not None and (not isinstance(self.company_id, str)
                                            or not self.company_id.strip()):
            raise ValueError("company_id must be a non-empty string or null")

    def _validate_pay_amounts(self):
        """Ensure the pay amount required by pay_type is present and valid."""
        if self.pay_type == "salaried":
            if self.annual_salary is None or self.annual_salary < 0:
                raise ValueError(
                    "salaried employees require a non-negative annual_salary")
        elif self.hourly_rate is None or self.hourly_rate < 0:
            raise ValueError(
                "hourly employees require a non-negative hourly_rate")


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
    pay_periods_per_year INTEGER NOT NULL DEFAULT 26,
    filing_status TEXT DEFAULT 'single',
    state TEXT,
    company_id TEXT,
    retirement_401k_percent TEXT NOT NULL DEFAULT '0'
)
"""

_PAY_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS pay_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL,
    pay_period_index INTEGER NOT NULL DEFAULT 0,
    hours_worked TEXT,
    gross_pay TEXT NOT NULL,
    federal_tax_withheld TEXT NOT NULL,
    social_security_withheld TEXT NOT NULL,
    medicare_withheld TEXT NOT NULL,
    benefits_deduction TEXT NOT NULL,
    net_pay TEXT NOT NULL,
    filing_status TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    state_tax_withheld TEXT NOT NULL DEFAULT '0',
    employer_social_security TEXT NOT NULL DEFAULT '0',
    employer_medicare TEXT NOT NULL DEFAULT '0',
    employer_futa TEXT NOT NULL DEFAULT '0',
    employer_suta TEXT NOT NULL DEFAULT '0',
    company_id TEXT,
    retirement_401k TEXT NOT NULL DEFAULT '0'
)
"""


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str):
    """Add a column to an existing table if it is missing (simple migration)."""
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def _connect() -> sqlite3.Connection:
    """Open a connection to the payroll database, creating/migrating the schema."""
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    conn.execute(_PAY_HISTORY_SCHEMA)
    # Migrate databases created before these columns existed.
    _ensure_column(conn, "employees", "filing_status",
                   "filing_status TEXT DEFAULT 'single'")
    _ensure_column(conn, "employees", "state", "state TEXT")
    _ensure_column(conn, "employees", "company_id", "company_id TEXT")
    _ensure_column(conn, "employees", "retirement_401k_percent",
                   "retirement_401k_percent TEXT NOT NULL DEFAULT '0'")
    _ensure_column(conn, "pay_history", "state_tax_withheld",
                   "state_tax_withheld TEXT NOT NULL DEFAULT '0'")
    _ensure_column(conn, "pay_history", "employer_social_security",
                   "employer_social_security TEXT NOT NULL DEFAULT '0'")
    _ensure_column(conn, "pay_history", "employer_medicare",
                   "employer_medicare TEXT NOT NULL DEFAULT '0'")
    _ensure_column(conn, "pay_history", "employer_futa",
                   "employer_futa TEXT NOT NULL DEFAULT '0'")
    _ensure_column(conn, "pay_history", "employer_suta",
                   "employer_suta TEXT NOT NULL DEFAULT '0'")
    _ensure_column(conn, "pay_history", "company_id", "company_id TEXT")
    _ensure_column(conn, "pay_history", "retirement_401k",
                   "retirement_401k TEXT NOT NULL DEFAULT '0'")
    conn.commit()
    return conn


def configure_store(path: str):
    """Point the payroll store at a different SQLite database file."""
    global _db_path  # pylint: disable=global-statement  # config singleton
    _db_path = path
    _connect().close()


def set_tax_brackets(brackets: List[tuple]):
    """Replace the tax brackets. Each entry is (upper_bound_or_None, rate)."""
    global _tax_brackets  # pylint: disable=global-statement  # config singleton
    _tax_brackets = list(brackets)


def set_tax_brackets_by_status(filing_status: str):
    """
    Select a preset tax table by filing status.
    One of: "single", "married_joint", "head_of_household".
    """
    if filing_status not in BRACKET_PRESETS:
        raise ValueError(f"filing_status must be one of {sorted(BRACKET_PRESETS)}")
    set_tax_brackets(BRACKET_PRESETS[filing_status])


def set_suta_rate(rate: Decimal):
    """
    Set the employer's SUTA tax rate (e.g. Decimal("0.031") for 3.1%).
    Rates are experience-rated per employer; this module uses one global rate.
    """
    global _suta_rate  # pylint: disable=global-statement  # config singleton
    if rate < 0 or rate > 1:
        raise ValueError("SUTA rate must be between 0 and 1")
    _suta_rate = rate


def _row_to_employee(row: sqlite3.Row) -> Employee:
    """Build an Employee from an employees-table row."""
    return Employee(
        id=row["id"],
        name=row["name"],
        pay_type=row["pay_type"],
        annual_salary=(Decimal(row["annual_salary"])
                       if row["annual_salary"] is not None else None),
        hourly_rate=(Decimal(row["hourly_rate"])
                     if row["hourly_rate"] is not None else None),
        benefits_deduction_per_period=Decimal(
            row["benefits_deduction_per_period"] or "0"),
        pay_periods_per_year=row["pay_periods_per_year"],
        filing_status=(row["filing_status"]
                       if "filing_status" in row.keys() else "single"),
        state=row["state"] if "state" in row.keys() else None,
        company_id=row["company_id"] if "company_id" in row.keys() else None,
        retirement_401k_percent=(
            Decimal(row["retirement_401k_percent"])
            if ("retirement_401k_percent" in row.keys()
                and row["retirement_401k_percent"])
            else Decimal("0")
        ),
    )


def get_tax_brackets() -> List[tuple]:
    """Return a copy of the currently active federal bracket table."""
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
            " benefits_deduction_per_period, pay_periods_per_year, filing_status,"
            " state, company_id, retirement_401k_percent)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                employee.name.strip(),
                employee.pay_type,
                (str(employee.annual_salary)
                 if employee.annual_salary is not None else None),
                str(employee.hourly_rate) if employee.hourly_rate is not None else None,
                str(employee.benefits_deduction_per_period),
                employee.pay_periods_per_year,
                employee.filing_status,
                employee.state,
                employee.company_id.strip() if employee.company_id else None,
                str(employee.retirement_401k_percent),
            ),
        )
        conn.commit()
        if cursor.lastrowid is None:  # sqlite always sets it after INSERT
            raise RuntimeError("INSERT did not return a row id")
        employee.id = cursor.lastrowid
        return employee
    finally:
        conn.close()


def get_employee(employee_id: int) -> Optional[Employee]:
    """Fetch one employee by id, or None when it does not exist."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM employees WHERE id = ?", (employee_id,)
        ).fetchone()
        return _row_to_employee(row) if row is not None else None
    finally:
        conn.close()


def list_employees(company_id: Optional[str] = None) -> List[Employee]:
    """List employees, optionally filtered by company_id."""
    conn = _connect()
    try:
        if company_id is None:
            rows = conn.execute("SELECT * FROM employees ORDER BY id").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM employees WHERE company_id = ? ORDER BY id",
                (company_id,),
            ).fetchall()
        return [_row_to_employee(row) for row in rows]
    finally:
        conn.close()


def delete_employee(employee_id: int) -> bool:
    """Delete an employee; True when a row was removed."""
    conn = _connect()
    try:
        cursor = conn.execute("DELETE FROM employees WHERE id = ?", (employee_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


UPDATABLE_FIELDS = {
    "name", "annual_salary", "hourly_rate", "benefits_deduction_per_period",
    "pay_periods_per_year", "filing_status", "state", "company_id",
    "retirement_401k_percent",
}


def update_employee(employee_id: int, updates: dict) -> Optional[Employee]:
    """
    Apply a partial update to an employee's record and return the updated
    employee, or None if the ID does not exist. Raises ValueError for
    invalid field names or values. `pay_type` is intentionally immutable.
    """
    existing = get_employee(employee_id)
    if existing is None:
        return None

    unknown = set(updates) - UPDATABLE_FIELDS
    if unknown:
        raise ValueError(f"cannot update fields: {sorted(unknown)}")

    for key, value in updates.items():
        if key in ("name",):
            setattr(existing, key, value)
        elif key in ("pay_periods_per_year",):
            setattr(existing, key, int(value))
        elif key in ("filing_status", "state", "company_id"):
            setattr(existing, key, value)  # validated below
        elif key == "retirement_401k_percent":
            setattr(existing, key, Decimal(str(value)))
        else:
            setattr(existing, key,
                    Decimal(str(value)) if value is not None else None)
    existing.validate()

    conn = _connect()
    try:
        conn.execute(
            "UPDATE employees SET name = ?, annual_salary = ?, hourly_rate = ?,"
            " benefits_deduction_per_period = ?, pay_periods_per_year = ?,"
            " filing_status = ?, state = ?, company_id = ?,"
            " retirement_401k_percent = ? WHERE id = ?",
            (
                existing.name.strip(),
                (str(existing.annual_salary)
                 if existing.annual_salary is not None else None),
                str(existing.hourly_rate) if existing.hourly_rate is not None else None,
                str(existing.benefits_deduction_per_period),
                existing.pay_periods_per_year,
                existing.filing_status,
                existing.state,
                existing.company_id.strip() if existing.company_id else None,
                str(existing.retirement_401k_percent),
                employee_id,
            ),
        )
        conn.commit()
        return existing
    finally:
        conn.close()


def reset_store():
    """Delete all employees and pay history, resetting IDs (useful for testing)."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM employees")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'employees'")
        conn.execute("DELETE FROM pay_history")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'pay_history'")
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

    `pay_period_index` is reserved for future period-specific pay schedules.
    """
    if employee.pay_type == "salaried":
        if employee.annual_salary is None:
            raise ValueError("salaried employees require an annual_salary")
        gross = employee.annual_salary / Decimal(employee.pay_periods_per_year)
    else:
        if hours_worked is None or hours_worked < 0:
            raise ValueError(
                "hours_worked is required and must be >= 0 for hourly employees")
        if employee.hourly_rate is None:
            raise ValueError("hourly employees require an hourly_rate")
        rate = employee.hourly_rate
        regular = min(hours_worked, OVERTIME_THRESHOLD_HOURS)
        overtime = max(hours_worked - OVERTIME_THRESHOLD_HOURS, Decimal("0"))
        gross = (rate * regular) + (rate * OVERTIME_MULTIPLIER * overtime)
    return gross.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def _tax_from_brackets(amount: Decimal, brackets: List[tuple]) -> Decimal:
    """Apply progressive marginal brackets to an amount. Shared by federal and state."""
    remaining = amount
    previous_bound = Decimal("0")
    tax = Decimal("0")
    for upper_bound, rate in brackets:
        bracket_width = ((upper_bound - previous_bound)
                         if upper_bound is not None else remaining)
        taxable_in_bracket = min(remaining, bracket_width)
        if taxable_in_bracket <= 0:
            break
        tax += taxable_in_bracket * rate
        remaining -= taxable_in_bracket
        previous_bound = upper_bound if upper_bound is not None else previous_bound
        if remaining <= 0:
            break
    return tax.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def calculate_tax(gross_annualized: Decimal,
                  brackets: Optional[List[tuple]] = None) -> Decimal:
    """
    Compute tax owed on an annualized amount using progressive brackets.
    Falls back to the module-level active brackets when not provided.
    """
    if brackets is None:
        brackets = _tax_brackets
    return _tax_from_brackets(gross_annualized, brackets)


def calculate_fica(gross_annualized: Decimal) -> tuple:
    """
    Compute annual employee-side FICA on an annualized gross amount.
    Returns (social_security, medicare_including_additional).
    """
    social_security = (min(gross_annualized, SOCIAL_SECURITY_WAGE_BASE)
                       * SOCIAL_SECURITY_RATE)
    medicare = gross_annualized * MEDICARE_RATE
    additional = max(gross_annualized - ADDITIONAL_MEDICARE_THRESHOLD,
                     Decimal("0")) * ADDITIONAL_MEDICARE_RATE
    return social_security, medicare + additional


def calculate_state_tax(gross_annualized: Decimal, state: Optional[str]) -> Decimal:
    """
    Compute annual state income tax for `state`. A state preset may be a
    flat Decimal rate or a progressive bracket list. Returns 0 when the
    state is None or "none".
    """
    if state is None:
        return Decimal("0")
    preset = STATE_TAX_RATES.get(state)
    if preset is None:
        raise ValueError(f"state must be one of {sorted(STATE_TAX_RATES)}")
    if isinstance(preset, list):
        return _tax_from_brackets(gross_annualized, preset)
    return (gross_annualized * preset).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def compute_employer_taxes(gross_annualized: Decimal) -> tuple:
    """
    Compute annual employer-side payroll taxes (matching FICA).
    The employer matches Social Security (same wage base) and Medicare,
    but does NOT pay the employee-only Additional Medicare surtax.
    Returns (employer_social_security, employer_medicare).
    """
    employer_ss = (min(gross_annualized, SOCIAL_SECURITY_WAGE_BASE)
                   * SOCIAL_SECURITY_RATE)
    employer_medicare = gross_annualized * MEDICARE_RATE
    return employer_ss, employer_medicare


def compute_employer_unemployment(gross_annualized: Decimal,
                                  state: Optional[str]) -> tuple:
    """
    Compute annual employer unemployment insurance on an annualized gross.
    - FUTA: effective 0.6% on the first $7,000 per employee per year
      (6.0% statutory minus the maximum 5.4% state credit).
    - SUTA: `_suta_rate` applied to the state's taxable wage base
      (no SUTA when state is None or "none").
    Returns (futa, suta).
    """
    futa = min(gross_annualized, FUTA_WAGE_BASE) * FUTA_RATE
    base = SUTA_WAGE_BASES.get(state, Decimal("0")) if state else Decimal("0")
    suta = min(gross_annualized, base) * _suta_rate
    return futa, suta


def _resolve_brackets(employee: Employee,
                      tax_brackets: Optional[List[tuple]]) -> Optional[List[tuple]]:
    """Explicit argument > employee's filing status preset > active module table."""
    if tax_brackets is not None:
        return tax_brackets
    if employee.filing_status:
        return BRACKET_PRESETS[employee.filing_status]
    return None  # fall back to the module-level active table in calculate_tax()


def generate_payslip(employee: Employee, hours_worked: Optional[Decimal] = None,
                     pay_period_index: int = 0,
                     tax_brackets: Optional[List[tuple]] = None) -> dict:
    """
    Generate a full payslip for one pay period including federal income tax
    withholding (per the employee's filing status unless overridden), FICA,
    and benefit deductions. Does not persist; use record_payslip() for that.
    """
    periods = Decimal(employee.pay_periods_per_year)
    gross = calculate_gross_pay(employee, hours_worked=hours_worked,
                                pay_period_index=pay_period_index)
    annual_gross = gross * periods

    # Pre-tax 401(k): reduces income-tax wages (federal + state) but is
    # still subject to FICA. Deducted from the employee's net pay.
    retirement = (gross * employee.retirement_401k_percent
                  / Decimal("100")).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
    annual_retirement = retirement * periods
    taxable_annual = annual_gross - annual_retirement

    annual_tax = calculate_tax(
        taxable_annual, brackets=_resolve_brackets(employee, tax_brackets))
    tax_withheld = (annual_tax / periods).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

    annual_ss, annual_medicare = calculate_fica(annual_gross)
    social_security = (annual_ss / periods).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
    medicare = (annual_medicare / periods).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

    annual_state_tax = calculate_state_tax(taxable_annual, employee.state)
    state_withheld = (annual_state_tax / periods).quantize(TWO_PLACES,
                                                           rounding=ROUND_HALF_UP)

    benefits = employee.benefits_deduction_per_period.quantize(TWO_PLACES)
    total_deductions = (tax_withheld + social_security + medicare
                        + state_withheld + retirement + benefits).quantize(TWO_PLACES)
    net = (gross - total_deductions).quantize(TWO_PLACES)

    employer_ss_annual, employer_medi_annual = compute_employer_taxes(annual_gross)
    employer_ss = (employer_ss_annual / periods).quantize(TWO_PLACES,
                                                          rounding=ROUND_HALF_UP)
    employer_medicare = (employer_medi_annual / periods).quantize(
        TWO_PLACES, rounding=ROUND_HALF_UP)

    futa_annual, suta_annual = compute_employer_unemployment(annual_gross,
                                                             employee.state)
    employer_futa = (futa_annual / periods).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
    employer_suta = (suta_annual / periods).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

    return {
        "employee_id": employee.id,
        "employee_name": employee.name,
        "pay_type": employee.pay_type,
        "filing_status": employee.filing_status,
        "state": employee.state,
        "company_id": employee.company_id,
        "hours_worked": float(hours_worked) if hours_worked is not None else None,
        "gross_pay": float(gross),
        "tax_withheld": float(tax_withheld),
        "social_security": float(social_security),
        "medicare": float(medicare),
        "state_tax_withheld": float(state_withheld),
        "retirement_401k": float(retirement),
        "benefits_deduction": float(benefits),
        "total_deductions": float(total_deductions),
        "net_pay": float(net),
        "employer_social_security": float(employer_ss),
        "employer_medicare": float(employer_medicare),
        "employer_futa": float(employer_futa),
        "employer_suta": float(employer_suta),
    }


def record_payslip(slip: dict, pay_period_index: int = 0) -> int:
    """Persist a generated payslip into the pay_history table. Returns row id."""
    conn = _connect()
    try:
        cursor = conn.execute(
            "INSERT INTO pay_history (employee_id, pay_period_index, hours_worked,"
            " gross_pay, federal_tax_withheld, social_security_withheld,"
            " medicare_withheld, benefits_deduction, net_pay, filing_status,"
            " state_tax_withheld, employer_social_security, employer_medicare,"
            " employer_futa, employer_suta, company_id, retirement_401k)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                slip["employee_id"],
                pay_period_index,
                str(slip["hours_worked"])
                if slip.get("hours_worked") is not None else None,
                str(slip["gross_pay"]),
                str(slip["tax_withheld"]),
                str(slip["social_security"]),
                str(slip["medicare"]),
                str(slip["benefits_deduction"]),
                str(slip["net_pay"]),
                slip.get("filing_status"),
                str(slip.get("state_tax_withheld", 0)),
                str(slip.get("employer_social_security", 0)),
                str(slip.get("employer_medicare", 0)),
                str(slip.get("employer_futa", 0)),
                str(slip.get("employer_suta", 0)),
                slip.get("company_id"),
                str(slip.get("retirement_401k", 0)),
            ),
        )
        conn.commit()
        if cursor.lastrowid is None:  # sqlite always sets it after INSERT
            raise RuntimeError("INSERT did not return a row id")
        return cursor.lastrowid
    finally:
        conn.close()


def get_payslip_record(history_id: int) -> Optional[dict]:
    """Fetch one recorded payslip by its history row id."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM pay_history WHERE id = ?", (history_id,)
        ).fetchall()
        return _history_row_to_dict(rows[0]) if rows else None
    finally:
        conn.close()


def get_pay_history(employee_id: int) -> List[dict]:
    """Return all recorded payslips for an employee, newest first."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM pay_history WHERE employee_id = ? ORDER BY id DESC",
            (employee_id,),
        ).fetchall()
        return [_history_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def _history_row_to_dict(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "employee_id": r["employee_id"],
        "pay_period_index": r["pay_period_index"],
        "hours_worked": float(r["hours_worked"]) if r["hours_worked"] else None,
        "gross_pay": float(r["gross_pay"]),
        "tax_withheld": float(r["federal_tax_withheld"]),
        "social_security": float(r["social_security_withheld"]),
        "medicare": float(r["medicare_withheld"]),
        "state_tax_withheld": float(r["state_tax_withheld"] or 0),
        "benefits_deduction": float(r["benefits_deduction"]),
        "net_pay": float(r["net_pay"]),
        "filing_status": r["filing_status"],
        "recorded_at": r["created_at"],
        "employer_social_security": float(r["employer_social_security"] or 0),
        "employer_medicare": float(r["employer_medicare"] or 0),
        "employer_futa": float(r["employer_futa"] or 0),
        "employer_suta": float(r["employer_suta"] or 0),
        "company_id": r["company_id"] if "company_id" in r.keys() else None,
        "retirement_401k": float(r["retirement_401k"] or 0),
    }


def get_liabilities_report(company_id: Optional[str] = None) -> dict:
    """
    Aggregate employer payroll tax liabilities from all recorded payslips.
    Employer taxes are the FICA match (Social Security 6.2% same wage base,
    Medicare 1.45%) plus unemployment insurance (FUTA + SUTA). The
    Additional Medicare surtax is employee-only. Optionally filter by
    company_id using the company snapshot recorded with each payslip.
    """
    conn = _connect()
    try:
        where = "WHERE company_id = ?" if company_id is not None else ""
        params = (company_id,) if company_id is not None else ()
        row = conn.execute(
            "SELECT COUNT(*) AS n,"
            " COALESCE(SUM(CAST(gross_pay AS REAL)), 0) AS gross,"
            " COALESCE(SUM(CAST(federal_tax_withheld AS REAL)), 0) AS fed,"
            " COALESCE(SUM(CAST(state_tax_withheld AS REAL)), 0) AS st,"
            " COALESCE(SUM(CAST(social_security_withheld AS REAL)), 0) AS ss,"
            " COALESCE(SUM(CAST(medicare_withheld AS REAL)), 0) AS medi,"
            " COALESCE(SUM(CAST(employer_social_security AS REAL)), 0) AS er_ss,"
            " COALESCE(SUM(CAST(employer_medicare AS REAL)), 0) AS er_medi,"
            " COALESCE(SUM(CAST(employer_futa AS REAL)), 0) AS futa,"
            " COALESCE(SUM(CAST(employer_suta AS REAL)), 0) AS suta"
            f" FROM pay_history {where}",
            params,
        ).fetchone()
    finally:
        conn.close()

    employee_ss = row["ss"]
    employer_ss = row["er_ss"]
    total_fica = employee_ss + employer_ss + row["medi"] + row["er_medi"]
    total_unemployment = row["futa"] + row["suta"]
    return {
        "company_id": company_id,
        "payslips_recorded": row["n"],
        "gross_wages": round(row["gross"], 2),
        "employee_withholdings": {
            "federal_tax": round(row["fed"], 2),
            "state_tax": round(row["st"], 2),
            "social_security": round(employee_ss, 2),
            "medicare": round(row["medi"], 2),
        },
        "employer_taxes": {
            "social_security_match": round(employer_ss, 2),
            "medicare": round(row["er_medi"], 2),
            "futa": round(row["futa"], 2),
            "suta": round(row["suta"], 2),
            "unemployment_total": round(total_unemployment, 2),
        },
        "total_fica_liability": round(total_fica, 2),
        "total_employer_liability": round(total_fica + total_unemployment, 2),
    }


def run_payroll(hours_by_employee: Optional[Dict[int, Decimal]] = None,
                tax_brackets: Optional[List[tuple]] = None,
                pay_period_index: int = 0,
                company_id: Optional[str] = None) -> dict:
    """
    Run payroll for all employees for one pay period.

    - hours_by_employee: mapping of employee_id -> hours worked. Entries are
      required for hourly employees; missing ones are reported as errors.
    - tax_brackets: optional explicit table overriding each employee's
      filing-status preset.
    - pay_period_index: period label stored with the recorded payslips.
    - company_id: optionally restrict the run to one company's employees.

    Each generated payslip is persisted to the pay_history table. Returns
    per-employee payslips (with their history row ids), per-employee errors,
    period totals, and counts.
    """
    hours_by_employee = hours_by_employee or {}
    slips = []
    errors = []
    total_gross = Decimal("0")
    total_tax = Decimal("0")
    total_state = Decimal("0")
    total_ss = Decimal("0")
    total_medicare = Decimal("0")
    total_employer_ss = Decimal("0")
    total_employer_medicare = Decimal("0")
    total_futa = Decimal("0")
    total_suta = Decimal("0")
    total_retirement = Decimal("0")
    total_benefits = Decimal("0")
    total_net = Decimal("0")

    for employee in list_employees(company_id=company_id):
        try:
            slip = generate_payslip(employee,
                                    hours_worked=hours_by_employee.get(employee.id),
                                    pay_period_index=pay_period_index,
                                    tax_brackets=tax_brackets)
            slip["history_id"] = record_payslip(slip, pay_period_index=pay_period_index)
            slips.append(slip)
            total_gross += Decimal(str(slip["gross_pay"]))
            total_tax += Decimal(str(slip["tax_withheld"]))
            total_state += Decimal(str(slip["state_tax_withheld"]))
            total_ss += Decimal(str(slip["social_security"]))
            total_medicare += Decimal(str(slip["medicare"]))
            total_employer_ss += Decimal(str(slip["employer_social_security"]))
            total_employer_medicare += Decimal(str(slip["employer_medicare"]))
            total_futa += Decimal(str(slip["employer_futa"]))
            total_suta += Decimal(str(slip["employer_suta"]))
            total_retirement += Decimal(str(slip["retirement_401k"]))
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
            "state_tax_withheld": _f(total_state),
            "social_security": _f(total_ss),
            "medicare": _f(total_medicare),
            "employer_social_security": _f(total_employer_ss),
            "employer_medicare": _f(total_employer_medicare),
            "employer_futa": _f(total_futa),
            "employer_suta": _f(total_suta),
            "retirement_401k": _f(total_retirement),
            "benefits_deduction": _f(total_benefits),
            "net_pay": _f(total_net),
        },
        "employees_paid": len(slips),
        "employees_errored": len(errors),
    }


def get_ytd_summary(employee_id: int) -> dict:
    """
    Year-to-date (all-time recorded) totals for one employee, aggregated
    from their pay history: wages, every withholding, deductions, and the
    employer-side costs associated with their pay.
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS periods,"
            " COALESCE(SUM(CAST(hours_worked AS REAL)), 0) AS hours,"
            " COALESCE(SUM(CAST(gross_pay AS REAL)), 0) AS gross,"
            " COALESCE(SUM(CAST(federal_tax_withheld AS REAL)), 0) AS fed,"
            " COALESCE(SUM(CAST(state_tax_withheld AS REAL)), 0) AS st,"
            " COALESCE(SUM(CAST(social_security_withheld AS REAL)), 0) AS ss,"
            " COALESCE(SUM(CAST(medicare_withheld AS REAL)), 0) AS medi,"
            " COALESCE(SUM(CAST(retirement_401k AS REAL)), 0) AS ret,"
            " COALESCE(SUM(CAST(benefits_deduction AS REAL)), 0) AS ben,"
            " COALESCE(SUM(CAST(net_pay AS REAL)), 0) AS net"
            " FROM pay_history WHERE employee_id = ?",
            (employee_id,),
        ).fetchone()
    finally:
        conn.close()

    return {
        "employee_id": employee_id,
        "pay_periods_paid": row["periods"],
        "hours_worked": round(row["hours"], 2),
        "gross_wages": round(row["gross"], 2),
        "retirement_401k": round(row["ret"], 2),
        "taxable_wages_income_tax": round(row["gross"] - row["ret"], 2),
        "withholdings": {
            "federal_tax": round(row["fed"], 2),
            "state_tax": round(row["st"], 2),
            "social_security": round(row["ss"], 2),
            "medicare": round(row["medi"], 2),
        },
        "benefits_deduction": round(row["ben"], 2),
        "net_pay": round(row["net"], 2),
    }


_CSV_COLUMNS = [
    "history_id", "employee_id", "employee_name", "company_id", "pay_period_index",
    "hours_worked", "gross_pay", "federal_tax_withheld", "state_tax_withheld",
    "social_security", "medicare", "retirement_401k", "benefits_deduction",
    "net_pay", "employer_social_security", "employer_medicare",
    "employer_futa", "employer_suta", "recorded_at",
]


def export_history_csv(company_id: Optional[str] = None,
                       employee_id: Optional[int] = None) -> str:
    """
    Export recorded payslips as CSV text (a payroll journal suitable for
    accounting imports), optionally scoped to a company and/or employee.
    """
    conn = _connect()
    try:
        where = []
        params: List[Any] = []
        if company_id is not None:
            where.append("h.company_id = ?")
            params.append(company_id)
        if employee_id is not None:
            where.append("h.employee_id = ?")
            params.append(employee_id)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        rows = conn.execute(
            "SELECT h.*, e.name AS employee_name FROM pay_history h"
            " LEFT JOIN employees e ON e.id = h.employee_id"
            f" {clause} ORDER BY h.id",
            params,
        ).fetchall()
    finally:
        conn.close()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_CSV_COLUMNS)
    for r in rows:
        d = _history_row_to_dict(r)
        d["employee_name"] = r["employee_name"] if "employee_name" in r.keys() else None
        writer.writerow([d.get(col, "") for col in _CSV_COLUMNS])
    return buffer.getvalue()
