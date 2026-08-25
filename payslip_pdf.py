"""
Payslip PDF rendering for OwlbanGroup-CoetusApp.

Renders a recorded payslip (a pay_history row dict) as a simple,
clean one-page PDF using fpdf2 and returns the raw bytes.
"""

from fpdf import FPDF

_MONEY_FIELDS = [
    ("gross_pay", "Gross Pay"),
    ("retirement_401k", "401(k) Contribution (pre-tax)"),
    ("tax_withheld", "Federal Tax Withheld"),
    ("state_tax_withheld", "State Tax Withheld"),
    ("social_security", "Social Security"),
    ("medicare", "Medicare"),
    ("benefits_deduction", "Benefits Deduction"),
    ("net_pay", "Net Pay"),
]

_EMPLOYER_FIELDS = [
    ("employer_social_security", "Employer Social Security Match"),
    ("employer_medicare", "Employer Medicare"),
]


def _safe(text) -> str:
    """Coerce to a latin-1-safe string for fpdf2's core fonts."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _label_value_row(pdf: FPDF, label: str, value: str):
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(95, 8, _safe(label))
    pdf.cell(0, 8, _safe(value), align="R", new_x="LMARGIN", new_y="NEXT")


def render_payslip_pdf(record: dict) -> bytes:
    """
    Build a one-page PDF for a recorded payslip dict (as returned by
    payroll.get_payslip_record / get_pay_history). Returns raw PDF bytes.
    """
    pdf = FPDF(format="letter")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "PAYSLIP", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(80, 80, 80)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(4)

    # Identity block
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _safe(record.get("employee_name", f"Employee {record['employee_id']}")),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Employee ID: {record['employee_id']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Pay Period Index: {record.get('pay_period_index', 0)}",
             new_x="LMARGIN", new_y="NEXT")
    if record.get("hours_worked") is not None:
        pdf.cell(0, 6, f"Hours Worked: {record['hours_worked']}",
                 new_x="LMARGIN", new_y="NEXT")
    if record.get("filing_status"):
        pdf.cell(0, 6, f"Filing Status: {_safe(record['filing_status'])}",
                 new_x="LMARGIN", new_y="NEXT")
    if record.get("recorded_at"):
        pdf.cell(0, 6, f"Recorded At: {_safe(record['recorded_at'])}",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Earnings & deductions
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Earnings & Deductions", new_x="LMARGIN", new_y="NEXT")
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    for key, label in _MONEY_FIELDS:
        value = float(record.get(key, 0))
        bold = key == "net_pay"
        pdf.set_font("Helvetica", "B" if bold else "", 11)
        pdf.cell(95, 8, label)
        pdf.cell(0, 8, f"${value:,.2f}", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Employer contributions (informational)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Employer Contributions (not deducted from pay)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    for key, label in _EMPLOYER_FIELDS:
        _label_value_row(pdf, label, f"${float(record.get(key, 0)):,.2f}")

    return bytes(pdf.output())
