import os
import uuid

DB = f"smoke_{uuid.uuid4().hex[:8]}.db"
os.environ['PAYROLL_DB_PATH'] = DB
try:
    from fastapi.testclient import TestClient
    from app import app

    c = TestClient(app)

    # Multi-company: CA progressive employee at acme, no-state employee at globex
    ca_emp = c.post('/payroll/employees', json={
        'name': 'CA Carl', 'pay_type': 'salaried', 'annual_salary': 60000,
        'state': 'ca', 'company_id': 'acme'}).json()
    tx_emp = c.post('/payroll/employees', json={
        'name': 'TX Tia', 'pay_type': 'salaried', 'annual_salary': 52000,
        'company_id': 'globex'}).json()

    listing = c.get('/payroll/employees', params={'company_id': 'acme'}).json()
    print('acme listing:', [e['name'] for e in listing])
    assert len(listing) == 1

    # Run each company separately
    run_acme = c.post('/payroll/run', json={'company_id': 'acme'}).json()
    run_globex = c.post('/payroll/run', json={'hours': {}}).json()
    slip = run_acme['payslips'][0]
    print(f"CA slip: gross {slip['gross_pay']} | state {slip['state_tax_withheld']} "
          f"| futa {slip['employer_futa']} | suta {slip['employer_suta']}")
    assert slip['state_tax_withheld'] > 0 and slip['employer_futa'] > 0 and slip['employer_suta'] > 0
    assert run_globex['employees_paid'] == 1

    # Company-scoped liabilities
    rep_acme = c.get('/payroll/liabilities', params={'company_id': 'acme'}).json()
    rep_globex = c.get('/payroll/liabilities', params={'company_id': 'globex'}).json()
    print('acme:', rep_acme['employer_taxes'], '| total:', rep_acme['total_employer_liability'])
    print('globex:', rep_globex['employer_taxes'])
    assert rep_acme['payslips_recorded'] == 1 and rep_globex['payslips_recorded'] == 1
    assert rep_acme['employer_taxes']['suta'] > 0      # CA has a SUTA base
    assert rep_globex['employer_taxes']['suta'] == 0   # no state -> no SUTA
    assert rep_globex['employer_taxes']['futa'] > 0    # FUTA applies regardless
finally:
    if os.path.exists(DB):
        os.remove(DB)
print('SMOKE TEST OK')
