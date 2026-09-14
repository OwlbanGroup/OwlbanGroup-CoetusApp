"""
Mobile-friendly web dashboard for OwlbanGroup-CoetusApp payroll.

Served at /dashboard as a single self-contained HTML page (no build step,
no external assets) that talks to the existing JSON API.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Coetus Payroll</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --ink:#e2e8f0; --dim:#94a3b8;
          --accent:#38bdf8; --ok:#4ade80; --warn:#fbbf24; }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--ink);
         font-family:-apple-system,'Segoe UI',Roboto,sans-serif;
         padding:16px; max-width:640px; margin:0 auto; }
  h1 { font-size:1.3rem; margin-bottom:4px; }
  .sub { color:var(--dim); font-size:.8rem; margin-bottom:14px; }
  button { background:var(--accent); color:#082f49; border:0; border-radius:10px;
           padding:12px 16px; font-weight:700; font-size:1rem; width:100%;
           margin:6px 0; }
  button.secondary { background:var(--card); color:var(--ink); font-weight:500; }
  .card { background:var(--card); border-radius:12px; padding:14px; margin:10px 0; }
  .name { font-weight:700; font-size:1.05rem; }
  .meta { color:var(--dim); font-size:.8rem; margin-top:2px; line-height:1.5; }
  .row { display:flex; justify-content:space-between; padding:3px 0;
         font-size:.9rem; }
  .row span:last-child { font-variant-numeric:tabular-nums; }
  .net { color:var(--ok); font-weight:700; }
  .hidden { display:none; }
  #status { color:var(--warn); font-size:.85rem; min-height:1.2em; margin:6px 0; }
  a { color:var(--accent); text-decoration:none; }
  .links { text-align:center; margin-top:14px; font-size:.9rem; }
  .links a { margin:0 10px; }
  input, select { background:#0f172a; color:var(--ink); border:1px solid #334155;
                  border-radius:8px; padding:10px; font-size:.95rem; }
</style>
</head>
<body>
<h1>Coetus Payroll</h1>
<div class="sub">OwlbanGroup &middot; live payroll console</div>

<button class="secondary" onclick="toggleForm()">+ Add employee</button>
<div class="card hidden" id="addform">
  <div class="name">New employee</div>
  <input id="f-name" placeholder="Full name" style="width:100%">
  <div style="display:flex;gap:8px;margin-top:8px">
    <select id="f-paytype" style="flex:1">
      <option value="salaried">Salaried</option>
      <option value="hourly">Hourly</option>
    </select>
    <input id="f-company" placeholder="Company (optional)" style="flex:1">
  </div>
  <div style="display:flex;gap:8px;margin-top:8px">
    <input id="f-salary" type="number" min="0" placeholder="Annual salary"
           style="flex:1">
    <input id="f-rate" type="number" min="0" step="0.01" placeholder="Hourly rate"
           style="flex:1">
  </div>
  <div style="display:flex;gap:8px;margin-top:8px">
    <select id="f-filing" style="flex:1"></select>
    <select id="f-state" style="flex:1"></select>
  </div>
  <div style="display:flex;gap:8px;margin-top:8px">
    <input id="f-401k" type="number" min="0" max="100" step="0.1"
           placeholder="401(k) %" style="flex:1">
    <input id="f-benefits" type="number" min="0" step="0.01"
           placeholder="Benefits $/period" style="flex:1">
  </div>
  <button style="margin-top:10px" onclick="addEmployee()">Save employee</button>
</div>

<div class="card" id="runcard">
  <div style="display:flex;gap:8px;align-items:center">
    <span class="meta" style="white-space:nowrap">Period #</span>
    <input id="run-period" type="number" min="0" step="1" value="0"
           style="width:80px">
    <button style="flex:1;margin:0" onclick="runPayroll()">Run Payroll</button>
  </div>
</div>
<div id="status"></div>
<div id="employees"></div>

<div class="card hidden" id="liab"></div>

<button class="secondary" onclick="loadLiabilities()">Show employer liabilities</button>

<div class="links">
  <a href="/payroll/export.csv">CSV journal</a>
  <a href="/docs">API docs</a>
  <a href="#" onclick="loadEmployees();return false;">Refresh</a>
</div>

<script>
const money = n => '$' + Number(n || 0).toLocaleString('en-US',
                     {minimumFractionDigits: 2, maximumFractionDigits: 2});
const esc = s => String(s ?? '').replace(/[&<>"']/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

async function loadRates() {
  let rates;
  try {
    rates = await (await fetch('/payroll/rates')).json();
  } catch (e) { return; }
  const filing = document.getElementById('f-filing');
  filing.innerHTML = rates.filing_statuses.map(
      s => '<option value="' + s + '">' +
          s.replaceAll('_', ' ') + '</option>').join('');
  const state = document.getElementById('f-state');
  state.innerHTML = '<option value="">No state tax</option>' +
      rates.states.filter(s => s !== 'none').map(
          s => '<option value="' + s + '">' + s.toUpperCase() + '</option>').join('');
}

function toggleForm() {
  document.getElementById('addform').classList.toggle('hidden');
}

async function addEmployee() {
  const s = document.getElementById('status');
  const paytype = document.getElementById('f-paytype').value;
  const body = {
    name: document.getElementById('f-name').value.trim(),
    pay_type: paytype,
    filing_status: document.getElementById('f-filing').value || null,
    state: document.getElementById('f-state').value || null,
    retirement_401k_percent: Number(document.getElementById('f-401k').value || 0),
    benefits_deduction_per_period:
        Number(document.getElementById('f-benefits').value || 0),
  };
  const company = document.getElementById('f-company').value.trim();
  if (company) body.company_id = company;
  if (paytype === 'salaried') {
    body.annual_salary = Number(document.getElementById('f-salary').value || 0);
  } else {
    body.hourly_rate = Number(document.getElementById('f-rate').value || 0);
  }
  if (!body.name) { s.textContent = 'Name is required.'; return; }
  const res = await fetch('/payroll/employees', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)});
  const data = await res.json();
  if (data.error) { s.textContent = 'Error: ' + data.error; return; }
  s.textContent = 'Added ' + data.name + ' (#' + data.id + ')';
  document.getElementById('addform').classList.add('hidden');
  loadEmployees();
}

async function voidPayslip(empId, slipId) {
  const s = document.getElementById('status');
  const res = await fetch('/payroll/employees/' + empId +
                          '/payslips/' + slipId, {method: 'DELETE'});
  const data = await res.json();
  if (data.error) { s.textContent = 'Error: ' + data.error; return; }
  s.textContent = 'Voided payslip #' + slipId;
  loadEmployees();
}

async function loadEmployees() {
  const box = document.getElementById('employees');
  box.innerHTML = '<div class="card">Loading…</div>';
  const emps = await (await fetch('/payroll/employees')).json();
  if (!emps.length) {
    box.innerHTML = '<div class="card">No employees yet. Use POST ' +
                    '/payroll/employees or the API docs to add one.</div>';
    return;
  }
  box.innerHTML = '';
  for (const e of emps) {
    const card = document.createElement('div');
    card.className = 'card';
    const bits = [e.pay_type,
      e.annual_salary ? 'salary ' + money(e.annual_salary) : null,
      e.hourly_rate ? 'rate ' + money(e.hourly_rate) + '/h' : null,
      e.filing_status ? 'fed: ' + e.filing_status : null,
      e.state ? 'state: ' + e.state : null,
      e.retirement_401k_percent ? '401k ' + e.retirement_401k_percent + '%' : null,
      e.company_id ? 'company: ' + e.company_id : null].filter(Boolean);
    card.innerHTML =
      '<div class="name">' + esc(e.name) + '</div>' +
      '<div class="meta">#' + e.id + ' · ' + bits.join(' · ') + '</div>' +
      '<button class="secondary" style="margin:10px 0 0">Show payslips / YTD</button>' +
      '<div class="hidden"></div>';
    const btn = card.querySelector('button');
    const panel = card.querySelector('.hidden');
    btn.onclick = async () => {
      if (!panel.classList.contains('hidden')) {
        panel.classList.add('hidden'); btn.textContent = 'Show payslips / YTD'; return;
      }
      btn.textContent = 'Hide';
      panel.classList.remove('hidden');
      panel.innerHTML = '<div class="meta">Loading…</div>';
      const slips = await (await fetch(
          '/payroll/employees/' + e.id + '/payslips')).json();
      const ytd = await (await fetch('/payroll/employees/' + e.id + '/ytd')).json();
      let html = '<div class="meta" style="margin:8px 0 2px">YTD (' +
                 ytd.pay_periods_paid + ' periods)</div>';
      html += row('Gross', ytd.gross_wages) +
              row('401(k)', ytd.retirement_401k) +
              row('Federal tax', ytd.withholdings.federal_tax) +
              row('State tax', ytd.withholdings.state_tax) +
              row('Social Security', ytd.withholdings.social_security) +
              row('Medicare', ytd.withholdings.medicare) +
              row('Benefits', ytd.benefits_deduction);
      html += row('YTD Net', ytd.net_pay, true);
      if (slips.length)
        html += '<div class="meta" style="margin:8px 0 2px">Latest slip #' +
                slips[0].id + '</div>' +
                row('Gross', slips[0].gross_pay) +
                row('Net', slips[0].net_pay, true) +
                '<div class="meta"><a href="/payroll/employees/' + e.id +
                '/payslips/' + slips[0].id + '/pdf">Download PDF</a>' +
                ' · <a href="#" onclick="voidPayslip(' + e.id + ',' +
                slips[0].id + ');return false;">Void slip</a></div>';
      else
        html += '<div class="meta">No payslips recorded yet.</div>';
      panel.innerHTML = html;
    };
    box.appendChild(card);
  }
}

const row = (label, v, hl) => '<div class="row' + (hl ? ' net' : '') +
          '"><span>' + label + '</span><span>' + money(v) + '</span></div>';

async function runPayroll() {
  const s = document.getElementById('status');
  s.textContent = 'Running payroll…';
  const period = parseInt(document.getElementById('run-period').value, 10) || 0;
  const res = await fetch('/payroll/run', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({pay_period_index: period})});
  const data = await res.json();
  if (data.error) { s.textContent = 'Error: ' + data.error; return; }
  s.textContent = 'Paid ' + data.employees_paid + ', errors ' +
                  data.employees_errored + ' · net ' +
                  money(data.totals.net_pay);
  loadEmployees();
}

async function loadLiabilities() {
  const el = document.getElementById('liab');
  el.classList.toggle('hidden');
  if (el.classList.contains('hidden')) return;
  const d = await (await fetch('/payroll/liabilities')).json();
  el.innerHTML = '<div class="name">Employer liabilities</div>' +
    row('Payslips', d.payslips_recorded) +
    row('Gross wages', d.gross_wages) +
    row('SS match', d.employer_taxes.social_security_match) +
    row('Employer Medicare', d.employer_taxes.medicare) +
    row('FUTA', d.employer_taxes.futa) +
    row('SUTA', d.employer_taxes.suta) +
    row('Total liability', d.total_employer_liability, true);
}

loadRates();
loadEmployees();
</script>
</body>
</html>"""


def get_dashboard_html() -> str:
    """Return the dashboard page markup."""
    return DASHBOARD_HTML
