import random
from datetime import date, timedelta
from typing import Dict, Any, List, Optional

# ── Vendor catalogue ──────────────────────────────────────────────────────────
_VENDORS = [
    {
        "name": "TechSupplies Inc.",
        "domain": "techsupplies.com",
        "fraud_domain": "techsuppIies.com",   # capital-I lookalike
        "tax_id": "TS-1234-56",
        "iban": "FR7630006000011234567890188",
        "items": [
            {"item": "Laptop",   "price_range": (700,  1200), "qty_range": (1, 4)},
            {"item": "Monitor",  "price_range": (200,   500), "qty_range": (1, 6)},
            {"item": "Mouse",    "price_range": (15,     40), "qty_range": (3, 10)},
            {"item": "Keyboard", "price_range": (50,    120), "qty_range": (2,  8)},
        ],
    },
    {
        "name": "OfficeMart Ltd.",
        "domain": "officemart.ltd",
        "fraud_domain": "0fficemart.ltd",     # zero for 'o'
        "tax_id": "OM-7788-99",
        "iban": "GB82WEST12345698765432",
        "items": [
            {"item": "Office Chair",   "price_range": (150, 350), "qty_range": (2, 6)},
            {"item": "Standing Desk",  "price_range": (400, 800), "qty_range": (1, 4)},
            {"item": "Filing Cabinet", "price_range": (80,  180), "qty_range": (1, 5)},
            {"item": "Whiteboard",     "price_range": (60,  200), "qty_range": (1, 3)},
        ],
    },
    {
        "name": "GlobalTech Solutions",
        "domain": "globaltech.com",
        "fraud_domain": "g1obaltech.com",     # numeral-1 for 'l'
        "tax_id": "GT-9988-77",
        "iban": "DE89370400440532013000",
        "items": [
            {"item": "Server",         "price_range": (3000, 8000), "qty_range": (1, 3)},
            {"item": "Network Switch", "price_range": (200,   500), "qty_range": (2, 8)},
            {"item": "Cable Kit",      "price_range": (30,     80), "qty_range": (5, 20)},
            {"item": "UPS Unit",       "price_range": (300,   700), "qty_range": (1, 4)},
        ],
    },
    {
        "name": "Vertex Software",
        "domain": "vertex.com",
        "fraud_domain": "vertx.com",
        "tax_id": "VS-4455-66",
        "iban": "NL91ABNA0417164300",
        "items": [
            {"item": "Enterprise License", "price_range": (5000, 15000), "qty_range": (1, 1)},
            {"item": "Support Contract",   "price_range": (1000,  3000), "qty_range": (1, 2)},
            {"item": "Training Package",   "price_range": (500,   2000), "qty_range": (1, 3)},
        ],
    },
    {
        "name": "CloudServe Inc.",
        "domain": "cloudserve.io",
        "fraud_domain": "c1oudserve.io",
        "tax_id": "CS-3322-11",
        "iban": "IE12BOFI90000112345678",
        "items": [
            {"item": "Cloud Storage (TB)", "price_range": (100, 300), "qty_range": (5, 20)},
            {"item": "Compute Instance",   "price_range": (200, 600), "qty_range": (2, 10)},
            {"item": "Bandwidth (Gbps)",   "price_range": (50,  150), "qty_range": (5, 20)},
        ],
    },
]

_TAX_RATES = [0.08, 0.10, 0.12, 0.15]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _rand_date() -> date:
    start = date(2024, 1, 1)
    return start + timedelta(days=random.randint(0, 700))


def _fmt(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _invoice_num() -> str:
    return f"INV-{random.randint(2024, 2025)}-{random.randint(100, 999)}"


def _build_line_items(vendor: dict, n: int = None) -> List[Dict]:
    pool = vendor["items"]
    n = n or random.randint(2, min(3, len(pool)))
    chosen = random.sample(pool, n)
    return [
        {
            "item": it["item"],
            "qty": random.randint(*it["qty_range"]),
            "unit_price": round(random.randint(*it["price_range"]) * 1.0, 2),
        }
        for it in chosen
    ]


def _totals(line_items: List[Dict], tax_rate: float):
    subtotal = round(sum(i["qty"] * i["unit_price"] for i in line_items), 2)
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)
    return subtotal, tax, total


def _invoice_body(vendor_name, inv_num, inv_date, due_date,
                  line_items, subtotal, tax_amount, total, tax_rate,
                  tax_id: str = None, iban: str = None) -> str:
    li_lines = "\n".join(
        f"  - {li['item']} x{li['qty']} @ ${li['unit_price']:.2f} each"
        f" = ${li['qty'] * li['unit_price']:.2f}"
        for li in line_items
    )
    tax_pct = int(tax_rate * 100)
    hdr_tax = f"Tax ID: {tax_id}\n" if tax_id else ""
    hdr_iban = f"Bank Account (IBAN): {iban}\n" if iban else ""
    return (
        f"INVOICE\nVendor: {vendor_name}\n{hdr_tax}{hdr_iban}"
        f"Invoice Number: {inv_num}\nInvoice Date: {inv_date}\nDue Date: {due_date}\n"
        f"Line Items:\n{li_lines}\n"
        f"Subtotal: ${subtotal:.2f}\nTax ({tax_pct}%): ${tax_amount:.2f}\nTotal: ${total:.2f}"
    )


# ── Per-task generators ───────────────────────────────────────────────────────
def _gen_easy() -> dict:
    v = random.choice(_VENDORS)
    tax_rate = random.choice(_TAX_RATES)
    inv_num = _invoice_num()
    inv_date = _rand_date()
    due_date = inv_date + timedelta(days=random.choice([15, 30, 45]))
    items = _build_line_items(v)
    sub, tax, total = _totals(items, tax_rate)
    body = _invoice_body(v["name"], inv_num, _fmt(inv_date), _fmt(due_date),
                         items, sub, tax, total, tax_rate, iban=v["iban"])
    return {
        "name": "easy",
        "description": "Read email to get the invoice, query ERP to get PO, and extract all fields.",
        "emails": [{
            "id": "email_001",
            "sender": f"billing@{v['domain']}",
            "subject": f"Invoice {inv_num} attached",
            "body": f"Hi Accounts Payable,\nPlease find our latest invoice below.\n\n{body}\n\nRegards,\n{v['name']} Billing",
        }],
        "erp_database": {
            v["name"]: {
                "po_number": f"PO-{inv_num[4:]}",
                "vendor": v["name"],
                "iban": v["iban"],
                "approved_amount": total,
                "line_items": items,
            }
        },
        "erp_schema": {"required_key": "vendor_name"},
        "ground_truth": {
            "vendor_name": v["name"], "invoice_number": inv_num,
            "invoice_date": _fmt(inv_date), "due_date": _fmt(due_date),
            "subtotal": sub, "tax_amount": tax, "total_amount": total,
            "iban": v["iban"],
        },
        "expected_decision": "approve",
    }


def _gen_medium() -> dict:
    v = random.choice(_VENDORS)
    tax_rate = random.choice(_TAX_RATES)
    inv_num = _invoice_num()
    inv_date = _rand_date()
    due_date = inv_date + timedelta(days=random.choice([15, 30, 45]))
    items = _build_line_items(v)
    sub, tax, total = _totals(items, tax_rate)
    # ERP has lower prices → mismatch
    po_items = [
        {"item": li["item"], "qty": li["qty"],
         "unit_price": round(li["unit_price"] * random.uniform(0.70, 0.88), 2)}
        for li in items
    ]
    _, _, po_total = _totals(po_items, tax_rate)
    body = _invoice_body(v["name"], inv_num, _fmt(inv_date), _fmt(due_date),
                         items, sub, tax, total, tax_rate, iban=v["iban"])
    return {
        "name": "medium",
        "description": "Read email, query ERP, and detect a line item price mismatch.",
        "emails": [{
            "id": "email_002",
            "sender": f"finance@{v['domain']}",
            "subject": f"Urgent: Invoice {inv_num}",
            "body": f"Hello team,\nHere is our invoice.\n\n{body}",
        }],
        "erp_database": {
            v["name"]: {
                "po_number": f"PO-{inv_num[4:]}",
                "vendor": v["name"],
                "iban": v["iban"],
                "approved_amount": po_total,
                "line_items": po_items,
            }
        },
        "erp_schema": {"required_key": "vendor_name"},
        "ground_truth": {
            "vendor_name": v["name"], "invoice_number": inv_num,
            "invoice_date": _fmt(inv_date), "due_date": _fmt(due_date),
            "subtotal": sub, "tax_amount": tax, "total_amount": total,
            "iban": v["iban"],
        },
        "expected_decision": "reject",
        "expected_flags": ["price_mismatch"],
    }


def _gen_hard() -> dict:
    v = random.choice(_VENDORS)
    inv_num = _invoice_num()
    inv_date = _rand_date()
    due_date = inv_date + timedelta(days=random.choice([15, 30, 45]))
    items = _build_line_items(v)
    invoice_tax_rate = 0.15        # inflated tax claimed
    correct_tax_rate = random.choice([0.08, 0.10, 0.12])
    sub, inv_tax, inv_total = _totals(items, invoice_tax_rate)
    _, _, correct_total = _totals(items, correct_tax_rate)
    body = _invoice_body(v["name"], inv_num, _fmt(inv_date), _fmt(due_date),
                         items, sub, inv_tax, inv_total, invoice_tax_rate,
                         tax_id=v["tax_id"], iban=v["iban"])
    return {
        "name": "hard",
        "description": "Multi-App: Schema Drift. ERP now requires tax_id. Invoice is also a duplicate.",
        "emails": [{
            "id": "email_003",
            "sender": f"invoicing@{v['domain']}",
            "subject": f"FWD: Invoice {inv_num[-3:]} - {v['name']}",
            "body": f"Forwarded message:\n{body}",
        }],
        "erp_database": {
            v["tax_id"]: {
                "po_number": f"PO-{inv_num[4:]}",
                "vendor": v["name"],
                "iban": v["iban"],
                "approved_amount": correct_total,
                "line_items": items,
            }
        },
        "erp_schema": {
            "required_key": "vendor_tax_id",
            "message": "SCHEMA DRIFT: API v2 requires vendor_tax_id. vendor_name is deprecated.",
        },
        "previously_processed": [inv_num],
        "ground_truth": {
            "vendor_name": v["name"], "invoice_number": inv_num,
            "invoice_date": _fmt(inv_date), "due_date": _fmt(due_date),
            "subtotal": sub, "tax_amount": inv_tax, "total_amount": inv_total,
            "iban": v["iban"],
        },
        "expected_decision": "reject",
        "expected_flags": ["duplicate_invoice", "tax_mismatch"],
    }


def _gen_expert_negotiation() -> dict:
    v = random.choice(_VENDORS)
    tax_rate = random.choice([0.0, 0.05])
    inv_num_orig = _invoice_num()
    inv_num_corr = inv_num_orig + "-B"
    inv_date = _rand_date()
    due_date = inv_date + timedelta(days=30)
    items = _build_line_items(v, n=1)
    discount = random.choice([0.10, 0.15, 0.20, 0.25])
    # Invoice is inflated — discount not applied
    inflated_price = round(items[0]["unit_price"] / (1 - discount), 2)
    inflated = [{"item": items[0]["item"], "qty": items[0]["qty"], "unit_price": inflated_price}]
    inf_sub, inf_tax, inf_total = _totals(inflated, tax_rate)
    cor_sub, cor_tax, cor_total = _totals(items, tax_rate)
    tax_pct = int(tax_rate * 100)

    orig_body = (
        f"Hi team,\nAttached is our invoice.\n\nINVOICE\nVendor: {v['name']}\nBank Account (IBAN): {v['iban']}\n"
        f"Invoice Number: {inv_num_orig}\nInvoice Date: {_fmt(inv_date)}\nDue Date: {_fmt(due_date)}\n"
        f"Line Items:\n  - {inflated[0]['item']} x{inflated[0]['qty']} @ ${inflated_price:.2f}"
        f" each = ${inflated[0]['qty']*inflated_price:.2f}\n"
        f"Subtotal: ${inf_sub:.2f}\nTax ({tax_pct}%): ${inf_tax:.2f}\nTotal: ${inf_total:.2f}"
    )
    corrected_body = (
        f"Hi Accounts Payable,\nApologies — we forgot to apply your {int(discount*100)}% "
        f"partnership discount.\n\nCORRECTED INVOICE\nVendor: {v['name']}\nBank Account (IBAN): {v['iban']}\n"
        f"Invoice Number: {inv_num_corr}\nInvoice Date: {_fmt(inv_date)}\nDue Date: {_fmt(due_date)}\n"
        f"Line Items:\n  - {items[0]['item']} x{items[0]['qty']} @ ${items[0]['unit_price']:.2f}"
        f" each = ${items[0]['qty']*items[0]['unit_price']:.2f}\n"
        f"Subtotal: ${cor_sub:.2f}\nTax ({tax_pct}%): ${cor_tax:.2f}\nTotal: ${cor_total:.2f}"
    )
    return {
        "name": "expert_negotiation",
        "description": "Email vendor about mismatch, get corrected invoice, then approve.",
        "emails": [{
            "id": "email_004",
            "sender": f"sales@{v['domain']}",
            "subject": f"Invoice for {items[0]['item']}",
            "body": orig_body,
        }],
        "erp_database": {
            v["name"]: {
                "po_number": f"PO-{inv_num_orig[4:]}",
                "vendor": v["name"],
                "iban": v["iban"],
                "approved_amount": cor_total,
                "line_items": items,
            }
        },
        "erp_schema": {"required_key": "vendor_name"},
        "simulated_responses": {
            "trigger_action": "send_email",
            "trigger_target": f"sales@{v['domain']}",
            "response_email": {
                "id": "email_005",
                "sender": f"sales@{v['domain']}",
                "subject": f"Re: Invoice for {items[0]['item']} - CORRECTED",
                "body": corrected_body,
            },
        },
        "ground_truth": {
            "vendor_name": v["name"], "invoice_number": inv_num_corr,
            "invoice_date": _fmt(inv_date), "due_date": _fmt(due_date),
            "subtotal": cor_sub, "tax_amount": cor_tax, "total_amount": cor_total,
            "iban": v["iban"],
        },
        "expected_decision": "approve",
        "expected_flags": [],
    }


def _gen_expert_fraud() -> dict:
    v = random.choice(_VENDORS)
    tax_rate = random.choice(_TAX_RATES)
    inv_num = _invoice_num()
    inv_date = _rand_date()
    due_date = inv_date + timedelta(days=10)   # suspiciously short
    items = _build_line_items(v, n=1)
    sub, tax, total = _totals(items, tax_rate)
    
    # Fraud: generate a unique, truly random IBAN per episode.
    # Using `secrets` (OS-level entropy) so random.seed() from module import cannot fix this value.
    import secrets, string as _string
    _cc = secrets.choice(["CH", "LI", "AT", "NO", "SE", "PL", "RO", "CZ"])
    _check = "".join(secrets.choice(_string.digits) for _ in range(2))
    _bban  = "".join(secrets.choice(_string.digits) for _ in range(18))
    fraud_iban = f"{_cc}{_check}{_bban}"
    
    body = _invoice_body(v["name"], inv_num, _fmt(inv_date), _fmt(due_date),
                         items, sub, tax, total, tax_rate, iban=fraud_iban)
    return {
        "name": "expert_fraud",
        "description": "Fraud detection. Invoice comes from a lookalike email domain and contains a fraudulent bank account (IBAN).",
        "emails": [{
            "id": "email_006",
            "sender": f"billing@{v['fraud_domain']}",   # ← lookalike domain
            "subject": "URGENT: Overdue Invoice — Process Immediately",
            "body": f"Hello,\nThis is URGENT. Please process immediately. Note our new payment details below.\n\n{body}",
        }],
        "erp_database": {
            v["name"]: {
                "po_number": f"PO-{inv_num[4:]}",
                "vendor": v["name"],
                "iban": v["iban"],
                "approved_amount": total,
                "line_items": items,
            }
        },
        "erp_schema": {"required_key": "vendor_name"},
        "ground_truth": {
            "vendor_name": v["name"], "invoice_number": inv_num,
            "invoice_date": _fmt(inv_date), "due_date": _fmt(due_date),
            "subtotal": sub, "tax_amount": tax, "total_amount": total,
            "iban": fraud_iban,
        },
        "expected_decision": "reject",
        "expected_flags": ["fraud", "fraud_iban"],
    }


_GENERATORS = {
    "easy":               _gen_easy,
    "medium":             _gen_medium,
    "hard":               _gen_hard,
    "expert_negotiation": _gen_expert_negotiation,
    "expert_fraud":       _gen_expert_fraud,
}


def generate_task(task_name: str, seed: int = None) -> dict:
    """Return a freshly randomised task dict for *task_name*.

    Pass *seed* to get a reproducible episode (useful for evaluation).
    """
    if task_name not in _GENERATORS:
        raise ValueError(f"Unknown task: {task_name!r}. Valid: {list(_GENERATORS)}")
    if seed is not None:
        random.seed(seed)
    return _GENERATORS[task_name]()


# ── Backward-compat static TASKS (seeded so they stay stable) ─────────────────
TASKS: Dict[str, dict] = {name: generate_task(name, seed=42) for name in _GENERATORS}


# ── Grader ────────────────────────────────────────────────────────────────────
def grade_task(task_name: str, extracted: Dict[str, Any], flags: List[str],
               decision: str, erp_queried: bool = False,
               negotiated: bool = False,
               task_dict: Optional[Dict[str, Any]] = None) -> float:
    """Score an episode.  Pass *task_dict* to grade against the live randomised
    task instead of the static fallback."""
    task = task_dict or TASKS.get(task_name) or generate_task(task_name, seed=42)
    score = 0.0
    gt = task["ground_truth"]

    # Must follow multi-app workflow
    if not erp_queried:
        return 0.1

    # Field extraction (40 %)
    fields = ["vendor_name", "invoice_number", "invoice_date", "due_date",
              "subtotal", "tax_amount", "total_amount", "iban"]
    correct = 0
    for f in fields:
        if f in extracted:
            if isinstance(gt[f], float):
                try:
                    if abs(float(extracted[f]) - gt[f]) < 0.01:
                        correct += 1
                except (ValueError, TypeError):
                    pass
            else:
                if str(extracted[f]).strip() == str(gt[f]).strip():
                    correct += 1
    score += 0.4 * (correct / len(fields))

    # Flag / negotiation score (30 %)
    if task_name == "easy":
        score += 0.30
    elif task_name == "expert_negotiation":
        if negotiated:
            score += 0.30
    else:
        expected_flags = task.get("expected_flags", [])
        if expected_flags:
            hits = sum(1 for f in expected_flags if f in flags)
            score += 0.30 * (hits / len(expected_flags))

    # Decision (30 %)
    if decision == task.get("expected_decision", ""):
        score += 0.30

    return round(min(1.0, max(0.01, score)), 2)