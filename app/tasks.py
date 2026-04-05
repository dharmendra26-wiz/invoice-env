from typing import Dict,Any,List

TASKS={
    "easy":{
        "name":"easy",
        "description":"Extract all fields from a clean invoice and match to purchase order",
        "invoice_text":"""
INVOICE
Vendor: TechSupplies Inc.
Invoice Number: INV-2024-001
Invoice Date: 2024-01-15
Due Date: 2024-02-15
Line Items:
  - Laptop x2 @ $800.00 each = $1600.00
  - Mouse x5 @ $20.00 each = $100.00
Subtotal: $1700.00
Tax (10%): $170.00
Total: $1870.00
""",
        "po_data":{
            "po_number":"PO-2024-001",
            "vendor":"TechSupplies Inc.",
            "approved_amount":1870.00,
            "line_items":[
                {"item":"Laptop","qty":2,"unit_price":800.00},
                {"item":"Mouse","qty":5,"unit_price":20.00}
            ]
        },
        "ground_truth":{
            "vendor_name":"TechSupplies Inc.",
            "invoice_number":"INV-2024-001",
            "invoice_date":"2024-01-15",
            "due_date":"2024-02-15",
            "subtotal":1700.00,
            "tax_amount":170.00,
            "total_amount":1870.00
        },
        "expected_decision":"approve"
    },

    "medium":{
        "name":"medium",
        "description":"Detect a line item price mismatch between invoice and purchase order",
        "invoice_text":"""
INVOICE
Vendor: OfficeMart Ltd.
Invoice Number: INV-2024-042
Invoice Date: 2024-02-01
Due Date: 2024-03-01
Line Items:
  - Office Chair x3 @ $250.00 each = $750.00
  - Desk x2 @ $600.00 each = $1200.00
Subtotal: $1950.00
Tax (10%): $195.00
Total: $2145.00
""",
        "po_data":{
            "po_number":"PO-2024-042",
            "vendor":"OfficeMart Ltd.",
            "approved_amount":1870.00,
            "line_items":[
                {"item":"Office Chair","qty":3,"unit_price":200.00},
                {"item":"Desk","qty":2,"unit_price":550.00}
            ]
        },
        "ground_truth":{
            "vendor_name":"OfficeMart Ltd.",
            "invoice_number":"INV-2024-042",
            "invoice_date":"2024-02-01",
            "due_date":"2024-03-01",
            "subtotal":1950.00,
            "tax_amount":195.00,
            "total_amount":2145.00
        },
        "expected_decision":"reject",
        "expected_flags":["price_mismatch"]
    },

    "hard":{
        "name":"hard",
        "description":"Detect duplicate invoice, wrong tax calculation, and missing PO reference",
        "invoice_text":"""
INVOICE
Vendor: GlobalTech Solutions
Invoice Number: INV-2024-099
Invoice Date: 2024-03-01
Due Date: 2024-04-01
Line Items:
  - Server x1 @ $5000.00 each = $5000.00
  - Network Switch x4 @ $300.00 each = $1200.00
  - Cable Kit x10 @ $50.00 each = $500.00
Subtotal: $6700.00
Tax (15%): $800.00
Total: $7500.00
""",
        "po_data":{
            "po_number":"PO-2024-099",
            "vendor":"GlobalTech Solutions",
            "approved_amount":7705.00,
            "line_items":[
                {"item":"Server","qty":1,"unit_price":5000.00},
                {"item":"Network Switch","qty":4,"unit_price":300.00},
                {"item":"Cable Kit","qty":10,"unit_price":50.00}
            ]
        },
        "previously_processed":["INV-2024-099"],
        "ground_truth":{
            "vendor_name":"GlobalTech Solutions",
            "invoice_number":"INV-2024-099",
            "invoice_date":"2024-03-01",
            "due_date":"2024-04-01",
            "subtotal":6700.00,
            "tax_amount":800.00,
            "total_amount":7500.00
        },
        "expected_decision":"reject",
        "expected_flags":["duplicate_invoice","tax_mismatch"]
    }
}

def grade_task(task_name:str,extracted:Dict[str,Any],flags:List[str],decision:str)->float:
    task=TASKS[task_name]
    score=0.0
    gt=task["ground_truth"]

    # field extraction score (50%)
    fields=["vendor_name","invoice_number","invoice_date","due_date","subtotal","tax_amount","total_amount"]
    correct=0
    for f in fields:
        if f in extracted:
            if isinstance(gt[f],float):
                if abs(float(extracted[f])-gt[f])<0.01:
                    correct+=1
            else:
                if str(extracted[f]).strip()==str(gt[f]).strip():
                    correct+=1
    score+=0.5*(correct/len(fields))

    # flag score (25%)
    if task_name=="easy":
        score+=0.25
    else:
        expected_flags=task.get("expected_flags",[])
        if expected_flags:
            flag_hits=sum(1 for f in expected_flags if f in flags)
            score+=0.25*(flag_hits/len(expected_flags))

    # decision score (25%)
    if decision==task.get("expected_decision",""):
        score+=0.25

    return round(score,2)