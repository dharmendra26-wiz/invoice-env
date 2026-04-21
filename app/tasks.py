from typing import Dict,Any,List

TASKS={
    "easy":{
        "name":"easy",
        "description":"Read email to get the invoice, query ERP to get PO, and extract all fields.",
        "emails":[
            {
                "id":"email_001",
                "sender":"billing@techsupplies.com",
                "subject":"Invoice INV-2024-001 attached",
                "body":"""Hi Accounts Payable,
Please find our latest invoice attached below.

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

Regards,
TechSupplies Billing"""
            }
        ],
        "erp_database":{
            "TechSupplies Inc.": {
                "po_number":"PO-2024-001",
                "vendor":"TechSupplies Inc.",
                "approved_amount":1870.00,
                "line_items":[
                    {"item":"Laptop","qty":2,"unit_price":800.00},
                    {"item":"Mouse","qty":5,"unit_price":20.00}
                ]
            }
        },
        "erp_schema": {"required_key": "vendor_name"},
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
        "description":"Read email, query ERP, and detect a line item price mismatch.",
        "emails":[
            {
                "id":"email_002",
                "sender":"finance@officemart.ltd",
                "subject":"Urgent: Invoice INV-2024-042",
                "body":"""Hello team,
Here is the invoice for the office furniture.
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
"""
            }
        ],
        "erp_database":{
            "OfficeMart Ltd.": {
                "po_number":"PO-2024-042",
                "vendor":"OfficeMart Ltd.",
                "approved_amount":1870.00,
                "line_items":[
                    {"item":"Office Chair","qty":3,"unit_price":200.00},
                    {"item":"Desk","qty":2,"unit_price":550.00}
                ]
            }
        },
        "erp_schema": {"required_key": "vendor_name"},
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
        "description":"Multi-App workflow with Schema Drift. ERP requires tax_id instead of vendor_name. Detect duplicate invoice.",
        "emails":[
            {
                "id":"email_003",
                "sender":"invoicing@globaltech.com",
                "subject":"FWD: Invoice 99 - GlobalTech",
                "body":"""Forwarded message:
Vendor: GlobalTech Solutions
Tax ID: GT-9988-77
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
"""
            }
        ],
        "erp_database":{
            "GT-9988-77": {
                "po_number":"PO-2024-099",
                "vendor":"GlobalTech Solutions",
                "approved_amount":7705.00,
                "line_items":[
                    {"item":"Server","qty":1,"unit_price":5000.00},
                    {"item":"Network Switch","qty":4,"unit_price":300.00},
                    {"item":"Cable Kit","qty":10,"unit_price":50.00}
                ]
            }
        },
        "erp_schema": {"required_key": "vendor_tax_id", "message": "SCHEMA DRIFT: API v2 requires vendor_tax_id. vendor_name is deprecated."},
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
    },

    "expert_negotiation":{
        "name":"expert_negotiation",
        "description":"Multi-Agent interaction. Email vendor about mismatch, get corrected invoice, and approve.",
        "emails":[
            {
                "id":"email_004",
                "sender":"sales@vertex.com",
                "subject":"Invoice for Q2 software license",
                "body":"""Hi team,
Attached is our Q2 invoice. 

INVOICE
Vendor: Vertex Software
Invoice Number: INV-2024-500
Invoice Date: 2024-05-01
Due Date: 2024-06-01
Line Items:
  - Enterprise License x1 @ $10000.00 each = $10000.00
Subtotal: $10000.00
Tax (0%): $0.00
Total: $10000.00
"""
            }
        ],
        "erp_database":{
            "Vertex Software": {
                "po_number":"PO-2024-500",
                "vendor":"Vertex Software",
                "approved_amount":8000.00,
                "line_items":[
                    {"item":"Enterprise License","qty":1,"unit_price":8000.00}
                ]
            }
        },
        "erp_schema": {"required_key": "vendor_name"},
        "simulated_responses": {
            "trigger_action": "send_email",
            "trigger_target": "sales@vertex.com",
            "response_email": {
                "id":"email_005",
                "sender":"sales@vertex.com",
                "subject":"Re: Invoice for Q2 software license - CORRECTED",
                "body":"""Hi Accounts Payable,
Apologies, you are completely right. We forgot to apply your standard 20% partnership discount to the Enterprise License.
Here is the corrected invoice:

INVOICE
Vendor: Vertex Software
Invoice Number: INV-2024-500-B
Invoice Date: 2024-05-01
Due Date: 2024-06-01
Line Items:
  - Enterprise License x1 @ $8000.00 each = $8000.00
Subtotal: $8000.00
Tax (0%): $0.00
Total: $8000.00
"""
            }
        },
        "ground_truth":{
            "vendor_name":"Vertex Software",
            "invoice_number":"INV-2024-500-B",
            "invoice_date":"2024-05-01",
            "due_date":"2024-06-01",
            "subtotal":8000.00,
            "tax_amount":0.00,
            "total_amount":8000.00
        },
        "expected_decision":"approve",
        "expected_flags":[]
    },

    "expert_fraud":{
        "name":"expert_fraud",
        "description":"Fraud detection. Lookalike domain name in email sender.",
        "emails":[
            {
                "id":"email_006",
                "sender":"billing@techsuppIies.com",
                "subject":"URGENT: Overdue Invoice",
                "body":"""Hello,
This is an urgent notice that your latest invoice is overdue. Please process immediately.

INVOICE
Vendor: TechSupplies Inc.
Invoice Number: INV-2024-999
Invoice Date: 2024-06-15
Due Date: 2024-06-25
Line Items:
  - Consulting Services x1 @ $5000.00 each = $5000.00
Subtotal: $5000.00
Tax (10%): $500.00
Total: $5500.00
"""
            }
        ],
        "erp_database":{
            "TechSupplies Inc.": {
                "po_number":"PO-2024-999",
                "vendor":"TechSupplies Inc.",
                "approved_amount":5500.00,
                "line_items":[
                    {"item":"Consulting Services","qty":1,"unit_price":5000.00}
                ]
            }
        },
        "erp_schema": {"required_key": "vendor_name"},
        "ground_truth":{
            "vendor_name":"TechSupplies Inc.",
            "invoice_number":"INV-2024-999",
            "invoice_date":"2024-06-15",
            "due_date":"2024-06-25",
            "subtotal":5000.00,
            "tax_amount":500.00,
            "total_amount":5500.00
        },
        "expected_decision":"reject",
        "expected_flags":["fraud"]
    }
}

def grade_task(task_name:str,extracted:Dict[str,Any],flags:List[str],decision:str,erp_queried:bool=False,negotiated:bool=False)->float:
    task=TASKS[task_name]
    score=0.0
    gt=task["ground_truth"]

    # Deduct heavily if ERP wasn't even queried
    if not erp_queried:
        return 0.1 # Absolute minimum for not following multi-app workflow

    # field extraction score (40%)
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
    score+=0.4*(correct/len(fields))

    # flag score (30%)
    if task_name=="easy":
        score+=0.30
    elif task_name=="expert_negotiation":
        if negotiated:
            score+=0.30
    else:
        expected_flags=task.get("expected_flags",[])
        if expected_flags:
            flag_hits=sum(1 for f in expected_flags if f in flags)
            score+=0.30*(flag_hits/len(expected_flags))

    # decision score (30%)
    if decision==task.get("expected_decision",""):
        score+=0.30

    score=round(score,2)
    score=min(1.0,max(0.01,score))
    return score