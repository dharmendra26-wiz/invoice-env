import sys
import os

# Ensure the app module can be imported
sys.path.insert(0, r"c:\Users\dharm\invoice-env")

from app.environment import InvoiceEnvironment
from app.models import Action

def test_expert_negotiation():
    print("--- TESTING EXPERT NEGOTIATION ---")
    env = InvoiceEnvironment("expert_negotiation")
    obs = env.reset()
    
    print(f"Step 0: {obs.message}")
    print(f"Inbox has {len(obs.inbox_status)} emails.")

    # Step 1: Read Email
    action = Action(action_type="read_email", email_id="email_004")
    res = env.step(action)
    print(f"Step 1 (read_email): {res.observation.message}")

    # Step 2: Query ERP
    action = Action(action_type="query_erp", api_endpoint="/api/v1/po", api_payload={"vendor_name": "Vertex Software"})
    res = env.step(action)
    print(f"Step 2 (query_erp): {res.observation.message}")

    # Step 3: Extract totals from first email
    env.step(Action(action_type="extract", field_name="total_amount", field_value=10000.00))
    
    # Step 4: Notice discrepancy, send email
    action = Action(action_type="send_email", email_id="sales@vertex.com", email_subject="Mismatch", email_body="Please fix your invoice price.")
    res = env.step(action)
    print(f"Step 4 (send_email): {res.observation.message}")
    print(f"Inbox now has {len(res.observation.inbox_status)} emails.")
    
    # Step 5: Read the new email
    new_email_id = res.observation.inbox_status[-1]["id"]
    action = Action(action_type="read_email", email_id=new_email_id)
    res = env.step(action)
    print(f"Step 5 (read_email reply): {res.observation.message}")

    # Step 6: Extract all correct fields from new email
    env.step(Action(action_type="extract", field_name="vendor_name", field_value="Vertex Software"))
    env.step(Action(action_type="extract", field_name="invoice_number", field_value="INV-2024-500-B"))
    env.step(Action(action_type="extract", field_name="invoice_date", field_value="2024-05-01"))
    env.step(Action(action_type="extract", field_name="due_date", field_value="2024-06-01"))
    env.step(Action(action_type="extract", field_name="subtotal", field_value=8000.00))
    env.step(Action(action_type="extract", field_name="tax_amount", field_value=0.00))
    env.step(Action(action_type="extract", field_name="total_amount", field_value=8000.00))

    # Step 7: Match PO
    res = env.step(Action(action_type="match_po"))
    print(f"Step 7 (match_po): {res.observation.message}")

    # Step 8: Approve
    res = env.step(Action(action_type="approve"))
    print(f"Final Step (approve): {res.observation.message}")
    print(f"Final Reward: {res.reward}")
    print("----------------------------------\n")

def test_expert_fraud():
    print("--- TESTING EXPERT FRAUD ---")
    env = InvoiceEnvironment("expert_fraud")
    obs = env.reset()
    
    print(f"Step 0: {obs.message}")

    # Step 1: Read Email
    action = Action(action_type="read_email", email_id="email_006")
    res = env.step(action)
    print(f"Step 1 (read_email): {res.observation.message}")
    
    # Step 2: Query ERP
    action = Action(action_type="query_erp", api_endpoint="/api/v1/po", api_payload={"vendor_name": "TechSupplies Inc."})
    res = env.step(action)
    print(f"Step 2 (query_erp): {res.observation.message}")

    # Step 3: Flag fraud
    action = Action(action_type="flag", field_name="fraud")
    res = env.step(action)
    print(f"Step 3 (flag fraud): {res.observation.message}")

    # Step 4: Extract all correct fields from email (to maximize score)
    env.step(Action(action_type="extract", field_name="vendor_name", field_value="TechSupplies Inc."))
    env.step(Action(action_type="extract", field_name="invoice_number", field_value="INV-2024-999"))
    env.step(Action(action_type="extract", field_name="invoice_date", field_value="2024-06-15"))
    env.step(Action(action_type="extract", field_name="due_date", field_value="2024-06-25"))
    env.step(Action(action_type="extract", field_name="subtotal", field_value=5000.00))
    env.step(Action(action_type="extract", field_name="tax_amount", field_value=500.00))
    env.step(Action(action_type="extract", field_name="total_amount", field_value=5500.00))

    # Step 5: Reject
    res = env.step(Action(action_type="reject"))
    print(f"Final Step (reject): {res.observation.message}")
    print(f"Final Reward: {res.reward}")
    print("----------------------------------\n")

if __name__ == "__main__":
    test_expert_negotiation()
    test_expert_fraud()
