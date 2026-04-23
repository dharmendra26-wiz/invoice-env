from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class InvoiceField(BaseModel):
    vendor_name:str
    invoice_number:str
    invoice_date:str
    due_date:str
    subtotal:float
    tax_amount:float
    total_amount:float
    line_items:List[Dict[str,Any]]

class Action(BaseModel):
    action_type:str 
    field_name:Optional[str]=None
    field_value:Optional[Any]=None
    reason:Optional[str]=None
    api_endpoint:Optional[str]=None
    api_payload:Optional[Dict[str,Any]]=None
    email_id:Optional[str]=None
    email_subject:Optional[str]=None
    email_body:Optional[str]=None

class Observation(BaseModel):
    inbox_status:List[Dict[str,Any]]=[]
    email_content:Optional[str]=None
    invoice_text:Optional[str]=None
    erp_response:Optional[Dict[str,Any]]=None
    extracted_fields:Dict[str,Any]={}
    flags:List[str]=[]
    current_step:int=0
    message:str=""

class Reward(BaseModel):
    value:float
    reason:str

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class ResetResponse(BaseModel):
    """Returned by POST /reset.  Carry session_id on every subsequent /step call."""
    session_id: str
    observation: Observation