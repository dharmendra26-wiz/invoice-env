from pydantic import BaseModel
from typing import Optional,List,Dict,Any

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

class Observation(BaseModel):
    invoice_text:str
    po_data:Optional[Dict[str,Any]]=None
    extracted_fields:Dict[str,Any]={}
    flags:List[str]=[]
    current_step:int=0
    message:str=""

class Reward(BaseModel):
    value:float
    reason:str

class StepResult(BaseModel):
    observation:Observation
    reward:float
    done:bool
    info:Dict[str,Any]={}