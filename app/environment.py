from typing import Dict,Any,Optional
from app.models import Action,Observation,Reward,StepResult
from app.tasks import generate_task,grade_task

class InvoiceEnvironment:
    def __init__(self,task_name:str="easy",seed:int=None):
        self.task_name=task_name
        self.seed=seed
        self.task=generate_task(task_name,seed=seed)
        self.extracted_fields:Dict[str,Any]={}
        self.flags=[]
        self.decision:Optional[str]=None
        self.current_step=0
        self.done=False
        self.total_reward=0.0
        
        # Enterprise State
        self.email_content:Optional[str]=None
        self.invoice_text:Optional[str]=None
        self.erp_response:Optional[Dict[str,Any]]=None
        self.erp_queried=False
        self.negotiated=False

    def reset(self)->Observation:
        # Regenerate randomised task data on every reset
        self.task=generate_task(self.task_name,seed=self.seed)
        self.extracted_fields={}
        self.flags=[]
        self.decision=None
        self.current_step=0
        self.done=False
        self.total_reward=0.0
        self.email_content=None
        self.invoice_text=None
        self.erp_response=None
        self.erp_queried=False
        self.negotiated=False
        
        return self._get_obs("New episode started. You have unread emails in your inbox.")

    def state(self)->Dict[str,Any]:
        return {
            "task_name":self.task_name,
            "extracted_fields":self.extracted_fields,
            "flags":self.flags,
            "decision":self.decision,
            "current_step":self.current_step,
            "done":self.done,
            "total_reward":self.total_reward,
            "erp_queried":self.erp_queried
        }

    def step(self,action:Action)->StepResult:
        if self.done:
            return StepResult(
                observation=self._get_obs("Episode already done."),
                reward=0.0,
                done=True,
                info={"error":"Episode finished"}
            )

        self.current_step+=1
        reward=0.0
        message=""

        if action.action_type=="read_email":
            if action.email_id:
                found = next((e for e in self.task["emails"] if e["id"] == action.email_id), None)
                if found:
                    self.email_content = found["body"]
                    self.invoice_text = found["body"] # alias for backwards compatibility
                    reward=0.05
                    message=f"Email {action.email_id} opened."
                else:
                    reward=-0.02
                    message=f"Email ID {action.email_id} not found."
            else:
                reward=-0.01
                message="read_email requires email_id"

        elif action.action_type=="query_erp":
            if action.api_endpoint == "/api/v1/po" or action.api_endpoint == "/api/v2/po":
                if action.api_payload:
                    schema = self.task.get("erp_schema", {})
                    req_key = schema.get("required_key", "vendor_name")
                    
                    if req_key not in action.api_payload:
                        reward=-0.05
                        err_msg = schema.get("message", f"Bad Request: Missing {req_key}")
                        self.erp_response = {"error": err_msg}
                        message="ERP Query Failed due to Schema Mismatch."
                    else:
                        query_val = action.api_payload[req_key]
                        db = self.task["erp_database"]
                        if query_val in db:
                            self.erp_response = db[query_val]
                            self.erp_queried=True
                            reward=0.1
                            message="ERP Query Successful. PO data retrieved."
                        else:
                            self.erp_response = {"error": "Vendor not found."}
                            reward=-0.02
                            message="ERP Query returned no results."
                else:
                    reward=-0.01
                    message="query_erp requires api_payload"
            else:
                reward=-0.01
                message="Unknown ERP api_endpoint"

        elif action.action_type=="extract":
            if action.field_name and action.field_value is not None:
                if not self.email_content:
                    reward=-0.05
                    message="Cannot extract fields without reading the email first."
                else:
                    self.extracted_fields[action.field_name]=action.field_value
                    gt=self.task["ground_truth"]
                    if action.field_name in gt:
                        expected=gt[action.field_name]
                        if isinstance(expected,float):
                            try:
                                val = float(action.field_value)
                                if abs(val-expected)<0.01:
                                    reward=0.07
                                    message=f"Correct extraction of {action.field_name}"
                                else:
                                    reward=-0.02
                                    message=f"Wrong value for {action.field_name}"
                            except (ValueError, TypeError):
                                reward=-0.02
                                message=f"Wrong value for {action.field_name} (expected float)"
                        else:
                            if str(action.field_value).strip()==str(expected).strip():
                                reward=0.07
                                message=f"Correct extraction of {action.field_name}"
                            else:
                                reward=-0.02
                                message=f"Wrong value for {action.field_name}"
                    else:
                        message=f"Extracted {action.field_name}"
            else:
                reward=-0.01
                message="Extract action missing field_name or field_value"

        elif action.action_type=="flag":
            if action.field_name:
                if action.field_name not in self.flags:
                    self.flags.append(action.field_name)
                expected_flags=self.task.get("expected_flags",[])
                if action.field_name in expected_flags:
                    reward=0.12
                    message=f"Correct flag raised: {action.field_name}"
                else:
                    reward=-0.05
                    message=f"Incorrect flag: {action.field_name}"
            else:
                reward=-0.01
                message="Flag action missing field_name"

        elif action.action_type=="match_po":
            if not self.erp_response or "po_number" not in self.erp_response:
                reward=-0.05
                message="Cannot match PO before successfully querying ERP."
            else:
                po=self.erp_response
                total=self.extracted_fields.get("total_amount",None)
                if total is not None:
                    if abs(float(total)-po["approved_amount"])<0.01:
                        reward=0.1
                        message="PO match successful"
                    else:
                        reward=0.05
                        message="PO amount mismatch detected"
                else:
                    reward=-0.01
                    message="Extract total_amount before matching PO"

        elif action.action_type in ("approve","reject"):
            self.decision=action.action_type
            final_score=grade_task(
                self.task_name,
                self.extracted_fields,
                self.flags,
                self.decision,
                self.erp_queried,
                self.negotiated,
                task_dict=self.task
            )
            reward=final_score
            self.done=True
            message=f"Episode complete. Final score: {final_score}"

        elif action.action_type=="send_email":
            if not action.email_id:
                reward=-0.01
                message="send_email requires email_id (target address)"
            else:
                target = action.email_id
                sim_resp = self.task.get("simulated_responses", {})
                if sim_resp.get("trigger_action") == "send_email" and sim_resp.get("trigger_target") == target:
                    new_email = sim_resp["response_email"]
                    if not any(e["id"] == new_email["id"] for e in self.task["emails"]):
                        self.task["emails"].append(new_email)
                    self.negotiated = True
                    reward = 0.2
                    message = f"Email sent to {target}. A new email just arrived in your inbox!"
                else:
                    reward = -0.05
                    message = f"Email sent to {target}, but no reply was received."

        elif action.action_type=="match_duplicate":
            prev=self.task.get("previously_processed",[])
            inv_num=self.extracted_fields.get("invoice_number","")
            if inv_num in prev:
                if "duplicate_invoice" not in self.flags:
                    self.flags.append("duplicate_invoice")
                reward=0.12
                message="Duplicate invoice detected correctly"
            else:
                reward=-0.05
                message="No duplicate found"
        else:
            reward=-0.01
            message=f"Unknown action type: {action.action_type}"

        self.total_reward+=reward

        # Expose final_score in info on terminal steps
        step_final_score = None

        # force end at step 30
        if self.current_step>=30 and not self.done:
            self.done=True
            step_final_score=grade_task(
                self.task_name,
                self.extracted_fields,
                self.flags,
                self.decision or "",
                self.erp_queried,
                self.negotiated,
                task_dict=self.task
            )
            message=f"Max steps reached. Final score: {step_final_score}"

        # Also expose final_score from approve/reject
        if action.action_type in ("approve","reject") and self.done:
            step_final_score = reward  # reward IS final_score for terminal actions

        return StepResult(
            observation=self._get_obs(message),
            reward=round(reward,2),
            done=self.done,
            info={"step":self.current_step,
                  "total_reward":round(self.total_reward,2),
                  "final_score":step_final_score}
        )

    def _get_obs(self,message:str)->Observation:
        inbox_status = [{"id": e["id"], "sender": e["sender"], "subject": e["subject"]} for e in self.task["emails"]]
        return Observation(
            inbox_status=inbox_status,
            email_content=self.email_content,
            invoice_text=self.invoice_text,
            erp_response=self.erp_response,
            extracted_fields=self.extracted_fields,
            flags=self.flags,
            current_step=self.current_step,
            message=message
        )