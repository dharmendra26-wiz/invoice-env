from typing import Dict,Any,Optional
from app.models import Action,Observation,Reward,StepResult
from app.tasks import TASKS,grade_task

class InvoiceEnvironment:
    def __init__(self,task_name:str="easy"):
        self.task_name=task_name
        self.task=TASKS[task_name]
        self.extracted_fields:Dict[str,Any]={}
        self.flags=[]
        self.decision:Optional[str]=None
        self.current_step=0
        self.done=False
        self.total_reward=0.0

    def reset(self)->Observation:
        self.extracted_fields={}
        self.flags=[]
        self.decision=None
        self.current_step=0
        self.done=False
        self.total_reward=0.0
        return Observation(
            invoice_text=self.task["invoice_text"],
            po_data=self.task["po_data"],
            extracted_fields={},
            flags=[],
            current_step=0,
            message="New episode started. Read the invoice and extract fields."
        )

    def state(self)->Dict[str,Any]:
        return {
            "task_name":self.task_name,
            "extracted_fields":self.extracted_fields,
            "flags":self.flags,
            "decision":self.decision,
            "current_step":self.current_step,
            "done":self.done,
            "total_reward":self.total_reward
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

        if action.action_type=="extract":
            if action.field_name and action.field_value is not None:
                self.extracted_fields[action.field_name]=action.field_value
                gt=self.task["ground_truth"]
                if action.field_name in gt:
                    expected=gt[action.field_name]
                    if isinstance(expected,float):
                        if abs(float(action.field_value)-expected)<0.01:
                            reward=0.07
                            message=f"Correct extraction of {action.field_name}"
                        else:
                            reward=-0.02
                            message=f"Wrong value for {action.field_name}"
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
            po=self.task["po_data"]
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
                self.decision
            )
            reward=final_score
            self.done=True
            message=f"Episode complete. Final score: {final_score}"

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

        # force end at step 20
        if self.current_step>=20 and not self.done:
            self.done=True
            final_score=grade_task(
                self.task_name,
                self.extracted_fields,
                self.flags,
                self.decision or ""
            )
            message=f"Max steps reached. Final score: {final_score}"

        return StepResult(
            observation=self._get_obs(message),
            reward=round(reward,2),
            done=self.done,
            info={"step":self.current_step,"total_reward":round(self.total_reward,2)}
        )

    def _get_obs(self,message:str)->Observation:
        return Observation(
            invoice_text=self.task["invoice_text"],
            po_data=self.task["po_data"],
            extracted_fields=self.extracted_fields,
            flags=self.flags,
            current_step=self.current_step,
            message=message
        )