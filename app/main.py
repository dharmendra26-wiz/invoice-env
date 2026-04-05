from fastapi import FastAPI,HTTPException
from app.models import Action,StepResult,Observation
from app.environment import InvoiceEnvironment
from typing import Dict,Any
import uvicorn

app=FastAPI(title="Invoice Processing Environment",version="1.0.0")

envs:Dict[str,InvoiceEnvironment]={}

@app.get("/")
def root():
    return {"name":"invoice-env","version":"1.0.0","status":"running"}

@app.post("/reset")
def reset(task_name:str="easy")->Observation:
    env=InvoiceEnvironment(task_name=task_name)
    envs[task_name]=env
    obs=env.reset()
    return obs

@app.post("/step")
def step(task_name:str,action:Action)->StepResult:
    if task_name not in envs:
        raise HTTPException(status_code=404,detail=f"No active env for task '{task_name}'. Call /reset first.")
    env=envs[task_name]
    result=env.step(action)
    return result

@app.get("/state")
def state(task_name:str="easy")->Dict[str,Any]:
    if task_name not in envs:
        raise HTTPException(status_code=404,detail=f"No active env for task '{task_name}'. Call /reset first.")
    return envs[task_name].state()

@app.get("/tasks")
def list_tasks():
    return {
        "tasks":[
            {"name":"easy","description":"Extract fields from clean invoice and match PO"},
            {"name":"medium","description":"Detect line item price mismatch"},
            {"name":"hard","description":"Detect duplicate invoice and tax miscalculation"}
        ]
    }

@app.get("/health")
def health():
    return {"status":"healthy"}

if __name__=="__main__":
    uvicorn.run("app.main:app",host="0.0.0.0",port=7860,reload=False)