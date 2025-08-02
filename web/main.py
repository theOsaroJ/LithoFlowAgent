from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.agent import run_agent

app = FastAPI(title="LithoFlowAgent API")

class Query(BaseModel):
    prompt: str

@app.post("/ask")
async def ask(query: Query):
    try:
        response = run_agent(query.prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
