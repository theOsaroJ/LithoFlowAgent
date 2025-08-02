from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from agent.agent import run_agent

app = FastAPI(title="LithoFlowAgent")

app.mount("/static", StaticFiles(directory="web/static"), name="static")

class AskRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("web/static/index.html") as f:
        return f.read()

@app.post("/ask")
async def ask(req: AskRequest):
    try:
        return {"response": run_agent(req.prompt)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status":"ok"}
