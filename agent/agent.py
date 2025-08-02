import openai
from config import OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    IngestLogsTool,
    ImputeLogsTool,
    LithoClassifierTool,
    ForecastTool,
    RetrieveDocsTool,
)

openai.api_key = OPENAI_API_KEY

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY,
)

tools = [
    Tool.from_function(IngestLogsTool().run, name=IngestLogsTool.name, description=IngestLogsTool.description),
    Tool.from_function(ImputeLogsTool().run, name=ImputeLogsTool.name, description=ImputeLogsTool.description),
    Tool.from_function(LithoClassifierTool().run, name=LithoClassifierTool.name, description=LithoClassifierTool.description),
    Tool.from_function(ForecastTool().run, name=ForecastTool.name, description=ForecastTool.description),
    Tool.from_function(RetrieveDocsTool().run, name=RetrieveDocsTool.name, description=RetrieveDocsTool.description),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    system_message=SYSTEM_PROMPT
)

def run_agent(prompt: str) -> str:
    return agent.run(prompt)
