import openai
from langchain import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from config import OPENAI_API_KEY
from agent.tools import IngestLogsTool, ImputeLogsTool, ForecastTool, RetrieveDocsTool

# Configure OpenAI client
openai.api_key = OPENAI_API_KEY

llm = OpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.2
)

tools = [
    Tool.from_function(IngestLogsTool()),
    Tool.from_function(ImputeLogsTool()),
    Tool.from_function(ForecastTool()),
    Tool.from_function(RetrieveDocsTool()),
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.REACT_DESCRIPTION,
    verbose=True
)

def run_agent(prompt: str) -> str:
    """Run the agent on a user prompt."""
    return agent.run(prompt)
