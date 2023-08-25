"""
The agent is implemented as Zero Shot ReAct one. Prompt-engineering is used for adding the tools.
Memory is added in form of a conversational buffer type, such that the agent can remember the chat history and perform follow-up queries. 
"""
from langchain.agents import AgentExecutor, AgentType, initialize_agent

from .config import Logger
from .model import chat_history, llm, memory
from .tools.tools import cdsapi_tool, chart_tool, meteogram_tool, retriever_tools

agent_tools = retriever_tools + meteogram_tool + chart_tool + cdsapi_tool

Logger.info("Initialising main agent...")
agent = initialize_agent(
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=agent_tools,
    llm=llm,
    verbose=True,
    memory=memory,
    agent_kwargs={
        "memory_prompts": [chat_history],
        "input_variables": ["input", "agent_scratchpad", "chat_history"],
    },
    max_iterations=5,
    return_intermediate_steps=True,
    get_chat_history=lambda h: h,
)
