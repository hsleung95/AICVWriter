from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from duckduckgo_search import DDGS
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.agents import AgentFinish
from langgraph.graph import END, StateGraph
from langchain.tools import tool
from langgraph.prebuilt.tool_executor import ToolExecutor
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
from PyPDF2 import PdfReader
import pdfkit
import functools, operator, requests, os, json
# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "AI CV Review"

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", streaming=True)

dir_path = ""

@tool("ask_candidate", return_direct=False)
def ask_candidate(query: str) -> str:
    """Ask candidate for provided query"""
    return input(query)

@tool("read_cv",return_direct=False)
def read_cv(filename):
    """read cv content from pdf path provided by the candidate, do not modify this file"""
    text = ""
    reader = PdfReader(dir_path + filename)
    for page in reader.pages:
        text += page.extract_text()
    return text

@tool("write_cv", return_direct=False)
def write_cv(data, filename):
    """write result cv as a pdf, input should be a html string and the filename"""
    new_path = dir_path + filename
    if not os.path.exists(new_path):
        os.mknod(new_path)
    pdfkit.from_string(data, new_path)

tools = [ask_candidate, read_cv, write_cv]

# 2. Agents 
# Helper function for creating agents
def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

# Define agent nodes
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

# Create Agent Supervisor
members = ["Candidate_interviewer", "CV_summerizor"]
system_prompt = (
    """As a experienced executive recruiter, your role is to help candidate to build a more outlined CV, 
    first you should ask for a CV or a general introduction about the candidate's job history or related experience, 
    then letting Candidate_interviewer to ask related questions to dig out more.
    Once enough information is gathered, assign CV_summerizor to generate a refined CV with a summary and job histories,
    finally indicate with 'FINISH'."""
)

options = ["FINISH"] + members
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}] }},
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
]).partial(options=str(options), members=", ".join(members))

supervisor_chain = (prompt | llm.bind_functions(functions=[function_def], function_call="route") | JsonOutputFunctionsParser())

candidate_interviewer = create_agent(llm, tools, 
        """You are a candidate interviewer. Interview with the candidate to gather more information based on questions provided
        """)
candidate_interviewer_node = functools.partial(agent_node, agent=candidate_interviewer, name="Candidate_interviewer")

cv_summerizor = create_agent(llm, [ask_candidate, write_cv], 
        """You are a experienced executive recruit CV summerizor, base on information provided by supervisor write a refined CV,
        file name should be cadidate's name _CV and is in pdf,
        start with candidate's name and latest title,
        then better outlined summary, job history, education, skills and languages, end with contact information,
        keep within 1 page. Your summarization should enhance on candidate's achievements, tech skills or soft skills, project scale etc""")
cv_summerizor_node = functools.partial(agent_node, agent=cv_summerizor, name="CV_summerizor")

supervisor_assistant = create_agent(llm, [ask_candidate, read_cv], 
        "You are a supervisor assistant, you ask the candidate for a CV or summary of job history")
supervisor_assistant_node = functools.partial(agent_node, agent=supervisor_assistant, name="Supervisor_assistant")
# Define the Agent State, Edges and Graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

workflow = StateGraph(AgentState)
workflow.add_node("Candidate_interviewer", candidate_interviewer_node)
workflow.add_node("CV_summerizor", cv_summerizor_node)
workflow.add_node("Supervisor_assistant", supervisor_assistant_node)
workflow.add_node("supervisor", supervisor_chain)

# Define edges
for member in members:
    workflow.add_edge(member, "supervisor")
workflow.add_edge("Supervisor_assistant", "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("Supervisor_assistant")

graph = workflow.compile()

# Run the graph
for s in graph.stream({
    "messages": [HumanMessage(content="")]
}):
    if "__end__" not in s:
        print(s)
        print("----")

