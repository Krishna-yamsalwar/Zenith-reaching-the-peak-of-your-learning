# chatbot.py

# --- 1. Environment and API Key Setup ---
import os
from dotenv import load_dotenv

# --- All other necessary imports from your original script ---
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

def get_graph():
    """
    This function encapsulates the entire graph creation logic.
    It returns a compiled LangGraph instance.
    """
    load_dotenv()
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not tavily_api_key or not groq_api_key:
        print("Warning: API keys not found. The script may fail.")

    # --- 3. Tool Definitions ---
    api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
    arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Searches for and returns summaries of papers from Arxiv.")
    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
    wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki, description="Searches Wikipedia for information.")
    tavily_tool = TavilySearch(max_results=3, description="A search engine for finding up-to-date information.")

    def calculator(expression: str) -> str:
        """Evaluates a mathematical expression or solves a single variable linear equation."""
        try:
            if '=' in expression:
                from sympy import sympify, solve
                if 'x' in expression and expression.count('x') == 1:
                    parts = expression.split('=')
                    lhs = sympify(parts[0])
                    rhs = sympify(parts[1])
                    solution = solve(lhs - rhs)
                    result = f"The solution to '{expression}' is x = {solution[0]}."
                else:
                    result = "Calculator can only solve single variable linear equations."
            else:
                result = eval(expression, {"__builtins__": None}, {})
            return f"The result of '{expression}' is {result}."
        except Exception as e:
            return f"Failed to evaluate the expression. Error: {e}"

    calculator_tool = StructuredTool.from_function(func=calculator, name="Calculator", description="Evaluates a mathematical expression or solves a single variable linear equation.")

    original_agent_tools = [arxiv_tool, wikipedia_tool, tavily_tool, calculator_tool]
    technical_tools = [arxiv_tool, wikipedia_tool]
    math_tools = [calculator_tool, tavily_tool]
    general_tools = [tavily_tool, wikipedia_tool]

    # --- 4. LLM and Agent Setup ---
    llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key, temperature=0.2)
    original_llm_with_tools = llm.bind_tools(tools=original_agent_tools)
    technical_llm_with_tools = llm.bind_tools(tools=technical_tools)
    math_llm_with_tools = llm.bind_tools(tools=math_tools)
    general_llm_with_tools = llm.bind_tools(tools=general_tools)

    # --- 5. Graph State and Router Definition ---
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], lambda x, y: x + y]

    def master_router(state: State) -> Literal["original_agent", "technical_agent", "math_agent", "general_agent"]:
        """Determines the most appropriate agent to handle the user's query."""
        last_message = state["messages"][-1].content
        router_prompt = f"""You are an expert at routing a user's query to the correct agent.
        You have four agents available:
        1.  **original_agent**: A generalist agent for simple, conversational queries.
        2.  **technical_agent**: A specialist for complex questions about computer science, programming, or scientific papers.
        3.  **math_agent**: A specialist for questions involving mathematical calculations, formulas, or solving equations.
        4.  **general_agent**: A specialist for other specific questions that require deep research using web search or Wikipedia.
        Based on the user's query below, choose the most appropriate agent.
        User Query: "{last_message}"
        Chosen Agent:"""
        response = llm.invoke(router_prompt)
        choice = response.content.strip().lower()
        if "technical_agent" in choice: return "technical_agent"
        if "math_agent" in choice: return "math_agent"
        if "general_agent" in choice: return "general_agent"
        return "original_agent"

    # --- 6. Agent Node Definitions ---
    def agent_node(state: State, llm_with_tools, agent_name: str):
        """Generic function to run an agent's logic."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful assistant, acting as a specialized {agent_name}. You talk like a friendly buddy. Answer all questions to the best of your ability. Include a 😄 emoji in your responses."),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain = prompt | llm_with_tools
        response = chain.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    def original_agent_node(state: State): return agent_node(state, original_llm_with_tools, "Original Agent")
    def technical_agent_node(state: State): return agent_node(state, technical_llm_with_tools, "Technical Specialist")
    def math_agent_node(state: State): return agent_node(state, math_llm_with_tools, "Math Specialist")
    def general_agent_node(state: State): return agent_node(state, general_llm_with_tools, "General Specialist")

    # --- 7. Graph Construction ---
    original_tool_node = ToolNode(original_agent_tools)
    technical_tool_node = ToolNode(technical_tools)
    math_tool_node = ToolNode(math_tools)
    general_tool_node = ToolNode(general_tools)

    def should_continue(state: State) -> Literal["tools", "__end__"]:
        if isinstance(state["messages"][-1], AIMessage) and state["messages"][-1].tool_calls:
            return "tools"
        return "__end__"

    builder = StateGraph(State)
    builder.add_conditional_edges(START, master_router, {
        "original_agent": "original_agent", "technical_agent": "technical_agent",
        "math_agent": "math_agent", "general_agent": "general_agent",
    })
    builder.add_node("original_agent", original_agent_node)
    builder.add_node("technical_agent", technical_agent_node)
    builder.add_node("math_agent", math_agent_node)
    builder.add_node("general_agent", general_agent_node)
    builder.add_conditional_edges("original_agent", should_continue, {"tools": "original_tools", "__end__": END})
    builder.add_conditional_edges("technical_agent", should_continue, {"tools": "technical_tools", "__end__": END})
    builder.add_conditional_edges("math_agent", should_continue, {"tools": "math_tools", "__end__": END})
    builder.add_conditional_edges("general_agent", should_continue, {"tools": "general_tools", "__end__": END})
    builder.add_node("original_tools", original_tool_node)
    builder.add_node("technical_tools", technical_tool_node)
    builder.add_node("math_tools", math_tool_node)
    builder.add_node("general_tools", general_tool_node)
    builder.add_edge("original_tools", "original_agent")
    builder.add_edge("technical_tools", "technical_agent")
    builder.add_edge("math_tools", "math_agent")
    builder.add_edge("general_tools", "general_agent")
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
