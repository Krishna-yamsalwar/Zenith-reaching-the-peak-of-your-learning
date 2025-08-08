# chatbot.py

# --- 1. Environment and API Key Setup ---
import os
from dotenv import load_dotenv

# --- All other necessary imports ---
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
    api_wrapper_arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Searches for and returns summaries of papers from Arxiv.")
    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
    wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wiki, description="Searches Wikipedia for information.")
    tavily_tool = TavilySearch(max_results=3, description="A search engine for finding up-to-date information.")

    def calculator(expression: str) -> str:
        """
        Solves single-variable equations, including linear and quadratic ones.
        The user's query is passed directly to this tool.
        Use standard Python syntax for equations, e.g., '14*x**2 + 20*x + 3 = 0'.
        """
        try:
            from sympy import sympify, solve, symbols, Eq
            from sympy.core.sympify import SympifyError

            if '=' in expression:
                expression = expression.replace('x*x', '*x**2').replace('^', '**')
                lhs, rhs = expression.split('=')
                x = symbols('x')
                lhs_expr = sympify(lhs.strip())
                rhs_expr = sympify(rhs.strip())
                equation = Eq(lhs_expr, rhs_expr)
                solutions = solve(equation, x)
                if not solutions:
                    return "No real solution found for the equation."
                formatted_solutions = [f"{s.evalf(4)}" for s in solutions]
                return f"The solutions are: x = {', '.join(formatted_solutions)}"
            else:
                result = eval(expression, {"__builtins__": None}, {})
                return f"The result of '{expression}' is {result}."
        except (SympifyError, NameError, SyntaxError, Exception) as e:
            return f"Could not process the expression. Please check the format. Error: {str(e)}"

    calculator_tool = StructuredTool.from_function(func=calculator, name="Calculator", description="Solves mathematical equations, including quadratic ones.")

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
    
    # FIX: Replaced LLM router with a more robust, deterministic keyword-based router
    def master_router(state: State) -> Literal["original_agent", "technical_agent", "math_agent", "general_agent"]:
        """Determines the most appropriate agent based on keywords in the user's query."""
        last_message = state["messages"][-1].content.lower()
        
        # 1. Prioritize math for any query with numbers or explicit math terms
        math_keywords = ['solve', 'calculate', 'equation', 'math', '+', '-', '*', '/', '=', 'x']
        if any(keyword in last_message for keyword in math_keywords) or any(char.isdigit() for char in last_message):
            print("--- Routing to Math Specialist ---")
            return "math_agent"
            
        # 2. Route to technical agent for specific tech terms
        tech_keywords = ['code', 'python', 'javascript', 'computer science', 'algorithm', 'arxiv', 'paper', 'ml', 'ai', 'neural network']
        if any(keyword in last_message for keyword in tech_keywords):
            print("--- Routing to Technical Specialist ---")
            return "technical_agent"
            
        # 3. Route to general agent for research-style questions
        general_keywords = ['who is', 'what is', 'what are', 'explain', 'summary of', 'tell me about']
        if any(keyword in last_message for keyword in general_keywords):
            print("--- Routing to General Specialist ---")
            return "general_agent"
            
        # 4. Default to the original agent for everything else (simple conversation)
        print("--- Routing to Original Agent ---")
        return "original_agent"

    # --- 6. Agent Node Definitions ---
    def agent_node(state: State, llm_with_tools, agent_name: str):
        """Generic function to run an agent's logic."""
        # FIX: Updated system prompt to be more direct and avoid "thinking out loud"
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful assistant, acting as a specialized {agent_name}. You talk like a friendly buddy. You must follow these rules: 1. If the user's query can be answered with a tool, call the tool immediately. Do not explain that you are using a tool. 2. If no tool is needed, provide a direct answer to the user's question without any introductory phrases about your process. 3. Always include a 😄 emoji in your final response."),
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
        """Decides whether to call a tool or end the turn."""
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
