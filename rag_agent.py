import os
import json
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

# --- 0. Environment Setup and Data Loading ---

# Load environment variables from .env file (for GEMINI_API_KEY)
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found. Please create a .env file.")

# Define the data file and vector store path
DATA_FILE = "knowledge_base.txt"
VECTOR_DB_PATH = "./chroma_db"
LLM_MODEL = "gemini-2.5-flash"

# Initialize LLM and Embeddings
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- 1. RAG Setup: Create/Load Vector Database ---

def setup_chroma_db(data_path: str, db_path: str):
    """Loads documents, splits them, and creates a Chroma vector store."""
    print("--- 1. RAG Setup: Loading and processing documents...")
    try:
        # Load data
        loader = TextLoader(data_path)
        documents = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Create/load vector store
        db = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
        print(f"--- ChromaDB initialized with {len(chunks)} chunks at {db_path}.")
        return db.as_retriever(search_kwargs={"k": 3})
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please ensure it exists.")
        exit(1)

# Initialize the retriever
retriever = setup_chroma_db(DATA_FILE, VECTOR_DB_PATH)


# --- 2. Agent State Definition (TypedDict) ---

class AgentState(TypedDict):
    """
    Represents the state of the agent workflow.
    """
    question: str
    context: str
    answer: str
    reflection: str
    max_retries: int # For tracking reflection retries (Bonus)
    current_retry: int

# --- 3. LangGraph Nodes ---

def plan_query(state: AgentState) -> AgentState:
    """
    Interprets the user question and decides if retrieval is needed.
    
    For simplicity in this basic RAG, we always decide to retrieve, 
    but the node structure is in place.
    """
    question = state["question"]
    print(f"\n[Node: plan] Analyzing query: '{question}'...")
    
    # LLM check to decide if retrieval is necessary (e.g., if it's a general knowledge question)
    planning_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI planner. Analyze the user's question and determine if it requires information retrieval from a specialized knowledge base. If it's a simple, common knowledge question (e.g., 'What is the sky color?'), output 'NO_RAG'. If it needs external context, output 'RAG'. Only output 'RAG' or 'NO_RAG'."),
        ("user", f"Question: {question}")
    ])
    
    # We will simulate the output to always be 'RAG' to demonstrate the full pipeline
    # For a real implementation, you would uncomment the line below:
    # decision = llm.invoke(planning_prompt.format(question=question)).content.strip().upper()
    decision = "RAG" 
    
    print(f"    -> Decision: {decision}")
    
    # Store decision for conditional edge routing (though the router will be simple here)
    state["plan"] = decision.strip().upper()
    
    return state


def retrieve_context(state: AgentState) -> AgentState:
    """
    Performs RAG to retrieve relevant documents based on the question.
    """
    question = state["question"]
    print(f"\n[Node: retrieve] Retrieving context for: '{question}'...")
    
    # Retrieve documents
    docs = retriever.get_relevant_documents(question)
    
    # Format context string
    context = "\n---\n".join([doc.page_content for doc in docs])
    
    print(f"    -> Retrieved {len(docs)} document chunks. Snippet: '{context[:100]}...'")
    
    return {"context": context, "question": question}


def generate_answer(state: AgentState) -> AgentState:
    """
    Generates the final answer using the LLM, the question, and the retrieved context.
    """
    question = state["question"]
    context = state["context"]
    print("\n[Node: answer] Generating final response...")
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful Q&A assistant. Use the provided context to answer the user's question. If the context does not contain the answer, state that you couldn't find the information in the provided context. Be concise and professional."),
        ("user", f"Context: {context}\n\nQuestion: {question}")
    ])
    
    answer = llm.invoke(answer_prompt).content
    print(f"    -> Answer generated. Starting reflection step.")
    
    return {"question": question, "context": context, "answer": answer}


def reflect_and_validate(state: AgentState) -> AgentState:
    """
    The agent evaluates the generated answer for relevance and completeness.
    """
    question = state["question"]
    answer = state["answer"]
    context = state["context"]
    print("\n[Node: reflect] Reflecting on the generated answer...")
    
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an internal validator. Evaluate the generated answer against the original question and the retrieved context. Your goal is to assess if the answer is directly relevant and logically complete based *only* on the context provided. Output your evaluation and a final decision: 'PASS' if the answer is good, or 'FAIL' if it's irrelevant or incomplete."),
        ("user", f"Original Question: {question}\n\nRetrieved Context: {context}\n\nGenerated Answer: {answer}")
    ])
    
    reflection_response = llm.invoke(reflection_prompt).content
    
    # Simple check for the decision keyword
    decision = "PASS" if "PASS" in reflection_response.upper() else "FAIL"
    
    print(f"    -> Reflection Result: {decision}")
    print(f"    -> Full Reflection: {reflection_response[:150]}...")
    
    return {"answer": answer, "reflection": reflection_response, "reflection_decision": decision}

# --- 4. Define the Graph and Edges ---

# Define the workflow
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("plan", plan_query)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("answer", generate_answer)
workflow.add_node("reflect", reflect_and_validate)

# Set the start node
workflow.set_entry_point("plan")

# Define edges
# 1. After planning, always go to retrieval (in this basic implementation)
workflow.add_edge("plan", "retrieve")

# 2. After retrieval, go to generation
workflow.add_edge("retrieve", "answer")

# 3. After generation, go to reflection
workflow.add_edge("answer", "reflect")

# 4. After reflection, END (no retry logic for simplicity, per basic requirement)
workflow.add_edge("reflect", END)

# Compile the graph
app = workflow.compile()

# --- 5. Execution and Output ---

def run_agent(question: str):
    """Runs the LangGraph agent workflow for a given question."""
    
    initial_state = {"question": question, "context": "", "answer": "", "reflection": ""}
    
    print("=====================================================")
    print(f"Agent Start: Processing Question: {question}")
    print("=====================================================")
    
    # Run the agent and capture the final state
    final_state = app.invoke(initial_state)
    
    print("\n=====================================================")
    print("Agent Workflow Complete.")
    print("=====================================================")
    print("\n--- FINAL ANSWER ---")
    print(final_state["answer"])
    
    print("\n--- AGENT REFLECTION ---")
    print(final_state["reflection"])
    print("=====================================================")


if __name__ == "__main__":
    # Example Q&A run
    query = "What are the primary benefits of using solar power, and what is its main limitation compared to coal?"
    run_agent(query)
    
    # Another example to test retrieval
    query_2 = "How does geothermal energy work, and which type of energy is the most dangerous?"
    # run_agent(query_2)
    
    # Optional: Clean up the ChromaDB directory if needed
    # import shutil
    # shutil.rmtree(VECTOR_DB_PATH, ignore_errors=True)