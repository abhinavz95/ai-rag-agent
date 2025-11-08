LangGraph RAG Agent with Gemini and ChromaDB

This project implements a basic Retrieval-Augmented Generation (RAG) AI Agent using LangGraph to define a clear, multi-step workflow.

Setup and Execution

1. Clone the repository and navigate to the directory (if applicable).

2. Install dependencies:

pip install -r requirements.txt


3. Configure your API Key:

Create a file named .env in the project root directory and add your Gemini API Key:

GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"


4. Run the RAG Agent:

python rag_agent.py


The script will first set up the Chroma vector store using knowledge_base.txt, then run an example question through the 4-node LangGraph workflow, printing the output of each step.

Agent Workflow Description (LangGraph Nodes)

The agent uses a StateGraph to manage the flow of information across the following nodes:

plan: Interprets the user's initial query. In this basic implementation, it is configured to always decide to proceed with RAG, but it demonstrates the point of intervention for more complex, tool-using agents.

retrieve: Uses the question to query the local ChromaDB vector store. It retrieves the most relevant document chunks from the knowledge_base.txt and compiles them into a context string in the state.

answer: Takes the question and the context and passes them to the Gemini LLM (gemini-2.5-flash) via a prompt template to generate the final, grounded response.

reflect: This validation step uses the LLM again as a Judge. It receives the original question, the retrieved context, and the answer, and evaluates if the answer is relevant and complete based on the context. The node outputs a reflection summary and a final 'PASS' or 'FAIL' decision. The agent then terminates.

Challenges Faced & Approach

The primary challenge was ensuring the reflect step was meaningful without introducing overly complex retry logic (which was excluded for simplicity as per the core requirements).

Approach to Reflection: I used a simple, single LLM call for reflection where the LLM is instructed to act as an "internal validator." This mimics the "LLM as a Judge" evaluation pattern. The prompt explicitly asks the LLM to output a PASS or FAIL keyword to enable potential future conditional routing (e.g., retrying the retrieval or regenerating the answer with a different prompt). This validation step successfully demonstrates the agent's ability to self-assess the quality of its output against the source materials.