import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import the retriever function from the setup file
try:
    from rag_setup import get_retriever
except ImportError:
    print("Error: rag_setup.py not found or has errors.")
    sys.exit(1)

# --- Configuration ---
OLLAMA_MODEL = 'gemma3'

# 1. Define the Prompt Template
# This is the system prompt that tells the LLM how to behave and how to use the context.
prompt_template = """You are a highly skilled professional CV writer and analyst. Your goal is to generate a comprehensive 
response based *only* on the provided context. When answering, if you use a specific piece of information, 
mention the source file it came from.

CONTEXT (Note: Each chunk is preceded by its source file path, e.g., 'Source: data/skills.txt'):
{context}

QUESTION:
{question}

Response:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)



# 2. Initialize LLM and Retriever
try:
    # Initialize Ollama LLM
    llm = Ollama(model=OLLAMA_MODEL)
    
    # Get the ready-to-use retriever (loads the saved index)
    retriever = get_retriever()
except Exception as e:
    print(f"Error initializing RAG components. Ensure Ollama is running and models are pulled. Error: {e}")
    sys.exit(1)

# 3. Define the formatting function for context documents
# This function formats the retrieved document chunks into a single string for the prompt.
def format_docs_with_source(docs):
    formatted_chunks = []
    for doc in docs:
        # Extract the source file path from metadata
        source = doc.metadata.get('source', 'Unknown Source')
        # Format the content to include the source
        formatted_chunks.append(f"Source: {source}\nContent: {doc.page_content}")
    return "\n\n".join(formatted_chunks)

# 4. Build the RAG Chain with Source Attribution

# A. Define a retrieval-only chain (takes question, returns documents)
retrieval_chain = (
    RunnablePassthrough.assign(
        # The retriever runs first, returning a list of documents
        documents=retriever,
        # The question is passed through unchanged
        question=RunnablePassthrough()
    )
)

# B. Define the generation chain (takes retrieved documents, formats them, and calls the LLM)
rag_chain = retrieval_chain.assign(
    answer=RunnablePassthrough.assign(
        # Format the retrieved documents for the LLM prompt
        context=lambda x: format_docs_with_source(x['documents']),
    )
    | prompt
    | llm
    | StrOutputParser()
)


# --- Chat Loop ---
print("\n--- LangChain RAG Chat (CV Generator) ---")
print(f"LLM Model: {OLLAMA_MODEL} | Retriever Index: chroma_db_career")
print("Type 'generate cv' to produce the CV, or ask a question about your summary. Type 'exit' to quit.")

# Your specific CV generation task prompt
CV_PROMPT = "Generate a full, professional Curriculum Vitae (CV) for a Senior Developer based on the career summary provided. Include sections for Experience, Education, and Key Skills."

# --- Chat Loop ---
while True:
    user_input = input('\nRAG Chat: ')
    if user_input.lower() == 'exit':
        break

    final_query = CV_PROMPT if "generate cv" in user_input.lower() else user_input
    input_dict = {"question": final_query}
    # ❗ Use invoke() instead of stream() for easier source display
    # The chain returns a dictionary: {'documents': [...], 'answer': '...'}
    result = rag_chain.invoke(input_dict)

    # --- Display the results ---
    print("\n\n--- LLM Response ---")
    print(result['answer'])
    print("--------------------")

    # --- Display the Sources ---
    print("\n--- Sources Used for Retrieval ---")
    
    # Use a set to display unique source files
    unique_sources = set(doc.metadata.get('source', 'Unknown Source') for doc in result['documents'])
    
    for source in unique_sources:
        print(f"-> {source}")
    print("----------------------------------\n")