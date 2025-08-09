from orchestrator import OrchestratorAgent
from rag import RAGAgent
from response import ResponseAgent
from validation import ValidationAgent
from web_search import google_search

# Read API keys and CX from external files
with open("openai_api_key.txt", "r") as file:
    openai_api_key = file.read().strip()

# Read Google API key and CX from a single file
with open("google_config.txt", "r") as file:
    google_api_key, google_cx = [line.strip() for line in file.readlines()]

# query = input("Enter your query: ")
query = "Compare FDA and WHO approaches to risk management for medical devices" 

print("Step 1: Generating action plan...")
first_agent = OrchestratorAgent(openai_api_key=openai_api_key)
action_plan = first_agent.process_message(query)
print(action_plan)

print("Step 2: Processing action plan and database with RAG agent...")
rag_agent = RAGAgent(chroma_db_path="./chroma_db")
rag_response = rag_agent.process_message(action_plan)
print(rag_response)

print("Step 3: Generating response...")
rep_agent = ResponseAgent(openai_api_key=openai_api_key)
response = rep_agent.process_message(rag_response)
print(response)

print("Step 4: Validating response...")
val_agent = ValidationAgent(openai_api_key=openai_api_key)
val_response = val_agent.process_message(response)
print(val_response)

print("Step 5: Performing web search...")
web_results = google_search(query + " medical device regulatory", google_api_key, google_cx)
print(web_results)

print("All steps completed.")
