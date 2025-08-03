from orchestrator import OrchestratorAgent
from rag import RAGAgent


with open("openai_api_key.txt", "r") as file:
    openai_api_key = file.read().strip()

test_agent = OrchestratorAgent(openai_api_key="openai_api_key")
# test_agent = OrchestratorAgent(openai_api_key="")
action_plan = test_agent.process_message("What's the regulatory status of AI in medical devices?")

print("History")
for msg in test_agent.message_history:
    print(msg)
print("Done")

print("Action Plan:")
print(action_plan)

# RAG
rag_agent = RAGAgent(chroma_db_path="./chroma_db")
print("Retrieved Documents:")
top3 = rag_agent.search_documents("AI in medical devices", top_k=3)
print(top3)

print("Processing action plan with RAG agent...")
response = rag_agent.process_message(action_plan)
print(response)