from orchestrator import OrchestratorAgent
from rag import RAGAgent
from response import ResponseAgent

with open("openai_api_key.txt", "r") as file:
    openai_api_key = file.read().strip()

test_agent = OrchestratorAgent(openai_api_key=openai_api_key)
# test_agent = OrchestratorAgent(openai_api_key="")
action_plan = test_agent.process_message("Is AI regulated in medtech?")

# print("History")
# for msg in test_agent.message_history:
#     print(msg)
# print("Done")

print("Action Plan:")
print(action_plan)

# RAG
rag_agent = RAGAgent(chroma_db_path="./chroma_db")
# print("Retrieved Documents:")
# top3 = rag_agent.search_documents("AI in medical devices", top_k=3)
# print(top3)

print("Processing action plan with RAG agent...")
response = rag_agent.process_message(action_plan)
print(response)

# rep_agent = ResponseAgent(client="OpenAI", openai_api_key="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")
# rep_agent = ResponseAgent(client="HF")
rep_agent = ResponseAgent(client="Ollama")
print("Generating final response...")
final_response = rep_agent.process_message(response)
print(final_response)