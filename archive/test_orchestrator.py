from orchestrator import OrchestratorAgent

with open("openai_api_key.txt", "r") as file:
    openai_api_key = file.read().strip()

# test_agent = OrchestratorAgent(openai_api_key="openai_api_key")
# test_agent = OrchestratorAgent(openai_api_key="")
test_agent = OrchestratorAgent(client="Ollama")
plan = test_agent.process_message("What's the regulatory status of AI in medical devices?")

print("History")
for msg in test_agent.message_history:
    print(msg)
    print()

print("Action Plan:")
print(plan)