import streamlit as st
from orchestrator import OrchestratorAgent
from rag import RAGAgent
from response import ResponseAgent
from validation import ValidationAgent
from web_search import google_search

st.set_page_config(layout="wide")
st.title("MedTech AI Regulatory Assistant")
st.write("Ask questions about AI regulations in medical technology.")

with open("openai_api_key.txt", "r") as file:
    openai_api_key = file.read().strip()

with open("google_config.txt", "r") as file:
    google_api_key, google_cx = [line.strip() for line in file.readlines()]

orchestrator = OrchestratorAgent(openai_api_key=openai_api_key)
rag_agent = RAGAgent(chroma_db_path="./chroma_db")
rep_agent = ResponseAgent(openai_api_key=openai_api_key)
val_agent = ValidationAgent(openai_api_key=openai_api_key)

# What are the design control requirements for verificationand validation?
query = st.text_input("Enter your query:", "Is our AI-powered MRI analysis tool considered a medical device software?")

if st.button("Submit"):
    tab1, tab2 = st.tabs(["Final Answer", "Intermediate Results"])

    with tab1:
        st.subheader("Answer")
        progress_bar = st.progress(0)
        progress_text = st.empty()

        progress_text.text("Step 1: Generating action plan...")
        action_plan = orchestrator.process_message(query)
        progress_bar.progress(20)

        progress_text.text("Step 2: Processing action plan and database with RAG agent...")
        rag_response = rag_agent.process_message(action_plan)
        progress_bar.progress(40)

        progress_text.text("Step 3: Generating response...")
        response = rep_agent.process_message(rag_response)
        progress_bar.progress(60)

        progress_text.text("Step 4: Validating response...")
        val_response = val_agent.process_message(response)
        progress_bar.progress(80)

        progress_text.text("Step 5: Performing web search...")
        web_results = google_search(query + " medical device regulatory", google_api_key, google_cx)
        progress_bar.progress(100)

        progress_text.text("All steps completed.")
        st.write(val_response)

        st.subheader("RAG Sources")
        sources = rag_response.get("results", [])
        # avoid duplicates in sources
        unique_sources = []
        seen_sources = set()
        for src in sources:
            if src['source'] not in seen_sources:
                st.markdown(f"**Rank:** {src['rank']} | **Source:** {src['source']} | **Page:** {src['page']}")
                st.text_area(label=f"Source extract", value=src["content"][:500] + "...", height=150, key=src['source'])
                unique_sources.append(src)
                seen_sources.add(src['source'])

        st.subheader("Related Top 5 Web Search Results")
        for result in web_results[:5]:
            st.markdown(f" - {result['title']} | **Link:** {result['link']}")

    with tab2:
        # Orchestrator Agent
        st.subheader("1. Action Plan")
        st.write(action_plan)
        st.divider()
        # RAG Agent
        st.subheader("2. RAG Processing")
        st.write(rag_response)
        st.divider()
        # Response Agent
        st.subheader("3. Response")
        st.write(response)
        st.divider()
        # Validation Agent
        st.subheader("4. Validation Result")
        st.write(val_response)
        st.divider()
        # Web Search Results
        st.subheader("Web Search Results")
        st.write(web_results)
