import streamlit as st
from orchestrator import OrchestratorAgent
from rag import RAGAgent
from response import ResponseAgent
from validation import ValidationAgent
from web_search import google_search

st.set_page_config(layout="wide")
st.title("MedTech AI Regulatory Assistant")
st.write("Ask questions about AI regulations in medical technology.")

query = st.text_input("Enter your query:", "Compare FDA and WHO approaches to risk management for medical devices")

if st.button("Submit"):
    tab1, tab2 = st.tabs(["Final Answer", "Intermediate Results"])

    with tab1:
        st.subheader("Answer")
        progress_bar = st.progress(0)
        progress_text = st.empty()

        progress_text.text("Step 1: Generating action plan...")
        first_agent = OrchestratorAgent(openai_api_key="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")
        action_plan = first_agent.process_message(query)
        progress_bar.progress(20)

        progress_text.text("Step 2: Processing action plan and database with RAG agent...")
        rag_agent = RAGAgent(chroma_db_path="./chroma_db")
        rag_response = rag_agent.process_message(action_plan)
        progress_bar.progress(40)

        progress_text.text("Step 3: Generating response...")
        rep_agent = ResponseAgent(openai_api_key="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")
        response = rep_agent.process_message(rag_response)
        progress_bar.progress(60)

        progress_text.text("Step 4: Validating response...")
        val_agent = ValidationAgent(openai_api_key="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")
        val_response = val_agent.process_message(response)
        progress_bar.progress(80)

        progress_text.text("Step 5: Performing web search...")
        web_results = google_search(query + " medical device regulatory", "AIzaSyCazrlL9XMtAgauwrF9zSITH8Ax0bFwQwI", "715f6ddbfb51f4d07")
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
