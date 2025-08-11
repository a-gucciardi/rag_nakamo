V2 : Fixed project. A good RAG and german practice.  

The first thing to do is to separate the src with the scripts (for vector db and other utilities).  
The main scripts will now be in src/rag_nakamo and will look something like this : 
  
  src/
    rag_nakamo/
      __init__.py
      **settings.py**
      agents/
      embeddings/
      vectorstore/
      UI/
      security-scripts/
      utils/
      tests/
      docs/
  scripts/

The settings.py script now handles the hyper-parameters set-up. 
```python3 -c "from rag_nakamo.settings import get_settings; print(get_settings())"``` prints config.
It also handles a new stable environment creation + secrets.  

Biggest issues this addresses are :
 - manual individual testing
 - manual settings change in different files
 - bad secret txt files
 - fragile env setup.

Next, to address the copy-pasted agent and redundancy, we create a base agent.  
First done and tested with orchestrator. OK
With the actual model running. OK
Added pretty logging.  OK

Next and before RAG Agent : isolate retrieval & context assembly. Subtasks:
 - Create vectorstore/ with chroma_manager.py and ingest.py OK
 - change naive test splitting to semantic chunking OK
 - improve ingest : sections, quality analysis, adaptive sizing OK~ (with chuncks + semantic)
 - relevance score OK with similarity_search_with_score
 - toggle rerank option, model choice ? OK
-> problem with reranking : it only refines existing results, add latency for minimal gains
 - Ensemble RAG : TODO

SECURITY SAFETY PHASE: 
Needs to independently protect the output
 - Added security/prompt_guard.py with sanitize_retrieved_text().
Core idea for prompt_guard.py, largely inspired from https://github.com/bogdan01m/security-rag/tree/main/services/sec_rag/llm:
 - Accept the raw RAG agent answer OK.
 - Normalize / validate with Pydantic schemas. for prompt and response harm check OK
 - Decide outcome: allow | block | sanitize (safe message if blocked) OK
 - Return answer and a decision object. OK 
 - To test after a response agent, curren is only dummy ~

Always apply sanitize before passing docs to response agent.

