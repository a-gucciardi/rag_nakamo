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
