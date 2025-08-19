# RAG Nakamo: What Was Actually Built (And Why)

## TL;DR - What This Thing Actually Does

**RAG Nakamo** started as a simple RAG project, but it evolved into something way more interesting. It's now a proper system that actually *evaluates* about whether it needs to search documents before bothering with expensive API calls.

The system is built for people dealing with medical device regulations (FDA, WHO stuff).

**What we're running:** V2.1  
**The stack:** Python 3.10+, OpenAI GPT-4o-mini everywhere, ChromaDB for vectors, LangChain for the heavy lifting  
**The domain:** Medical device regulatory compliance (because someone has to do it)  

---

## 1. Architecture Story

### 1.1 Why We Built It This Way

I went with a **layered approach where each piece does one thing well**:

```
User asks question → Orchestrator decides what to do → Maybe search docs → 
Generate answer → Security check → Return result
```

### 1.2 The new Rules Followed

1. **Make the orchestrator actually orchestrate** - No more "here's what you should do" responses
2. **Security isn't optional** - Every response gets checked before it goes out the door  
3. **New stuff should be easy to add** - Plugin architecture so we can bolt on new agents
4. **Show your work** - Log everything with timestamps so we know where time gets wasted
5. **Configuration in one place** - Because hunting through 15 files to change a model name is painful

---

## 2. What Each Piece Actually Does

### 2.1 Settings Management (Or: Why We're Not Hard-Coding API Keys Anymore)

Fixed the dark days of `openai_api_key.txt` files scattered everywhere.

```python
class Settings(BaseSettings):
    # All the LLM stuff
    orchestrator_model: str = "gpt-4o-mini"
    response_model: str = "gpt-4o-mini" 
    guard_model: str = "gpt-4o-mini"
    
    # Vector search settings
    embeddings_model: str = "text-embedding-3-large"
    retrieval_top_k: int = 5
    enable_rerank: bool = False  # Spoiler: we turned this off
```

**What this gets**:
- **No more API key hunting** - Everything comes from `.env` files
- **Type safety** - Pydantic yells at us if we mess up configuration
- **Environment switching** - Dev/test/prod configs without code changes
- **One place for all settings** - Change a model name once, affects everywhere

The `@lru_cache` decorator means we create this settings object once and reuse it.

### 2.2 Agent Architecture 

#### 2.2.1 Base Agent Pattern

Every agent inherits from this base class. Giving consistent timing and interfaces:

```python
class BaseAgent():
    def timed(self, query: str):
        start = time.perf_counter()
        result = self.process_message(query)
        duration = time.perf_counter() - start
        return result, duration
```

Simple idea: wrap every operation with timing so we know what's slow. Turns out vector searches take way longer than LLM calls, unless LLM context gets high. 

#### 2.2.2 The Orchestrator

This was the big V2.1 change. The old orchestrator was basically a fancy todo list generator. Now it actually *think* stuff.

**The problem**: User asks "What are FDA software requirements?" and gets back "You should probably search some documents and then generate a response.".

**The solution**: Make decisions and execute them.

```python
def _should_use_rag(self, query: str) -> bool:
    regulatory_keywords = ['fda', 'who', 'regulation', 'medical device'] # a bit more in practice
    return any(keyword in query.lower() for keyword in regulatory_keywords)
```

Currenlty, it's just keyword matching. Future clever solution should involve some smart ML matching.

**What happens now**:
1. Look at the question - is it smell regulatory?
2. If yes: dig through documents, find relevant docs
3. If no: skip the expensive search, just answer directly  
4. Generate response with whatever context we found
5. Security check 
6. Send it back with sources and metadata

The agent registration system means we can plug in new capabilities without rewriting the orchestrator. A web search agent? We can register it. A SQL database agent? Register it too. The orchestrator doesn't change.

#### 2.2.3 RAG Agent

This is where the search logic happens - turning user questions into vector searches and getting back relevant chunks.

**Key decisions we made**:
- **ChromaDB** for ease of use
- **OpenAI embeddings** working good enough and consistent  
- **Semantic chunking** instead of "split every 500 characters" - context matters
- **Reranking** - as an option, after tests it adds latency for minimal gains

The `similarity_search_with_score()` gives us confidence levels. Below a certain threshold, we don't even bother including the document.

#### 2.2.4 Response Agent

Takes the user question plus any documents we found and crafts a draft final answer. 

**Features that matter**:
- **Source citations** - With doc name and pages
- **Fallback handling** - Works with or without retrieved documents  
- **Context integration** - Weaves document content into natural responses

### 2.3 Vector Store

#### 2.3.1 ChromaDB Integration

Went with ChromaDB because:
- It persists to disk (no rebuilding indexes every startup)
- Local-first approach (no external dependencies)
- Performance is good enough for the document collection size (3)

```python
def create_and_populate_vector_store(chunks, embeddings, path, collection):
    persistent_client = chromadb.PersistentClient(path=path)
    vector_store = Chroma.from_documents(chunks, embeddings, client=persistent_client)
```

The collection concept could let us have different document types in separate buckets. FDA docs in one collection, WHO docs in another, etc.

#### 2.3.2 Document Ingestion 

This is where hard lessons were learned:

**Naive approach**: "Split every document into 1000 character chunks"  
**Result**: Sentences cut in half, context lost, possible terrible retrieval quality

**Better approach**: Semantic chunking that preserves meaning
- Section-aware splitting
- Quality analysis during ingestion  
- Adaptive chunk sizes based on content density
- Source metadata preservation

The ingestion pipeline now handles PDF text extraction, cleans up formatting artifacts, and creates meaningful chunks that actually answer questions.

### 2.4 Security Layer

#### 2.4.1 PromptGuard System

Every RAG-based response goes through security validation.

**The workflow**:
```python
def classify_and_decide(self, user_prompt, draft_answer, context_docs):
    # Analyze the prompt and response for safety issues
    classification = self._call_classifier(user_prompt, draft_answer, context)
    
    # Make a decision: allow, block, or sanitize
    decision = self._decide(classification) 
    
    # Return structured result with reasoning
    return GuardedResponse(decision=decision, final_answer=final_answer)
```

**Three possible outcomes**:
- **ALLOW**: Response looks good, send it through
- **SANITIZE**: Fix problematic parts, send modified version
- **BLOCK**: Too risky, send generic safety message

The key insight: only apply this to RAG-based responses. If someone asks "What's the weather?" we don't need to security-check the answer about it being sunny.

#### 2.4.2 Schema Definitions

Everything uses Pydantic models. 

```python
class GuardDecision(BaseModel):
    status: Literal["allow","block","sanitize"]
    reason: str 
    safe_message: Optional[str] = None
    classification: ClassificationResult
```

Type hints everywhere, automatic validation, and clear error messages when things go wrong.

---

## 3. How It All Fits Together 

### 3.1 What Actually Happens When Someone Asks a Question

Let's trace through a real example. User asks: "What are FDA software validation requirements?"

1. **Input**: Query hits the orchestrator
2. **Decision**: Sees "FDA" and "validation" - definitely a regulatory query, fire up the RAG
3. **Search**: ChromaDB finds 5 relevant document chunks about software validation
4. **Context**: Feeds the question + document chunks to the response agent  
5. **Generation**: GPT-4o-mini crafts an answer with proper citations
6. **Security**: PromptGuard analyzes the response - looks good, allow it
7. **Output**: User gets a detailed answer with source references

Total time: ~3-4 seconds. Not bad for searching document pages and generating a custom response.

Now let's try: "What's the weather like?"

1. **Input**: Query hits the orchestrator
2. **Decision**: No regulatory keywords found - skip the expensive stuff
3. **Direct Response**: Straight to GPT-4o-mini without document search
4. **Output**: "I don't have access to current weather data..."

Total time: ~15-20 seconds. Much faster because we skipped the vector search entirely.

### 3.2 Performance Reality Check

**What we learned from timing everything**:
- Vector searches: ~1 seconds (depends on collection size)
- LLM generation: ~10-15 seconds (depends on response length)
- Security classification: 0.5-1 seconds (analyzing content)
- Everything else: Negligible but adds up slightly

**Scaling considerations**:
- ChromaDB handles our document collection just fine, but this is not a large scale pdf db
- OpenAI has rate limits - we'd need to add queuing for high-traffic scenarios  
- Memory usage scales with context window size (we cap at 10K tokens)
- Complex regulatory queries generate longer responses (more LLM time)

### 3.3 When Things Go Wrong 

**Handled failures**:
- RAG agent unavailable? Fall back to direct LLM response with a warning
- No relevant documents found? Generate response without RAG context
- Security classification fails? Play it safe and block the response
- OpenAI API down? Retry with exponential backoff, then fail

**What we log**:
```python
logger.info(f"Orchestrating query: {query}")
logger.info(f"RAG search found {len(rag_results)} sources") 
logger.info(f"Response generated in {duration:.2f}s")
logger.info(f"Guard decision: {decision.status} - {decision.reason}")
```

### 3.4 Monitoring What Matters

**Performance metrics tracked**:
- End-to-end query latency (most important for user experience)
- RAG decision accuracy (are we classifying queries correctly?)
- Document retrieval quality (relevance scores from vector search)
- Security action frequency (how often do we block/sanitize responses?)

**Development vs Production**:
- Dev mode: Verbose logging, debug info, longer timeouts
- Production mode: Essential logs only, optimized for speed

---

## 5. The Decisions Made

### 5.1 Why OpenAI Everything?

**The decision**: Use OpenAI for literally every LLM operation - orchestration, RAG, response generation, security classification.

**Why it made sense**:
- **Simple** - Enough in test settings, no speed threshold
- **One API to rule them all** - Single authentication, billing, rate limiting system
- **Consistent performance** - GPT-4o-mini gives us predictable quality everywhere (most of the time)
- **Reliability** - OpenAI's uptime is better than anything we could self-host

**The downside**: Single point of failure and vendor lock-in. 

### 5.2 Simple Keyword Matching vs ML Classification  

**The decision**: Use basic keyword matching to decide if a query needs RAG search.

**Why we didn't overthink it**:
- **It actually works** - Regulatory queries have predictable vocabulary  
- **Fast** - No extra API calls or model loading
- **Debuggable** - When it's wrong, we can see exactly why
- **Good enough** - 95% accuracy for way less complexity

### 5.3 Security Everything 

**The decision**: Every single RAG-based response gets security checked before going out.

**The reasoning**:
- **Regulatory content is high-stakes** - Wrong advice about FDA requirements isn't just embarrassing
- **Documents can contain sensitive info** - Better to over-filter than leak something problematic  
- **Audit trails matter** - Decision logging for compliance purposes
- **Configurable paranoia** - We can tune the sensitivity based on deployment environment

**The tradeoff**: Adds ~5 second to every RAG response. But for regulatory content, that's totally worth it.

### 5.4 Agent Architecture

**The decision**: Split everything into specialized agents with a registration system.

**What this brings**:
- **Easy testing** - Each agent can be tested in isolation
- **Clear responsibilities** - Orchestrator orchestrates, RAG searches, Response generates
- **Swappable components** - Want to try a different response strategy? Just register a new agent
- **Step development** - Different steps can be made on different agents without conflicts

**The complexity cost**: More files, more interfaces, more abstraction. But the benefits outweigh the costs as soon as you have more than one person working on the project, or a complex projext.

### 5.5 Reranking

**The experiment**: Added cross-encoder reranking to improve document retrieval quality.

**The results**: 
- Latency increased by ~800ms per query
- Quality improvement was marginal (maybe 5-10% better)
- Added another model dependency and potential failure point

**The decision**: Disabled by default. Further tests needed.

**The lesson**: Always measure real-world impact, not just theoretical improvements. Sometimes "good enough" is actually good enough.

---

<!-- ## 6. Performance Tuning (What We Learned the Hard Way)

### 6.1 Vector Search Optimization

**Semantic chunking experiment**: Instead of "split every 1000 characters", we implemented content-aware chunking that preserves sentence and paragraph boundaries.

**Result**: Way better retrieval quality. Turns out cutting sentences in half makes them harder to search and understand. Who would have thought?

**Relevance scoring**: We use `similarity_search_with_score()` to get confidence metrics. Below 0.7 similarity? We don't include the document. Prevents irrelevant regulatory jargon from polluting responses.

**The reranking experiment**: Added cross-encoder reranking to refine search results.
- **Expected**: Better document ranking, more relevant responses
- **Reality**: Minimal quality improvement, significant latency increase
- **Decision**: Disabled by default. Not worth the performance hit.

### 6.2 LLM Optimization  

**Model choice**: Went with GPT-4o-mini everywhere after testing various options.
- **GPT-4**: Too expensive for the marginal quality improvement
- **GPT-3.5**: Noticeably worse at following complex prompts
- **GPT-4o-mini**: Sweet spot of cost, speed, and quality

**Context management**: 10K token limit prevents context overflow and keeps costs reasonable. For regulatory documents, this is usually plenty.

**Prompt engineering**: Spent way too much time optimizing prompts. The biggest wins came from clear instructions and good examples, not fancy prompt tricks.

### 6.3 Security Optimization

**Selective application**: Only run security checks on RAG-based responses. If someone asks about the weather, we don't need to analyze it for regulatory compliance issues.

**Context truncation**: Limit document context sent to security analysis. The full context matters for response generation, but security classification works fine with summaries.

**Decision caching**: Same query + same retrieved documents = same security decision. We cache results to avoid redundant API calls.

### 6.4 What Actually Slows Things Down

**Vector search**: 1-2 seconds. Scales with document collection size.  
**LLM generation**: 1-3 seconds. Scales with response complexity.
**Security classification**: 0.5-1 seconds. Pretty consistent.
**Everything else**: Under 100ms. Python overhead is negligible.

**The 80/20 rule**: Vector search and LLM calls account for 90% of latency. Optimizing anything else is bikeshedding.

---

## 7. What's Next (The Roadmap)

### 7.1 Stuff We Want to Build

**Ensemble RAG**: Instead of just ChromaDB vector search, combine multiple retrieval strategies. Maybe add BM25 for exact keyword matching alongside semantic search. The research suggests this works better, but we haven't had time to implement it properly.

**Smarter orchestration**: The keyword matching works well enough, but it's pretty crude. Could train a small classifier to make better decisions about when to engage RAG. Though honestly, the current approach works fine for 95% of cases.

**Multi-modal support**: Right now we only handle text. But regulatory documents have tables, diagrams, flowcharts. Would be cool to extract and search those too. PDF parsing is a pain though.

**Real-time document updates**: Currently you have to manually re-ingest documents when they change. Would be nice to have a file watcher that automatically updates the vector store when new regulatory guidance comes out.

### 7.2 Architecture Evolution Ideas

**Microservices**: Each agent could be its own service with API endpoints. Would enable horizontal scaling and independent deployments. Also would enable different teams to own different components. But also, complexity.

**API gateway**: Right now it's a library, not a service. Could wrap it in FastAPI and make it accessible to other systems. Would need to think through authentication, rate limiting, monitoring.

**Stream processing**: For high-throughput scenarios, could redesign around async/await and streaming responses. User starts getting response chunks while we're still searching documents.

**Multi-tenant**: Support multiple organizations with separate document collections and security policies. Would need to rethink data isolation and access controls.

### 7.3 The Realistic Next Steps

Let's be honest - most of the above is "nice to have" engineering porn. The current system works well for its intended purpose.

**Actual immediate priorities**:
1. **Better error handling** - More graceful degradation when things go wrong
2. **UI implementation** - Currently it's all command-line demos
3. **Better monitoring** - Metrics dashboard for production deployment
4. **Documentation** - More examples and deployment guides

**The lesson**: Build what you need, not what sounds cool. The current architecture solves the actual problem we set out to solve.

---

## 8. The Bottom Line

### What We Actually Built

RAG Nakamo went from "search PDFs and ask ChatGPT" to a proper intelligent system that makes decisions about information retrieval. The V2.1 orchestrator actually orchestrates instead of just making todo lists. The security layer prevents us from accidentally giving bad regulatory advice. The modular architecture means we can add new capabilities without rewriting everything.

**Is it perfect?** No. The keyword matching is crude, the performance could be better, and there's definitely more we could do.

**Does it work?** Hell yes. It answers regulatory questions accurately, cites sources properly, runs fast enough for real use, and doesn't break when things go wrong.

### The Real Wins

**Smart resource usage**: Only searches documents when the question actually needs it. Saves time and API costs.

**Security that makes sense**: Validates responses without being annoying. Only applies heavy-handed security to high-risk regulatory content.

**Actually maintainable**: Clear code organization, comprehensive logging, centralized configuration. Future developers (including us in 6 months) won't hate us.

**Production-ready**: Proper error handling, environment management, dependency control. You can actually deploy this thing.

### What We Learned

**Simple often beats clever**: Keyword matching works better than we expected. Complex ML solutions aren't always the answer.

**Measure everything**: Built-in timing showed us where the real bottlenecks are (spoiler: it's not Python overhead).

**Security is hard**: Getting the balance between safety and usability took multiple iterations. Still not perfect, but much better than our first attempt.

**Documentation matters**: Future us will thank present us for writing things down properly.

### Current Status

✅ **Core functionality**: Intelligent orchestration with conditional RAG  
✅ **Security validation**: PromptGuard safety checking  
✅ **Performance monitoring**: Comprehensive timing and logging  
✅ **Deployment ready**: Proper environment and dependency management  
✅ **Maintainable**: Clean architecture with separation of concerns

The project successfully demonstrates that you can build a sophisticated RAG system without overengineering it. Sometimes the best solution is the one that works reliably and doesn't make you want to rewrite it every six months.

**Status**: Production-ready for regulatory intelligence use cases. The kind of system you can actually deploy, maintain, and extend without losing your sanity. -->
