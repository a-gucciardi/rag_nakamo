# RAG Nakamo: Was war eigentlich gebaut (und warum)

## TL;DR – Was dieses Ding eigentlich macht

**RAG Nakamo** hat als einfaches RAG-Projekt angefangen, aber es ist zu etwas viel interessanterem entwickelt. Jetzt ist es ein richtiges System, das wirklich *entscheidet*, ob es Dokumente durchsuchen muss bevor es teure API-Anfragen macht.

Das System ist gebaut für Leute, die mit medizinische Geräte Regulierung (FDA, WHO Sachen) arbeiten.

**Was wir nutzen:** V2.1
**Stack:** Python 3.10+, OpenAI GPT-4o-mini überall, ChromaDB für Vektoren, LangChain für schwere Arbeit
**Domain:** Medizinische Geräte regulatorische Compliance (weil jemand muss es machen)

---

## 1. Architektur Geschichte

### 1.1 Warum wir es so gebaut haben

Ich habe eine **geschichtete Ansatz gewählt, wo jedes Teil eine Sache gut macht**:

```
User fragt → Orchestrator entscheidet was zu machen → Vielleicht Dokumente suchen → 
Antwort generieren → Sicherheitscheck → Ergebnis zurückgeben
```

### 1.2 Neue Regeln

1. **Orchestrator soll wirklich orchestrieren** – Nicht mehr „hier was du sollst machen“ Antworten
2. **Sicherheit ist nicht optional** – Jede Antwort wird geprüft bevor sie rausgeht
3. **Neue Sachen sollen einfach zu addieren sein** – Plugin Architektur für neue Agents
4. **Zeig deine Arbeit** – Alles mit Timestamp loggen, damit wir wissen wo Zeit verschwendet wird
5. **Konfiguration an einem Ort** – Durch 15 Dateien für Modelname zu suchen ist zu mühsam

---

## 2. Was jedes Teil wirklich macht

### 2.1 Settings Management (oder: Warum wir keine API Keys hart-coden mehr)

Die dunklen Tage von `openai_api_key.txt` überall sind vorbei.

```python
class Settings(BaseSettings):
    orchestrator_model: str = "gpt-4o-mini"
    response_model: str = "gpt-4o-mini" 
    guard_model: str = "gpt-4o-mini"
    
    embeddings_model: str = "text-embedding-3-large"
    retrieval_top_k: int = 5
    enable_rerank: bool = False
```

**Vorteile**:

* **Keine API Key Jagd mehr** – Alles kommt aus `.env`
* **Typ-Sicherheit** – Pydantic schreit wenn wir Konfiguration falsch machen
* **Umgebungswechsel** – Dev/Test/Prod ohne Codeänderung
* **Ein Ort für alles** – Modelname ändern, wirkt überall

`@lru_cache` heißt, wir erzeugen Settings Objekt einmal und benutzen es wieder.

### 2.2 Agent Architektur

#### 2.2.1 Basis Agent

Jeder Agent erbt von dieser Basis-Klasse. Gibt konsistente Zeitmessung und Interfaces:

```python
class BaseAgent():
    def timed(self, query: str):
        start = time.perf_counter()
        result = self.process_message(query)
        duration = time.perf_counter() - start
        return result, duration
```

Idee: jede Operation messen, damit wir wissen was langsam ist. Vector search dauert oft viel länger als LLM calls.

#### 2.2.2 Orchestrator

Große Änderung in V2.1. Früher war es nur eine fancy Todo Liste, jetzt *denkt* es.

**Problem**: User fragt „Was sind FDA Software Anforderungen?“ und bekommt „Vielleicht such Dokumente und dann generiere Antwort“.

**Lösung**: Entscheidungen treffen und ausführen.

```python
def _should_use_rag(self, query: str) -> bool:
    regulatory_keywords = ['fda', 'who', 'regulation', 'medical device']
    return any(keyword in query.lower() for keyword in regulatory_keywords)
```

Aktuell nur Keyword-Matching. Später vielleicht ML.

**Ablauf jetzt**:

1. Frage anschauen – regulatorisch oder nicht?
2. Ja: Dokumente durchsuchen, relevante finden
3. Nein: teure Suche skippen, direkt antworten
4. Antwort generieren mit Kontext
5. Sicherheitscheck
6. Ergebnis zurück mit Quellen und Metadaten

Neues Agent Registrierungssystem → neue Agents können registriert werden ohne Orchestrator ändern.

#### 2.2.3 RAG Agent

Hier passiert Suchlogik – User-Frage → Vektor-Suche → relevante Chunks.

**Entscheidungen**:

* ChromaDB für Einfachheit
* OpenAI Embeddings → gut genug
* Semantische Chunking statt „alle 500 Zeichen“ – Kontext wichtig
* Reranking optional, wegen Latenz kaum Verbesserung

#### 2.2.4 Response Agent

Nimmt Frage + gefundene Dokumente → draft Antwort.

**Features**:

* Quellenzitate
* Fallback handling – auch ohne Dokumente
* Kontext Integration

### 2.3 Vector Store

#### 2.3.1 ChromaDB

* Persistiert auf Disk
* Local-first
* Performance gut für kleine Sammlung

#### 2.3.2 Dokument ingestion

**Naiv:** „Split 1000 Zeichen“ → Sätze kaputt, schlechter Kontext
**Besser:** Semantisch, Abschnitt-bewusst, adaptive Größe, Metadaten erhalten

### 2.4 Sicherheitslayer

#### 2.4.1 PromptGuard

Jede RAG-Antwort → Sicherheitsprüfung

```python
def classify_and_decide(self, user_prompt, draft_answer, context_docs):
    classification = self._call_classifier(user_prompt, draft_answer, context)
    decision = self._decide(classification)
    return GuardedResponse(decision=decision, final_answer=final_answer)
```

Ergebnisse: ALLOW / SANITIZE / BLOCK

#### 2.4.2 Schema

Alles Pydantic Modelle → Typprüfung, klare Fehler.

---

## 3. Wie alles zusammenpasst

### 3.1 Ablauf Beispiel

User fragt: „Was sind FDA Software Validierung Anforderungen?“

1. Query → Orchestrator
2. Entscheidung → regulatorisch → RAG starten
3. Suche → 5 relevante Chunks
4. Kontext → Response Agent
5. GPT-4o-mini → Antwort + Quellen
6. Sicherheitscheck → ALLOW
7. Ergebnis an User

\~3-4 Sekunden

Nicht regulatorische Frage: „Wie ist das Wetter?“ → direkt Antwort \~15-20 Sekunden

### 3.2 Performance Check

* Vector search: \~1s
* LLM: \~10-15s
* Sicherheitscheck: 0.5-1s
* Rest: klein

### 3.3 Fehler Handling

* RAG Agent down → fallback LLM
* Keine Dokumente → ohne RAG antworten
* Sicherheitscheck fails → block
* OpenAI down → Retry mit Backoff

### 3.4 Monitoring

* End-to-end Zeit
* RAG Entscheidung korrekt?
* Dokument Qualität
* Sicherheitsaktionen

---

## 5. Entscheidungen

### 5.1 OpenAI alles

* Einfach, ein API, konsistent, zuverlässig
* Nachteil: Single point of failure, Vendor lock-in

### 5.2 Keyword vs ML

* Keyword reicht, schnell, debugbar, \~95% Genauigkeit

### 5.3 Sicherheit

* RAG Antwort → immer prüfen
* Wichtig für regulatorische Inhalte
* Audit Trails
* Konfigurierbare Sensitivität

### 5.4 Agent Architektur

* Spezialisierte Agents, Registrierung
* Leicht testen, klar Verantwortlichkeiten, austauschbare Komponenten

### 5.5 Reranking

* Latenz +800ms, Qualität nur minimal besser
* Standardmäßig off

**Lektion:** Messen realen Impact, manchmal „gut genug“ ist gut genug

___ 
