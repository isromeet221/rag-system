# KRUTRIM RAG — Retrieval System Improvement Plan

> **Scope**: Part B retrieval pipeline (`partb/retrieval/pipeline.py` + supporting modules)
> **Goal**: Improve answer quality, citation accuracy, latency, and robustness
> **Status**: Planning phase — no code implemented yet

---

## Table of Contents

1. [Query Understanding & Pre-processing](#1-query-understanding--pre-processing)
2. [Retrieval Quality](#2-retrieval-quality)
3. [Reranking & Scoring](#3-reranking--scoring)
4. [Context Assembly & Construction](#4-context-assembly--construction)
5. [Prompt Engineering](#5-prompt-engineering)
6. [Generation & Post-Generation](#6-generation--post-generation)
7. [Graph-Specific Enhancements](#7-graph-specific-enhancements)
8. [Infrastructure & Performance](#8-infrastructure--performance)
9. [Evaluation & Monitoring](#9-evaluation--monitoring)
10. [Prioritization Matrix](#10-prioritization-matrix)

---

## 1. Query Understanding & Pre-processing

### 1.1 Query Rewriting (HyDE-style)

**Problem**: The user's raw query goes directly to vector search. Short/ambiguous queries (e.g. "thrust?" or "what about pressure?") produce weak embeddings that miss relevant chunks. Conversation history is appended to the prompt but not used to reformulate the query.

**Solution**: Before any retrieval, use a fast/cheap LLM call to rewrite the query considering conversation context.

```
Input:  "what about pressure?"  [history: "What is the Vikas engine thrust?"]
Output: "What is the combustion chamber pressure of the Vikas engine?"
```

**Implementation sketch** (`pipeline.py`):

```python
@log_process
def rewrite_query(query: str, history: list[dict]) -> str:
    """Reformulate the user query in light of conversation history."""
    if not history:
        return query
    msgs = [
        {"role": "system", "content": "Rewrite the user's question to be fully "
         "self-contained, incorporating relevant context from the conversation history. "
         "Output ONLY the rewritten question."},
    ]
    for h in history[-4:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": f"Rewrite this question: {query}"})
    # Call a cheap model (e.g. the Fast mode model) with low temperature
    rewritten = cheap_llm_call(msgs, temperature=0.1)
    return rewritten if rewritten else query
```

- **Effort**: 1-2 days
- **Impact**: Very High — eliminates the single biggest cause of retrieval misses
- **Risk**: Additional latency (~500ms-2s). Mitigate by running concurrently with initial retrieval.
- **Dependency**: A cheap/always-available LLM endpoint (the Fast mode `open-mistral-nemo` works)

### 1.2 Query Classification & Routing

**Problem**: All queries are treated identically. A "what is the thrust of X?" spec lookup needs different retrieval than "explain how the ignition sequence works".

**Solution**: Classify the query into a type, then route to a specialized retrieval strategy.

| Query Type | Detection | Strategy |
|---|---|---|
| `spec_lookup` | Contains numbers, units, "what is the X of Y" pattern | Boost Neo4j specs, prioritize table chunks |
| `process` | "how does", "explain", "sequence", "steps" | Expand page range (N-2, N+2), boost text chunks |
| `comparison` | "vs", "versus", "compare", "difference between" | Multi-book retrieval, graph entity links |
| `overview` | "what is", "tell me about", "summarize" | Fetch section title hierarchy + first paragraphs |
| `followup` | Short query, history exists | Query rewriting (1.1) is the primary fix; no separate routing needed |

**Implementation sketch**:

```python
QUERY_PATTERNS = {
    "spec_lookup": re.compile(r"(what\s+is\s+(the\s+)?\w+\s+(of|for|in)|[\d]+\s*\w+|(value|spec|parameter)\b)", re.I),
    "process": re.compile(r"(how\s+(does|is|are|was)|explain|sequence|steps|process|procedure)", re.I),
    "comparison": re.compile(r"\b(vs|versus|compare|difference|versus)\b", re.I),
    "overview": re.compile(r"^(what\s+is|tell\s+me\s+about|summarize|overview)", re.I),
}

def classify_query(query: str) -> str:
    for qtype, pattern in QUERY_PATTERNS.items():
        if pattern.search(query):
            return qtype
    return "general"
```

- **Effort**: 1 day
- **Impact**: Medium — fine-tuning of existing parameters per type
- **Note**: The mode system (Fast/Balanced/Deep) partially covers this; classification adds finer control

### 1.3 Query Expansion with Domain Synonyms

**Problem**: ISRO documentation uses specific terminology that may not match user vocabulary. A user asking about "SSLV" may need to find "SSLV-D1" or "Small Satellite Launch Vehicle".

**Solution**: Build a domain synonym map/phrase table and expand the query with related terms before encoding.

```
Input query:  "SSLV motor"
Expanded:     "SSLV SSLV-D1 Small Satellite Launch Vehicle motor engine propulsion"
```

Approaches:
1. **Static synonym map**: Pre-built dictionary of ISRO terms → variants
2. **LLM-generated**: Ask the LLM to generate 3-5 alternative phrasings
3. **Embedding-based**: For each query term, find nearest neighbors in the embedding space

- **Effort**: 2-3 days (static map) or 3-4 days (LLM-based)
- **Impact**: High for recall on terminology-heavy queries
- **Risk**: Query drift if expansion adds noise. Mitigate by keeping original query as primary.

### 1.4 Query Decomposition

**Problem**: Multi-part questions like "Compare the thrust of Vikas and the chamber pressure of S200" are embedded as a single vector, which averages away the distinctive parts.

**Solution**: Decompose into sub-queries, retrieve for each, then merge.

```
"Compare the thrust of Vikas and the chamber pressure of S200"
  → "What is the thrust of the Vikas engine?"
  → "What is the chamber pressure of the S200 booster?"
  → [merge + deduplicate]
```

- **Effort**: 3-5 days
- **Impact**: High for complex questions
- **Implementation**: Ask LLM to split the query, then run each sub-query through the full retrieval pipeline, deduplicating via `chunk_id`.

---

## 2. Retrieval Quality

### 2.1 Hybrid Search (Vector + BM25)

**Problem**: Pure vector search misses exact-term matches. A user asking about "PSLV-C50" in a document that uses "PSLV C50" gets a lower similarity score. Conversely, "4th stage" might be written as "PS-4" in the document.

**Solution**: Add a keyword/BM25 retrieval pass and fuse results with vector results using Reciprocal Rank Fusion (RRF).

```
RRF score = Σ 1/(k + rank_vector(d)) + Σ 1/(k + rank_bm25(d))
```

Qdrant supports hybrid search natively via `prefer_oversampling` + `scoring` parameters. Alternative: use `QdrantClient`'s `search_batch` with two queries (vector + keyword) and merge.

**Implementation points**:
- Qdrant's `full_text_filter` on text payload works for exact term matching
- Or, maintain a separate BM25 index using `rank_bm25` library
- RRF constant `k=60` is standard

```python
def hybrid_search(query_vec, query_text, book_ids, limit):
    # Vector search
    vector_results = qdrant_client.query_points(
        collection=COLLECTION_SECTIONS,
        query=query_vec,
        limit=limit * 2,
    )
    # Keyword search (using Qdrant's should/match)
    keyword_results = qdrant_client.query_points(
        collection=COLLECTION_SECTIONS,
        query=query_vec,  # still need a vector
        query_filter=Filter(
            should=[FieldCondition(key="text", match=MatchText(text=query_text))]
        ),
        limit=limit * 2,
    )
    return reciprocal_rank_fusion(vector_results, keyword_results, k=60)
```

- **Effort**: 3-5 days
- **Impact**: High — catches exact-match queries that vector search misses
- **Note**: Need to verify Qdrant plan includes full-text filter support

### 2.2 Multi-Query Retrieval

**Problem**: A single query vector only captures one facet. Different phrasings of the same information need may retrieve different chunks.

**Solution**: Generate 3-5 query variations via LLM, retrieve for each, merge results.

```
Query: "Vikas engine fuel"
  → "What propellant does the Vikas engine use?"
  → "Vikas engine fuel composition"
  → "Vikas engine propellant type"
  → (original)
```

**Merge strategy**: Collect all results, deduplicate by `chunk_id`, keep the best score across queries for each chunk.

- **Effort**: 3-4 days
- **Impact**: High — improves recall significantly
- **Cost**: 3-5× more vector searches per query
- **Optimization**: Only do multi-query for queries below a confidence threshold (e.g., top-1 proposition score < 0.75)

### 2.3 MMR (Maximum Marginal Relevance) for Diversity

**Problem**: Top-N after reranking are often all from the same page/section (same chunk). The answer misses peripheral but important information from other sections.

**Solution**: Instead of pure score-sorting, use MMR to balance relevance vs. diversity:

```
MMR = λ * score(c) - (1-λ) * max_{j in selected} sim(c, c_j)
```

Where `sim(c, c_j)` is cosine similarity of Nomic embeddings (already computed during retrieval — reuse them).

```python
def mmr_select(candidates, query_vec, lambda_=0.7, top_n=8):
    selected = []
    remaining = list(candidates)
    # Get embedding for each candidate (from Qdrant vector)
    while len(selected) < top_n and remaining:
        mmr_scores = []
        for c in remaining:
            rel = c["rerank_score"]
            div = max(cosine_sim(c["vector"], s["vector"]) for s in selected) if selected else 0
            mmr_scores.append(lambda_ * rel - (1 - lambda_) * div)
        best_idx = max(range(len(remaining)), key=lambda i: mmr_scores[i])
        selected.append(remaining.pop(best_idx))
    return selected
```

- **Effort**: 1-2 days
- **Impact**: Medium — prevents redundant context
- **Tuning**: `λ` parameter needs calibration per mode. Start with λ=0.7.

### 2.4 Adaptive Retrieval Depth

**Problem**: Retrieval depth is fixed per mode (40 propositions, 40 sections). Simple queries need far fewer, complex ones might need more.

**Solution**: Use a two-pass approach:
1. Initial retrieval with conservative limits (e.g., 20 props, 20 sections)
2. If reranker scores are low across the board (< 0.5), re-retrieve with double limits
3. Rerank the combined set

- **Effort**: 2-3 days
- **Impact**: Medium — reduces latency for simple queries, improves recall for hard ones

### 2.5 Proposition-to-Section Mapping Improvement

**Problem**: The current pipeline searches propositions (atomic), then maps to parent sections. If a proposition's `parent_chunk_id` is missing or points to a section that wasn't indexed, the small-to-big chain breaks.

**Solution**:
1. **Multi-parent fallback**: In addition to `parent_chunk_id`, store the `section_path` on each proposition. If parent section not found by ID, do a secondary lookup by `section_path + book_id`.
2. **Section-level direct search fallback**: If < 3 parent sections are found, supplement with direct section search results regardless.

- **Effort**: 1 day
- **Impact**: Medium — more robust when ingestion has gaps
- **Files**: `pipeline.py:fetch_sections_by_chunk_ids()` and `merge_candidates()`

### 2.6 Weighted Fusion of Proposition and Section Scores

**Problem**: Currently, propositions and sections are searched independently. The merge step doesn't use proposition scores to influence section weighting.

**Solution**: When `search_propositions()` returns results, record the top proposition score per `parent_chunk_id`. When merging candidates, pre-score sections by their max child proposition score, and use this as a tiebreaker/boost in reranking.

- **Effort**: 1-2 days
- **Impact**: Low-Medium — marginal signal improvement

---

## 3. Reranking & Scoring

### 3.1 Reranker Model Upgrade

**Current**: `BAAI/bge-reranker-base` (XLM-RoBERTa, ~512 tokens, ~280M params)

**Options** (ordered by quality):
| Model | Params | Quality | Runtime (CPU) | Notes |
|---|---|---|---|---|
| `bge-reranker-v2-m3` | 568M | Better | ~2× slower | Multilingual, better cross-attention |
| `jina-reranker-v2` | ~300M | Better | ~1.5× slower | Fine-tuned for general RAG |
| `ms-marco-MiniLM-L-12-v2` | ~110M | Comparable | ~1.5× faster | Lighter, good for Fast mode |
| `cohere rerank` (API) | — | Best | Fast (API) | External dependency, cost |

**Recommendation**: Upgrade to `bge-reranker-v2-m3` for all modes, or use cascade:
- Fast mode: `bge-reranker-base` (current, fast enough)
- Deep mode: `bge-reranker-v2-m3` (best quality)

- **Effort**: 1 day (model download + path change in config)
- **Impact**: Medium-High — better relevance scores at the top of the funnel

### 3.2 Cascade Reranking

**Problem**: CrossEncoder reranking is O(n) where n = candidates (40+). Running a slow/accurate model on all candidates is expensive.

**Solution**: Two-stage cascade:
1. **Stage 1**: Cheap ranker (e.g., bge-reranker-base or even Qdrant scores) on all candidates → keep top 20
2. **Stage 2**: Expensive/accurate ranker (e.g., bge-reranker-v2-m3) on top 20 → final top 8-10

```python
def cascade_rerank(query, candidates, cheap_top=20, expensive_top=8):
    cheap_ce = get_reranker_cheap()
    expensive_ce = get_reranker_expensive()
    
    # Stage 1
    for c in candidates:
        c["stage1_score"] = cheap_ce.predict([(query, c["text"])])[0]
    candidates.sort(key=lambda x: x["stage1_score"], reverse=True)
    
    # Stage 2
    top_k = candidates[:cheap_top]
    for c in top_k:
        c["rerank_score"] = expensive_ce.predict([(query, c["text"])])[0]
    top_k.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    return top_k[:expensive_top]
```

- **Effort**: 1-2 days
- **Impact**: Medium — better quality at the same or lower total latency
- **Risk**: If cheap ranker misses relevant chunks entirely, they never reach stage 2

### 3.3 Score Calibration Per Mode

**Current**: BOOST_BOTH = 0.12 (flat) and RERANK_MIN_SCORE = 0.0 (disabled)

**Problem**: The reranker's sigmoid output produces scores in [0,1], but without calibration we don't know what scores actually mean. The boost of 0.12 was chosen arbitrarily.

**Solution**:
1. Collect score distributions over real queries (already logged via `RERANK_LOG_DISTRIBUTION_EVERY`)
2. Calibrate BOOST_BOTH per mode: Fast mode may need a different boost than Deep because of different models/context budgets
3. Set RERANK_MIN_SCORE based on observed distributions (e.g., drop bottom 10%)

```
Tuning process:
  1. Run 100+ queries with RERANK_LOG_DISTRIBUTION_EVERY=1
  2. Analyze logs: min, median, max scores
  3. Set RERANK_MIN_SCORE = 10th percentile
  4. Set BOOST_BOTH = what lifts a "medium relevance" chunk into "high relevance"
```

- **Effort**: 1-2 days (measurement + tuning)
- **Impact**: Medium — prevents low-quality chunks from reaching context

### 3.4 Segment-Max Pooling Enhancement

**Current**: Long chunks split into ~220-token overlapping windows, max score taken.

**Problem**: Window overlap is 50% (hardcoded). For very long chunks (1000+ words), many windows are scored, but only the max is kept — information about the runner-up windows is lost.

**Solutions**:
1. **Top-K pooling**: Take average of top-3 window scores instead of max. Smoother signal for multi-fact chunks.
2. **Content-aware segmentation**: Split on section/subsection boundaries within the chunk, not just sentences. A chunk that contains multiple subsections should be scored per-subsection.

```python
def _rerank_score_one_v2(ce, query, text):
    if len(text.split()) <= RERANK_FULL_CHUNK_WORDS:
        return float(ce.predict([(query, text)])[0])
    windows = _segment_windows(text, RERANK_SEGMENT_TOKENS)
    if not windows:
        return 0.0
    scores = ce.predict([(query, w) for w in windows])
    # Top-3 mean pooling
    top_k = sorted(scores, reverse=True)[:3]
    return float(sum(top_k)) / len(top_k)
```

- **Effort**: 0.5 days
- **Impact**: Low-Medium — marginal improvement for very long chunks
- **Note**: Requires retuning RERANK_MIN_SCORE since score distribution shifts

---

## 4. Context Assembly & Construction

### 4.1 Dynamic Context Budget Allocation

**Current**: Context budget is fixed per mode (12K/14K/16K chars). If chunks are short, budget is wasted. If content exceeds budget, it's truncated arbitrarily.

**Solutions**:

**4.1a Greedy Budget Allocation**:
Instead of "specs → pages → chunks in order", use a greedy approach:
1. Always include specs (small, high-value)
2. Score each page block and chunk by (rerank_score / chars) — value density
3. Greedily pick highest density items until budget is full

```python
def build_context_greedy(spec_block, page_blocks, chunks, max_chars):
    items = []
    if spec_block:
        items.append({"text": spec_block, "density": 999, "type": "spec"})
    for i, pb in enumerate(page_blocks):
        items.append({"text": pb, "density": page_scores[i] / max(1, len(pb)), "type": "page"})
    for c in chunks:
        items.append({"text": format_chunk(c), "density": c["rerank_score"] / max(1, len(c["text"])), "type": "chunk"})
    
    items.sort(key=lambda x: x["density"], reverse=True)
    selected = []
    total = 0
    for item in items:
        if total + len(item["text"]) <= max_chars:
            selected.append(item["text"])
            total += len(item["text"])
    return "\n\n---\n\n".join(selected)
```

- **Effort**: 2-3 days
- **Impact**: Medium — better content utilization, especially when chunks vary in length
- **Risk**: May break the "tables-first" guarantee (partial tables). Mitigate by keeping tables atomic.

**4.1b Dynamic Mode Selection**:
Simple queries get Fast mode budget; complex ones get Deep mode budget regardless of user selection. Detect from query length and entity count.

**4.1c Progressive Context Loading**:
Start with spec block + top-3 chunks. After generation, if the LLM indicates insufficient information, stream additional context and regenerate the relevant portion.

### 4.2 Structured Context (JSON / Tagged Sections)

**Problem**: Flat text context causes "lost in the middle" — LLMs pay more attention to content at the start and end. All chunks look the same to the LLM.

**Solution**: Structure the context with clear typed sections that the LLM can attend to selectively:

```
<context>
  <specifications book="PSLV-C50">
    - Propellant: HTPB (Hydroxyl-terminated polybutadiene)
    - Thrust: 799 kN
  </specifications>

  <page book="PSLV-C50" page="42" section="4.3 Propulsion System">
    The Vikas engine is a liquid-fueled rocket engine...
  </page>

  <chunk book="PSLV-C50" page="43" relevance="0.92" source="qdrant">
    Combustion chamber pressure is maintained at 58 bar...
  </chunk>
</context>
```

Benefits:
- LLM can clearly distinguish spec facts from narrative text
- Source attribution is built into the structure
- Relevance scoring is visible to the LLM (it can trust high-scoring chunks more)

- **Effort**: 2-3 days
- **Impact**: Medium-High — reduces lost-in-the-middle, improves citation accuracy
- **Requires**: System prompt update to explain the structure

### 4.3 Table Formatting Improvements

**Current**: Tables are converted to bullet lists:
```
Specification Data [Section] [Book: X | Page: Y]:
  - Parameter: Value Unit
```

**Problems**:
1. Multi-column tables lose relational structure (which value goes with which parameter across columns)
2. The LLM still sometimes says "as shown in the table" despite the anti-table reminder
3. Tables with many rows become long bullet lists

**Solutions**:

**4.3a Markdown Table Preservation**:
For well-structured tables, preserve the pipe table format that the LLM was trained on:
```
| Parameter     | Value | Unit |
|---------------|-------|------|
| Thrust        | 799   | kN   |
| Chamber Press | 58    | bar  |
```

LLMs (especially Mistral) understand pipe tables natively. Test shows they extract values more accurately from pipe tables than from bullet lists.

**4.3b Row Summarization**:
For tables with 10+ rows, add a summary row:
```
| Parameter | Value | Unit |
|-----------|-------|------|
| ... (12 rows omitted — see above for full data) |
| Max Thrust | 799 | kN |
```

- **Effort**: 1-2 days
- **Impact**: Medium — better numerical accuracy
- **Implementation**: Update `format_table_for_llm()` in `pipeline.py`

### 4.4 Hierarchical Context (Summary + Detail)

**Problem**: For overview questions ("tell me about the propulsion system"), the LLM needs a broad view across many pages. Current page expansion only gives 3 pages.

**Solution**: Two-level context assembly:
1. **Summary level**: Section title hierarchy + first sentence of each subsection → gives the LLM a map of what's available
2. **Detail level**: Full content of top-ranked pages/chunks

```python
def build_hierarchical_context(top_chunks, spec_block, book_ids):
    summary = extract_section_hierarchy(top_chunks, book_ids)
    detail = assemble_detail_context(top_chunks, spec_block)
    return f"## Document Structure\n{summary}\n\n## Detailed Content\n{detail}"
```

- **Effort**: 3-5 days
- **Impact**: High for overview questions
- **Dependency**: Need a service to extract section hierarchies from metadata or head sections

### 4.5 Spec Block Enhancement

**Current**: Spec block uses Neo4j results only. If GLiNER fails, specs are empty.

**Solutions**:

**4.5a Regex-Based Spec Extraction from Chunks**:
After reranking, scan top chunks for spec-like patterns directly:
```
Patterns: "<entity> is <value> <unit>"
          "<entity>: <value> <unit>"
          "<entity> = <value> <unit>"
```
No NER model needed. Works on any chunk text.

**4.5b Spec Conflict Resolution**:
If the same entity has different values across sources, explicitly flag the conflict:
```
[CONFLICT] Vikas Engine Thrust:
  - Page 42: 799 kN
  - Page 87: 800 kN
```

- **Effort**: 2-3 days (4.5a) or 1 day (4.5b)
- **Impact**: Medium (4.5a — more specs found), Medium (4.5b — prevents contradictory answers)

---

## 5. Prompt Engineering

### 5.1 Few-Shot Examples in System Prompt

**Current**: Zero-shot system prompt with guidelines.

**Problem**: The LLM's output format is inconsistent. Sometimes it uses bullet lists, sometimes paragraphs. Citation placement varies.

**Solution**: Add 2-3 few-shot examples demonstrating desired behavior:

```
Examples:

Question: What is the thrust of the Vikas engine?
Answer: The Vikas engine produces **799 kN** of thrust.
[Book: PSLV-C50 | Page: 42]

Question: Compare the propellants used in the first and fourth stages of PSLV.
Answer: The PSLV uses different propellants across its stages:
- **First stage (PS1)**: Uses HTPB (hydroxyl-terminated polybutadiene) solid propellant [Book: PSLV_UG | Page: 15]
- **Fourth stage (PS4)**: Uses MMH (monomethylhydrazine) fuel and N₂O₄ oxidizer in a liquid propulsion system [Book: PSLV_UG | Page: 18]
```

- **Effort**: 1 day
- **Impact**: Medium — consistent formatting
- **Risk**: Longer prompts = fewer tokens for context. Mitigate: use short examples.

### 5.2 Dynamic System Prompt per Query Type

**Current**: One system prompt per mode (Fast/Balanced/Deep).

**Solution**: Generate the system prompt based on query classification (1.2):

```python
def get_dynamic_system_prompt(mode: str, query_type: str) -> str:
    base = BASE_SYSTEM_PROMPT + MODE_STYLE.get(mode, "")
    
    type_specific = {
        "spec_lookup": "\n7. When answering about specifications, present the value and unit first, then describe context.",
        "process": "\n7. Explain each step in sequence. Use numbered lists for step-by-step processes.",
        "comparison": "\n7. Present comparisons using a structured format (table or bullet pairs). Highlight differences clearly.",
        "overview": "\n7. Start with a one-sentence summary, then expand with details organized by section.",
    }
    
    return base + type_specific.get(query_type, "")
```

- **Effort**: 1 day
- **Impact**: Medium — more tailored answers per query type

### 5.3 Chain-of-Thought (CoT) for Complex Queries

**Problem**: For multi-step questions, the LLM sometimes jumps to conclusions without reasoning through the available context.

**Solution**: For queries classified as "complex" (multi-sentence, multiple entities), add a CoT preamble:

```
Before answering, analyze what information is needed:
1. What entities/parameters are mentioned in the question?
2. Which parts of the context contain relevant data for each?
3. How do the pieces fit together?

Then provide the answer following the standard format.
```

- **Effort**: 1 day
- **Impact**: Medium — better structured reasoning
- **Cost**: Longer generation, more tokens

### 5.4 Citation Format Enhancement

**Current**: `[Book: X | Page: Y]`

**Problems**:
- Section name not included (the most useful navigational hint)
- Adjacent pages not distinguishable from the exact page
- No way to reference multiple sources for a single fact

**Enhanced format**: `[Book: X | § Section Name | Page: Y]`

Multi-source: `[Book: X | § Propulsion | Page: 42][Book: Y | § Engine Specs | Page: 15]`

- **Effort**: 0.5 days (prompt change only)
- **Impact**: Low — marginal improvement over current format

---

## 6. Generation & Post-Generation

### 6.1 Self-Correction / Self-RAG

**Problem**: The LLM cannot check its own work. If retrieval is slightly off or the LLM hallucinates, the user gets a wrong answer.

**Solution**: Two-stage generation:
1. **Generation**: LLM produces answer from context (current)
2. **Verification**: LLM re-reads the context and checks each factual claim in the answer against it

```
Verification prompt:
"You produced this answer based on the context below.
For EACH factual claim in your answer, cite the exact context line(s) that support it.
If a claim is NOT supported by the context, mark it as UNSUPPORTED.

Answer: [LLM's answer]
Context: [context]

Verification result:"
```

If any claims are flagged as unsupported → regenerate with a warning: "Your previous answer had unsupported claims. Ensure ALL statements are directly supported by the context."

- **Effort**: 3-5 days
- **Impact**: Very High — catches hallucinations before the user sees them
- **Cost**: 2× LLM calls per query. Use Fast mode for verification to reduce cost.
- **Latency**: +2-5 seconds for verification pass

### 6.2 Citation Validation (Post-Processing)

**Problem**: The LLM may cite `[Book: X | Page: 99]` for a page that wasn't actually in the context. Or it may format citations incorrectly.

**Solution**: After generation, scan the output for citation patterns and validate against the actual source list:

```python
import re

def validate_citations(text: str, sources: list[dict]) -> tuple[str, list[dict]]:
    """Check each citation against actual sources. Flag mismatches."""
    actual_sources = {(s["book_id"], s.get("page_range", [0])[0]) for s in sources if s.get("included_in_context")}
    
    cited = re.findall(r'\[Book:\s*([^|]+)\|Page:\s*(\d+)\]', text)
    invalid = []
    for book, page in cited:
        if (book.strip(), int(page)) not in actual_sources:
            invalid.append({"cited": f"[Book: {book} | Page: {page}]", "book": book.strip(), "page": int(page)})
    
    if invalid:
        # Option A: Remove invalid citations
        # Option B: Flag them with [UNCITED] tag
        # Option C: Regenerate with a warning
        logger.warning("[CITATION] %d invalid citations detected: %s", len(invalid), invalid)
    
    return text, invalid
```

- **Effort**: 2-3 days
- **Impact**: High — ensures every cited source actually exists
- **Risk**: May flag correct citations that use slightly different formatting. Use fuzzy matching.

### 6.3 Answer Completeness Check

**Problem**: No feedback loop on whether the answer fully addresses the question.

**Solution**: After generation, ask the LLM to check completeness:

```python
def check_completeness(question: str, answer: str) -> tuple[bool, list[str]]:
    """Returns (is_complete, missing_points)"""
    check_prompt = f"""
Question: {question}
Answer: {answer}

Did the answer fully address ALL parts of the question?
List any missing information or unanswered sub-questions.
If complete, respond: COMPLETE
"""
    result = cheap_llm_call(check_prompt)
    if "COMPLETE" in result:
        return True, []
    return False, [line for line in result.split("\n") if line.strip()]
```

If incomplete → re-retrieve with focus on missing topics and regenerate.

- **Effort**: 3-4 days
- **Impact**: High — ensures thorough answers
- **Cost**: Another LLM call. Run only for Deep mode where quality matters most.

### 6.4 Factual Consistency Verification (NLI)

**Problem**: The LLM may introduce facts that contradict the context, even if every individual statement is grounded somewhere.

**Solution**: Use a Natural Language Inference model (e.g., `facebook/bart-large-mnli`) to check each sentence of the answer against the entire context:

```
Answer sentence: "The Vikas engine produces 800 kN of thrust."
Context passage: "The Vikas engine has a thrust of 799 kN."
NLI result: CONTRADICTION
```

- **Effort**: 4-6 days (integration + model download)
- **Impact**: High — catches subtle contradictions
- **Cost**: Heavy — per-sentence NLI is expensive. Only for critical answers.

### 6.5 Answer Formatting Standardization

**Problem**: The LLM sometimes uses inconsistent formatting (mixing markdown styles, varying heading levels).

**Solution**: Post-process the output:
1. Normalize heading levels (start at `##`, never `#`)
2. Ensure consistent bullet style (`-` not `*`)
3. Wrap code blocks with language specification
4. Normalize citation spacing

```python
def normalize_formatting(text: str) -> str:
    text = re.sub(r'^# ', '## ', text, flags=re.MULTILINE)  # No single-hash headings
    text = re.sub(r'(?<!\n)\n(?!\n)', '\n\n', text)  # Double newlines for paragraphs
    text = re.sub(r'\[Book:\s+', '[Book: ', text)  # Normalize spacing
    return text.strip()
```

- **Effort**: 0.5 days
- **Impact**: Low-Medium — cosmetic but improves readability

---

## 7. Graph-Specific Enhancements

### 7.1 Multi-Hop Graph Traversal

**Current**: Neo4j is used for:
1. Entity → Section mapping (MENTIONED_IN)
2. Entity → Spec mapping (HAS_SPECIFICATION)

**Missed opportunities**: The graph has richer structure:
- `NEXT_SECTION`: adjacent sections for context
- `SENTENCE_CO_OCCURS`: related entities that appear together
- `HAS_TABLE` / `HAS_TABLE_ROW`: structured data access
- Entity → Entity co-occurrence: find related topics

**Solution**: After finding initial entities, traverse the graph to discover related context:

```python
def neo4j_related_entities(entity_terms, book_ids):
    """Find entities that co-occur with the query entities."""
    cypher = """
    MATCH (e1:Entity)-[co:SENTENCE_CO_OCCURS]-(e2:Entity)
    WHERE e1.book_id IN $book_ids AND toLower(e1.name) IN $terms
    RETURN e2.name AS entity, e2.book_id AS book_id, 
           co.count AS co_occurrence_count
    ORDER BY co.count DESC LIMIT 20
    """
    rows = session.run(cypher, ...)
    return [{"entity": r["entity"], "book_id": r["book_id"]} for r in rows]

def neo4j_adjacent_sections(section_names, book_ids):
    """Find sections before and after the matched ones."""
    cypher = """
    MATCH (s1:Section)-[:NEXT_SECTION]->(s2:Section)
    WHERE s1.name IN $names AND s1.book_id IN $book_ids
    RETURN s2.name AS next_section, s2.book_id AS book_id
    """
    ...
```

- **Effort**: 3-5 days
- **Impact**: High — leverages existing graph investment, enables richer context discovery
- **Risk**: More Neo4j queries → higher latency. Mitigate with parallel execution.

### 7.2 Entity Disambiguation

**Problem**: User query terms may not match graph entity names exactly. "Vikas" might be stored as "Vikas Engine", "Vikas-2", or "Vikas engine".

**Current**: Case-insensitive exact match + contains (if term >= 4 chars).

**Improved approach**:
1. **Fuzzy matching**: Use `apoc.text.fuzzyMatch` or trigram similarity in Neo4j
2. **Abbreviation expansion**: Map common abbreviations (LPSC → Liquid Propulsion Systems Centre)
3. **Synonym resolution**: Use the synonym map from 1.3

```cypher
// Fuzzy match entities
MATCH (e:Entity)
WHERE e.book_id IN $book_ids
  AND ANY(t IN $terms WHERE 
    toLower(e.name) CONTAINS toLower(t)
    OR apoc.text.distance(toLower(e.name), toLower(t)) < 3
    )
```

- **Effort**: 2-3 days
- **Impact**: Medium — catches entities that were previously missed
- **Dependency**: APOC plugin in Neo4j

### 7.3 Graph-Based Spec Comparison Across Documents

**Problem**: When comparing specs across different books (e.g., Vikas in PSLV vs. Vikas in GSLV), the current pipeline doesn't leverage the graph to find the same entity in both books.

**Solution**: After entity extraction, query Neo4j for specs of the same entity name across all selected books:

```cypher
MATCH (e:Entity)-[:HAS_SPECIFICATION]->(sp:Spec)
WHERE toLower(e.name) IN $lower_entity_names
RETURN e.book_id AS book_id, e.name AS entity, 
       sp.value AS value, sp.unit AS unit, sp.raw AS raw
```

Then explicitly format the comparison in context:
```
=== COMPARISON ACROSS DOCUMENTS ===
Parameter: Vikas Engine Thrust
  - PSLV-C50: 799 kN [Page 42]
  - GSLV-Mk3: 805 kN [Page 38]
  - LVM3:     805 kN [Page 22]
```

- **Effort**: 2-3 days
- **Impact**: Medium — useful for comparative questions
- **Dependency**: Requires same entity name across books (entity resolution)

### 7.4 TABLE Node Integration

**Current**: Neo4j has `Table` and `TableRow` nodes with `row_data` but the pipeline never queries them.

**Solution**: After finding a relevant section, check if it contains tables:

```cypher
MATCH (s:Section)-[:HAS_TABLE]->(t:Table)
WHERE s.name IN $section_names AND s.book_id IN $book_ids
OPTIONAL MATCH (t)-[:HAS_ROW]->(r:TableRow)
RETURN t.table_id AS table_id, 
       t.linearized_text AS linearized,
       collect(r.row_data) AS rows
```

This provides an alternate path to structured data when Qdrant vector search fails for table chunks.

- **Effort**: 1-2 days
- **Impact**: Low-Medium — redundant with Qdrant but provides a backup path

---

## 8. Infrastructure & Performance

### 8.1 Async Parallel Execution

**Current**: The pipeline runs synchronously in sequence (propositions → entities → graph → sections → direct → merge → rerank → expand → context).

**Problem**: Several steps have no dependency on each other and could run in parallel.

**Parallelizable groups**:

```
Group A (no dependencies): 
  - search_propositions()
  - extract_query_entities()
  - search_sections_direct()

Group B (depends on Group A):
  - neo4j_sections_for_entities()   [depends on entities]
  - neo4j_specs_for_terms()         [depends on entities]

Group C (depends on Group A + B):
  - fetch_sections_by_chunk_ids()   [depends on propositions]
  
Group D (depends on Group B + C):
  - merge_candidates()
  - rerank_candidates()
```

Using `asyncio.gather()` or `ThreadPoolExecutor`:

```python
async def parallel_retrieve(query, book_ids):
    loop = asyncio.get_running_loop()
    
    # Group A
    props_task = loop.run_in_executor(None, search_propositions, query, book_ids, 40)
    entities_task = loop.run_in_executor(None, extract_query_entities, query)
    sections_task = loop.run_in_executor(None, search_sections_direct, query, book_ids, [], 40)
    
    prop_hits, entity_terms, direct_sections = await asyncio.gather(props_task, entities_task, sections_task)
    
    # Group B (from entities)
    graph_task = loop.run_in_executor(None, neo4j_sections_for_entities, book_ids, entity_terms)
    specs_task = loop.run_in_executor(None, neo4j_specs_for_terms, book_ids, entity_terms)
    neo4j_names, specs = await asyncio.gather(graph_task, specs_task)
    
    # Group C (from proposition parent IDs)
    parent_ids = list({p["parent_chunk_id"] for p in prop_hits if p.get("parent_chunk_id")})
    parent_task = loop.run_in_executor(None, fetch_sections_by_chunk_ids, parent_ids, book_ids)
    parent_sections = await parent_task
    
    # Groups D, E, F remain sequential
    ...
```

- **Effort**: 2-4 days
- **Impact**: Medium — reduces wall-clock time by 30-50% depending on network latency
- **Risk**: Increased memory usage from concurrent operations

### 8.2 Retrieval Caching (Semantic Cache)

**Problem**: Similar queries re-run the full retrieval pipeline unnecessarily.

**Solution**: Cache retrieval results (context + sources) keyed by:
- `(query_embedding, book_ids, mode)` — semantic similarity
- Or simpler: `(query_hash, book_ids, mode)` — exact match

When a new query's embedding is within `ε` (cosine distance < 0.05) of a cached query, reuse the context:

```python
class SemanticCache:
    def __init__(self, threshold=0.95, max_size=100):
        self.cache: list[tuple[np.ndarray, dict]] = []
        self.threshold = threshold
        self.max_size = max_size
    
    def get(self, query_vec: np.ndarray) -> dict | None:
        for cached_vec, result in self.cache:
            similarity = cosine_similarity(query_vec, cached_vec)
            if similarity > self.threshold:
                return result
        return None
    
    def put(self, query_vec: np.ndarray, result: dict):
        self.cache.append((query_vec, result))
        if len(self.cache) > self.max_size:
            self.cache.pop(0)  # LRU eviction
```

- **Effort**: 2-3 days
- **Impact**: Medium-High — eliminates retrieval latency for repeated/similar questions
- **Risk**: Stale results if documents are updated. Mitigate with TTL per cache entry.

### 8.3 Embedding Cache

**Problem**: The same query (or very similar) is re-encoded every time. SentenceTransformer encoding takes ~200-500ms on CPU.

**Solution**: LRU cache keyed by query text:

```python
from functools import lru_cache

@lru_cache(maxsize=256)
def cached_encode(query: str) -> list[float]:
    model = get_nomic()
    return model.encode("search_query: " + query, show_progress_bar=False).tolist()
```

- **Effort**: 0.5 days
- **Impact**: Low — modest improvement, but free to implement

### 8.4 First-Token Latency Reduction

**Problem**: `retrieve_bundle()` blocks before any token is generated. Total pre-generation time: ~3-8 seconds (GLiNER + Neo4j + 2× Qdrant + reranking + metadata loading).

**Solutions**:

**8.4a Speculative Start**: Start sending the system prompt to the LLM while retrieval is still running. When retrieval completes, append the context and user message.

**8.4b Partial Streaming**: Stream the spec block immediately (always fast to fetch), then append page/chunk context as it becomes available. The LLM starts processing the spec block while page content is still being fetched.

**8.4c Pre-compute**: Keep the Nomic model's embedding matrix warm, pre-load reranker, pre-connect Neo4j, etc. (most of this is already done via lazy initialization in `pipeline.py`).

- **Effort**: 4-6 days (8.4a) or 2-3 days (8.4b)
- **Impact**: Perceived speed improvement (user sees first token faster)

### 8.5 GPU Acceleration for Embedding & Reranking

**Current**: All ML inference runs on CPU (Nomic embedding, CrossEncoder reranker, GLiNER NER).

**Solution**: Move embedding and reranking to GPU if available:

```python
def get_nomic():
    global _nomic_model
    if _nomic_model is None:
        model_dir = PORTABLE_DIR / "nomic"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _nomic_model = SentenceTransformer(str(model_dir), trust_remote_code=True, device=device)
        if device == "cuda":
            _nomic_model.half()  # FP16 for 2× speed
    return _nomic_model
```

- **Effort**: 1 day
- **Impact**: High — embedding 3-5× faster, reranking 2-3× faster
- **Dependency**: CUDA-capable GPU + PyTorch CUDA

### 8.6 Model Quantization / ONNX Export

**Problem**: ML models are memory-heavy and slow on CPU.

**Solution**: Export SentenceTransformer and CrossEncoder to ONNX format for 2-3× CPU inference speedup:

```bash
pip install optimum onnxruntime
python -m optimum.exporters.onnx -m path/to/nomic --output path/to/nomic-onnx
```

Then load with `OnnxRuntime` provider instead of PyTorch.

- **Effort**: 2-4 days
- **Impact**: Medium — faster inference without GPU
- **Risk**: Compatibility issues with SentenceTransformer API

### 8.7 LiteLLM Connection Pool

**Problem**: Each query opens a new httpx connection to LiteLLM.

**Solution**: Use `httpx.AsyncClient` as a singleton with connection pooling:

```python
_llm_client: httpx.AsyncClient | None = None

def get_llm_client() -> httpx.AsyncClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=httpx.Limits(max_keepalive_connections=5))
    return _llm_client
```

- **Effort**: 0.5 days
- **Impact**: Low — reduces connection setup overhead by ~100-200ms per query

---

## 9. Evaluation & Monitoring

### 9.1 Retrieval Quality Metrics

**Problem**: There is no systematic way to measure whether retrieval quality improves or degrades after changes.

**Solution**: Log per-query metrics for offline analysis:

```python
# Added to retrieve_bundle() return dict
"metrics": {
    "num_propositions_found": len(prop_hits),
    "num_entities_extracted": len(entity_terms),
    "num_neo4j_sections": len(neo4j_section_names),
    "num_neo4j_specs": len(specs),
    "num_candidates": len(candidates),
    "num_reranked": len(top),
    "rerank_scores_distribution": [min, median, max],
    "page_expansions_used": len(expansions),
    "context_chars_used": total_chars,
    "context_budget_pct": total_chars / max_chars * 100,
    "retrieval_time_ms": elapsed_ms,
    "num_chunks_in_context": num_chunks_included,
    "num_pages_in_context": num_pages_included,
}
```

Then build an analysis dashboard: "What % of queries use code spec blocks? What's the average rerank score? How often does page expansion trigger?"

- **Effort**: 2-3 days for logging, 1-2 days for dashboard
- **Impact**: High — enables data-driven tuning

### 9.2 User Feedback Collection

**Problem**: No way for users to signal answer quality.

**Solution**: Add thumbs-up/thumbs-down buttons in the chat UI, linked to each message:

```python
# New endpoint in chats_router.py
@router.post("/messages/{message_id}/feedback")
def submit_feedback(message_id: str, body: FeedbackBody, user=Depends(verify_token)):
    col = messages_col()
    col.update_one({"message_id": message_id}, {"$set": {"feedback": body.dict()}})
    return {"status": "ok"}
```

Feedback data: `{rating: 1/0, comment?: str, incorrect_sources?: list}`

Use feedback to:
1. Identify consistently bad sources → investigate ingestion quality
2. Tune RERANK_MIN_SCORE (high feedback on low-scoring chunks → lower threshold)
3. Identify missing synonyms → add to expansion map

- **Effort**: 2-3 days (backend + UI)
- **Impact**: Very High — direct signal of real-world quality

### 9.3 A/B Testing Framework

**Problem**: Hard to know if a retrieval change improves quality without a controlled experiment.

**Solution**: Add mode-parameter overrides per query (already partially supported via env vars). Route a percentage of queries to the experimental config:

```python
# In retrieve_bundle()
if random.random() < 0.1:  # 10% of traffic
    cfg = get_experiment_config("reranker_v2")
```

Compare user feedback (9.2) between control and experiment groups.

- **Effort**: 3-5 days
- **Impact**: Medium — enables disciplined iteration

### 9.4 Answer Quality Scoring

**Problem**: No automated way to score answer quality for regression testing.

**Solution**: Build a test suite of 50-100 Q&A pairs with golden answers. After each retrieval change, run the suite and measure:

1. **Exact match**: Does the answer contain the expected value?
2. **Citation accuracy**: Are all citations valid?
3. **Completeness**: Does the answer cover all expected points?
4. **Conciseness**: Answer length vs. golden answer length

Automate with an LLM-as-judge:

```python
def score_answer(question, answer, golden_answer, context):
    prompt = f"""
Rate the following answer on a scale of 1-5 for:
- Accuracy (all claims supported by context)
- Completeness (covers all aspects of the question)
- Citation quality (sources correctly cited)

Question: {question}
Expected answer: {golden_answer}
Actual answer: {answer}
Context: {context[:2000]}

Output: {{"accuracy": N, "completeness": N, "citations": N, "notes": "..."}}
"""
    result = llm_judge(prompt)
    return json.loads(result)
```

- **Effort**: 5-8 days (test suite + automation)
- **Impact**: Very High — enables confident iteration

---

## 10. Prioritization Matrix

| Priority | Item | Phase | Effort (days) | Impact | Risk | Dependencies |
|---|---|---|---|---|---|---|
| **P0** | 1.1 Query Rewriting | 1 | 1-2 | Very High | Low | Cheap LLM endpoint |
| **P0** | 1.3 Query Expansion (domain synonyms) | 1 | 2-3 | High | Low | Domain glossary |
| **P0** | 4.5a Spec Extraction from Chunks | 1 | 2-3 | Medium | Low | None |
| **P0** | 6.1 Self-Correction | 2 | 3-5 | Very High | Medium | LLM endpoint |
| **P0** | 6.2 Citation Validation | 2 | 2-3 | High | Low | None |
| **P0** | 9.2 User Feedback Collection | 1 | 2-3 | Very High | Low | UI changes |
| **P1** | 2.1 Hybrid Search (Vector + BM25) | 2 | 3-5 | High | Low | Qdrant config |
| **P1** | 2.3 MMR Diversity | 1 | 1-2 | Medium | Low | None |
| **P1** | 3.1 Reranker Upgrade | 1 | 1 | Medium-High | Low | Model download |
| **P1** | 4.2 Structured Context | 2 | 2-3 | Medium-High | Medium | Prompt redesign |
| **P1** | 7.1 Multi-Hop Graph Traversal | 2 | 3-5 | High | Low | Neo4j queries |
| **P1** | 9.4 Answer Quality Scoring | 2 | 5-8 | Very High | Low | Test suite |
| **P2** | 1.2 Query Classification & Routing | 2 | 1 | Medium | Low | None |
| **P2** | 1.4 Query Decomposition | 3 | 3-5 | High | Medium | LLM endpoint |
| **P2** | 2.2 Multi-Query Retrieval | 2 | 3-4 | High | Medium | LLM endpoint |
| **P2** | 2.4 Adaptive Retrieval Depth | 2 | 2-3 | Medium | Low | None |
| **P2** | 4.1 Greedy Context Allocation | 2 | 2-3 | Medium | Medium | Testing |
| **P2** | 5.1 Few-Shot Prompts | 2 | 1 | Medium | Low | None |
| **P2** | 6.3 Answer Completeness Check | 3 | 3-4 | High | Medium | LLM endpoint |
| **P2** | 8.1 Async Parallel Execution | 2 | 2-4 | Medium | Low | Refactoring |
| **P2** | 8.2 Semantic Cache | 2 | 2-3 | Medium-High | Low | None |
| **P3** | 2.5 Proposition Mapping Fallback | 3 | 1 | Medium | Low | None |
| **P3** | 3.2 Cascade Reranking | 3 | 1-2 | Medium | Low | 2nd reranker |
| **P3** | 3.3 Score Calibration | 2 | 1-2 | Medium | Low | Log data |
| **P3** | 3.4 Top-K Segment Pooling | 3 | 0.5 | Low | Low | None |
| **P3** | 4.3 Table Formatting | 3 | 1-2 | Medium | Low | None |
| **P3** | 4.4 Hierarchical Context | 3 | 3-5 | High | Medium | Metadata service |
| **P3** | 5.2 Dynamic System Prompt | 3 | 1 | Medium | Low | Query classifier |
| **P3** | 5.3 CoT for Complex Queries | 3 | 1 | Medium | Low | None |
| **P3** | 5.4 Citation Format | 3 | 0.5 | Low | Low | None |
| **P3** | 6.4 NLI Factual Consistency | 3 | 4-6 | High | High | NLI model |
| **P3** | 6.5 Format Standardization | 3 | 0.5 | Low | Low | None |
| **P3** | 7.2 Entity Disambiguation | 3 | 2-3 | Medium | Low | APOC plugin |
| **P3** | 7.3 Cross-Doc Spec Comparison | 3 | 2-3 | Medium | Low | None |
| **P3** | 7.4 TABLE Node Integration | 3 | 1-2 | Low | Low | None |
| **P3** | 8.4 First-Token Latency | 3 | 4-6 | Medium | High | Complex |
| **P3** | 8.5 GPU Acceleration | 3 | 1 | High | Medium | GPU |
| **P3** | 8.6 ONNX Quantization | 3 | 2-4 | Medium | Medium | ONNX toolkit |
| **P3** | 9.1 Retrieval Metrics | 2 | 2-3 | High | Low | None |
| **P3** | 9.3 A/B Testing | 3 | 3-5 | Medium | Low | Metrics infra |

---

## Phase Recommendations

### Phase 1 — Quick Wins (2-3 weeks)

Focus on low-effort, high-impact items that provide immediate quality improvements:

1. **Query Rewriting** (1.1) — 2 days
2. **Domain Synonym Expansion** (1.3) — 2-3 days
3. **MMR Diversity** (2.3) — 1-2 days
4. **Reranker Upgrade** (3.1) — 1 day
5. **Spec Extraction from Chunks** (4.5a) — 2-3 days
6. **Few-Shot Prompts** (5.1) — 1 day
7. **User Feedback Collection** (9.2) — 2-3 days
8. **Retrieval Metrics Logging** (9.1) — 2-3 days

**Expected impact**: 20-30% improvement in perceived answer quality

### Phase 2 — Core Improvements (4-6 weeks)

Items that require more build but provide structural improvements:

1. **Self-Correction** (6.1) — 3-5 days
2. **Citation Validation** (6.2) — 2-3 days
3. **Hybrid Search** (2.1) — 3-5 days
4. **Structured Context** (4.2) — 2-3 days
5. **Multi-Hop Graph Traversal** (7.1) — 3-5 days
6. **Async Parallel Execution** (8.1) — 2-4 days
7. **Semantic Cache** (8.2) — 2-3 days
8. **Score Calibration** (3.3) — 1-2 days
9. **Answer Quality Scoring** (9.4) — 5-8 days (start early, runs in background)

**Expected impact**: 40-60% reduction in hallucinations, 30% latency improvement

### Phase 3 — Advanced (8-12 weeks)

High-effort, high-reward items that transform the system:

1. **Multi-Query Retrieval** (2.2) — 3-4 days
2. **Query Decomposition** (1.4) — 3-5 days
3. **Answer Completeness Check** (6.3) — 3-4 days
4. **Hierarchical Context** (4.4) — 3-5 days
5. **First-Token Latency Reduction** (8.4) — 4-6 days
6. **GPU Acceleration** (8.5) — 1 day (if GPU available)
7. **Entity Disambiguation** (7.2) — 2-3 days
8. **NLI Factual Consistency** (6.4) — 4-6 days
9. **A/B Testing Framework** (9.3) — 3-5 days

**Expected impact**: 2-3× quality improvement over baseline, near-zero hallucination rate

---

## Architecture Diagram (Proposed)

```
User Query
    │
    ├── Query Rewriting (1.1) ─── with conversation history
    ├── Query Classification (1.2) ─── route to strategy
    ├── Query Expansion (1.3) ─── domain synonyms
    │
    ▼
┌──────────────────────────────────────────────────────┐
│              Parallel Retrieval Group A              │
│                                                      │
│  ┌──────────────┐   ┌─────────────┐   ┌───────────┐ │
│  │ Proposition  │   │ Section     │   │  GLiNER   │ │
│  │ Search       │   │ Direct      │   │  Entity   │ │
│  │ (Qdrant)     │   │ (Qdrant)    │   │  Extract  │ │
│  └──────┬───────┘   └──────┬──────┘   └─────┬─────┘ │
└─────────┼──────────────────┼────────────────┼───────┘
          │                  │                │
          ▼                  ▼                ▼
┌──────────────────────────────────────────────────────┐
│              Parallel Retrieval Group B              │
│                                                      │
│     ┌─────────────────┐      ┌──────────────────┐    │
│     │ Neo4j Sections  │      │ Neo4j Specs      │    │
│     │ (by entity)     │      │ (by entity)      │    │
│     └────────┬────────┘      └────────┬─────────┘    │
│              │                       │               │
│     ┌────────▼────────────────────────▼─────────┐    │
│     │   Multi-Hop Graph Traversal (7.1)         │    │
│     │   (adjacent sections, related entities,   │    │
│     │    cross-doc specs)                       │    │
│     └────────────────┬──────────────────────────┘    │
└──────────────────────┼──────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  Section Fetch by Chunk IDs (Qdrant scroll)          │
│  Hybrid Search BM25 + Vector Fusion (2.1)             │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  Merge Candidates + MMR Diversity (2.3)              │
│  Cascade Reranking (3.2)                              │
│  Score Calibration (3.3)                               │
│  Semantic Cache Check (8.2)                            │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  Page Expansion (N-1/N/N+1)                          │
│  Spec Extraction from Chunks (4.5a)                   │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  Context Assembly                                     │
│  1. Structured Spec Block (with conflict resolution) │
│  2. Full Page Content (tagged)                        │
│  3. Fallback Chunks (greedy, density-ordered)         │
│  4. Hierarchical Summary (for overview questions)     │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  LLM Generation                                      │
│  1. System Prompt (dynamic per query type)            │
│  2. User Message (with CoT if complex)               │
│  3. Stream tokens via LiteLLM/Ollama                 │
│  4. Self-Correction verification (6.1)                │
│  5. Citation Validation (6.2)                         │
│  6. Completeness Check (6.3)                          │
│  7. Format Standardization (6.5)                       │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
                    Answer
```

---

## Appendix: Configuration Changes Required

New environment variables / config keys needed for various improvements:

```python
# config.py additions

# Query rewriting
ENABLE_QUERY_REWRITING = os.environ.get("RAG_ENABLE_QUERY_REWRITING", "1") == "1"
QUERY_REWRITE_MODEL = os.environ.get("RAG_QUERY_REWRITE_MODEL", "open-mistral-nemo")

# Synonym expansion
SYNONYM_FILE = PARTB_DIR / "isro_synonyms.json"  # path to synonym map

# Hybrid search
ENABLE_HYBRID_SEARCH = os.environ.get("RAG_ENABLE_HYBRID", "0") == "1"
HYBRID_RRF_K = int(os.environ.get("RAG_HYBRID_RRF_K", "60"))
HYBRID_VECTOR_WEIGHT = float(os.environ.get("RAG_HYBRID_VECTOR_WEIGHT", "0.7"))

# MMR
ENABLE_MMR = os.environ.get("RAG_ENABLE_MMR", "0") == "1"
MMR_LAMBDA = float(os.environ.get("RAG_MMR_LAMBDA", "0.7"))

# Self-correction
ENABLE_SELF_CORRECTION = os.environ.get("RAG_ENABLE_SELF_CORRECTION", "0") == "1"

# Cascade reranking
RERANKER_V2_DIR = PORTABLE_DIR / "reranker_v2"
CASCADE_CHEAP_TOP = int(os.environ.get("RAG_CASCADE_CHEAP_TOP", "20"))

# Context assembly
CONTEXT_STRUCTURED = os.environ.get("RAG_CONTEXT_STRUCTURED", "0") == "1"
CONTEXT_GREEDY = os.environ.get("RAG_CONTEXT_GREEDY", "0") == "1"

# Cache
CACHE_MAX_SIZE = int(os.environ.get("RAG_CACHE_MAX_SIZE", "100"))
CACHE_THRESHOLD = float(os.environ.get("RAG_CACHE_THRESHOLD", "0.95"))
CACHE_TTL_SECONDS = int(os.environ.get("RAG_CACHE_TTL", "300"))
```

---

*Generated: 2026-07-03 | Based on codebase analysis of KRUTRIM RAG Part B retrieval pipeline*
