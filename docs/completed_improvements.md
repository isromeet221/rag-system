# KRUTRIM RAG — Completed Retrieval Pipeline Improvements

> **Last updated**: 2026-07-07
> **Scope**: Part B retrieval pipeline (`partb/retrieval/pipeline.py` + supporting modules)

---

## Table of Contents

1. [Query Classification & Routing (1.2)](#12-query-classification--routing)
2. [Hybrid Search — Vector + BM25 (2.1)](#21-hybrid-search--vector--bm25)
3. [MMR Diversity Selection (2.3)](#23-mmr-diversity-selection)
4. [Adaptive Retrieval Depth (2.4)](#24-adaptive-retrieval-depth)
5. [Proposition-to-Section Mapping Improvement (2.5)](#25-proposition-to-section-mapping-improvement)
6. [Weighted Fusion of Proposition and Section Scores (2.6)](#26-weighted-fusion-of-proposition-and-section-scores)
7. [Reranker Upgrade — Jina Reranker v3 (3.1)](#31-reranker-upgrade--jina-reranker-v3)
8. [Greedy Context Budget Allocation (4.1a)](#41a-greedy-context-budget-allocation)
9. [Markdown Table Preservation (4.3a)](#43a-markdown-table-preservation)
10. [Hierarchical Context — Summary + Detail (4.4)](#44-hierarchical-context--summary--detail)
11. [Dynamic System Prompt per Query Type (5.2)](#52-dynamic-system-prompt-per-query-type)
12. [Citation Format Enhancement (5.4)](#54-citation-format-enhancement)

---

## 1.2 Query Classification & Routing

**Commit**: `fc8dd11`
**Files**: `partb/retrieval/pipeline.py`

### What was done

Added a `classify_query()` function that categorizes user queries into one of four types using regex patterns, then routes to a specialized retrieval strategy.

### Query Types

| Type | Detection Pattern | Strategy |
|---|---|---|
| `spec_lookup` | Numbers + units, "what is the X of Y" | Boost Neo4j specs, prioritize table chunks |
| `process` | "how does", "explain", "sequence" | Expand page range (N-2, N+2), boost text chunks |
| `comparison` | "vs", "versus", "compare", "difference" | Multi-book retrieval, graph entity links |
| `overview` | "what is", "tell me about", "summarize" | Section title hierarchy + first paragraphs |
| `general` | (fallback) | Balanced default settings |

### Key implementation details

- Controlled by `ENABLE_QUERY_CLASSIFICATION` config flag (default: on)
- Each type has its own config overrides for: `boost_both_mult`, `page_expand_range`, `final_top_n_adjust`, `context_max_chars_adjust`, `cross_book`
- The classified `query_type` is returned as part of the bundle dict and used by downstream features

### Verification

- `test_classifier.py` — 12+ tests covering all query types and edge cases

---

## 2.1 Hybrid Search — Vector + BM25

**Commit**: `5e5989e`
**Files**: `partb/retrieval/pipeline.py`, `partb/retrieval/hybrid.py`, `partb/config.py`

### What was done

Added a BM25 keyword search pass that runs alongside the vector search, fusing results using Reciprocal Rank Fusion (RRF). This catches exact-term matches that vector search may miss (e.g., "PSLV-C50" vs "PSLV C50").

### How it works

```python
RRF score = 1/(k + rank_vector(d)) + 1/(k + rank_bm25(d))
```

1. Scrolled all sections from Qdrant for the active book IDs
2. Built an in-memory BM25 index from section text
3. Ranked sections by BM25 relevance against the query
4. Fused BM25 and vector results using configurable RRF constant `k`
5. Merged back into `direct_sections` for downstream processing

### Key implementation details

- Controlled by `ENABLE_HYBRID` config flag (default: off)
- BM25 index built on-demand from Qdrant scroll — no separate index to maintain
- RRF constant `k` configurable via `HYBRID_RRF_K` (default: 60)
- Pool multiplier controls how many extra candidates per path to allow RRF room to re-rank
- Implemented in `partb/retrieval/hybrid.py`: `build_and_fuse()` function

---

## 2.3 MMR Diversity Selection

**Commit**: `be64f91`
**Files**: `partb/retrieval/pipeline.py`

### What was done

Added Maximum Marginal Relevance (MMR) as an optional diversity layer after reranking. Prevents the top-N from being dominated by chunks from the same page/section.

### How it works

```python
MMR = λ * score(c) - (1-λ) * max sim(c, selected)
```

- **λ** controls the tradeoff between relevance (score) and diversity (novelty vs. already-selected items)
- Similarity computed using Nomic embeddings (already available from Qdrant search results)
- Controlled by `ENABLE_MMR` and `MMR_LAMBDA` config variables

### Key implementation details

- Applies after reranking but before page expansion
- Only activates when candidates > top_n (otherwise no diversity needed)
- Reuses cached Nomic embedding vectors from Qdrant search results — no additional encoding cost

---

## 2.4 Adaptive Retrieval Depth

**Commit**: `3911387`
**Files**: `partb/retrieval/pipeline.py`, `partb/config.py`

### What was done

Replaced fixed retrieval limits with a two-pass adaptive approach. Simple queries use fewer candidates (lower latency), while hard queries automatically expand to find more relevant content.

### How it works

1. **First pass**: Retrieves propositions and sections at a conservative limit (50% of normal by default)
2. **Rerank check**: If the top reranker score is above `ADAPTIVE_DEPTH_SCORE_THRESHOLD`, the results are good enough — done
3. **Second pass**: If the top score is too low, re-retrieve with expanded limits (2x by default), merging new results with existing candidates

### Key implementation details

- Controlled by `ENABLE_ADAPTIVE_DEPTH` config flag (default: off)
- First pass fraction: `ADAPTIVE_DEPTH_INITIAL_FRACTION` (default: 0.5)
- Score threshold: `ADAPTIVE_DEPTH_SCORE_THRESHOLD` (default: 0.5)
- Expand multiplier: `ADAPTIVE_DEPTH_EXPAND_MULTIPLIER` (default: 2)
- Per-mode limits set independently in `MODE_CONFIG` (`prop_retrieve_limit`, `sect_retrieve_limit`)

---

## 2.5 Proposition-to-Section Mapping Improvement

**Commit**: `752bd4b`
**Files**: `partb/retrieval/pipeline.py`, `partb/config.py`

### What was done

Strengthened the small-to-big retrieval chain when a proposition's `parent_chunk_id` fails to resolve to a section. Two safeguards were added:

**2.5.1 Multi-parent section_path fallback**: If a proposition's `parent_chunk_id` doesn't exist in the sections collection, the system falls back to matching the proposition's `section_path + book_id` to find the parent section. A new function `fetch_sections_by_section_path()` scrolls sections for the matching book and filters by section path in Python.

**2.5.2 Direct search supplement**: If fewer than `MIN_PARENT_SECTIONS` (default: 3) parent sections are found after both chunk_id and section_path lookups, the pipeline runs an additional direct section search with an expanded limit to fill the gap.

### Key implementation details

- `fetch_sections_by_section_path()` uses `setdefault` to collect unique (book_id, section_path) pairs from orphan propositions
- Scrolls up to 1000 sections per book, filters by section_path in Python
- Section found by fallback is extended into `parent_sections` list for dedup in `merge_candidates`
- Supplement limit = `first_sect_lim + (MIN_PARENT_SECTIONS - total_parent) * 5`
- Configurable via `MIN_PARENT_SECTIONS` env var (default: 3)

---

## 2.6 Weighted Fusion of Proposition and Section Scores

**Commit**: `752bd4b`
**Files**: `partb/retrieval/pipeline.py`, `partb/config.py`

### What was done

Proposition search scores are now carried forward to influence section reranking. When `search_propositions()` returns results, the max proposition score per `parent_chunk_id` is recorded and later used as a boost in `rerank_candidates()`.

### How it works

1. **Build prop_score_map**: After `search_propositions()`, a dict maps each `parent_chunk_id` to its max proposition score
2. **Apply in merge**: `merge_candidates()` stores `prop_child_score` on sections whose chunk_id is in the map
3. **Boost in rerank**: `rerank_candidates()` adds `prop_child_score * PROP_SCORE_BOOST_WEIGHT * boost_mult` to each candidate's rerank score

### Key implementation details

- Controlled by `PROP_SCORE_BOOST_WEIGHT` config var (env: `RAG_PROP_SCORE_BOOST`, default: 0.10)
- Boost scales with `boost_mult` (used by query classification for spec_lookup queries)
- Not applied in the reranker fallback path (when Jina reranker fails, scores are from a different scale)
- Higher prop_score wins on dedup when a section appears in both parent and direct results

---

## 3.1 Reranker Upgrade — Jina Reranker v3

**Commits**: `3c69fca` (initial), `78e3871` (fix for transformers 5.x compat)
**Files**: `partb/service/`, `partb/retrieval/pipeline.py`

### What was done

Upgraded from `BAAI/bge-reranker-base` to `jina-reranker-v3` for the listwise reranking approach.

### Key benefits

- **Listwise scoring**: Scores candidates as a ranked list rather than independently
- **Quality**: Superior cross-attention for technical/scientific content
- **131K context**: Can process all candidates in a single forward pass

### Fix (commit `78e3871`)

The saved model's `config.json` was created with transformers 4.55.2, but the runtime has transformers 5.6.2. `AutoModel.from_pretrained()` with `trust_remote_code=True` could not resolve the `auto_map` entry, falling back to loading a bare `Qwen3Model` (no `.rerank()` method).

**Fix**: Directly import `JinaForRanking` from the local `modeling.py` via `importlib` instead of relying on `AutoModel`'s class resolution. This bypasses the auto-class resolution entirely.

```python
import importlib.util
spec = importlib.util.spec_from_file_location("jina_reranker_modeling", modeling_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
_reranker = mod.JinaForRanking.from_pretrained(reranker_str)
```

---

## 4.1a Greedy Context Budget Allocation

**Commit**: `d04f967`
**Files**: `partb/retrieval/pipeline.py`

### What was done

Replaced the fixed-order context assembly with a greedy density-based approach. Instead of "specs → pages → chunks in order", the system now:

1. **Always includes specs** (small, high-value)
2. **Scores each item** by `rerank_score / text_length` — value density
3. **Greedily picks** the highest-density items until the budget is full

### Key implementation details

- Density = `rerank_score / max(1, len(text))`
- Specs get a fixed high density (999) to ensure they're always included
- Chunks are kept atomic (no partial chunks)
- Separator `\n\n---\n\n` is accounted for in budget
- Controlled by `ENABLE_GREEDY_CONTEXT` config flag

### Verification

- `test_greedy_context.py` — 15+ tests covering basic selection, single column, single row, empty headers, pipe escaping, newline handling, missing keys, structured JSON, linearized fallback, raw text fallback, many columns

---

## 4.3a Markdown Table Preservation

**Commit**: (part of `d04f967` and subsequent fixes)
**Files**: `partb/retrieval/pipeline.py`

### What was done

Tables are now rendered as clean Markdown pipe tables instead of bullet lists. This leverages the fact that LLMs (especially Mistral) understand pipe tables natively and extract values more accurately.

### Before

```
Specification Data [Section] [Book: X | Page: Y]:
  - Parameter: Value Unit
  - Parameter: Value Unit
```

### After

```
Specification Data: [Book: X | § Section Name | Page: Y]:
| Parameter     | Value | Unit |
|---------------|-------|------|
| Thrust        | 799   | kN   |
| Chamber Press | 58    | bar  |
```

### Key implementation details

- Updated `format_table_for_llm()` to detect structured JSON (with `data` + `headers`) and render as pipe table
- Falls back to linearized text or raw text when structured data is unavailable
- Pipe escaping for cells containing `|`
- Headers bolded with Markdown table syntax
- Section label (from `section_path`) embedded in the citation

---

## 4.4 Hierarchical Context — Summary + Detail

**Commit**: `c5aa03b`
**Files**: `partb/retrieval/pipeline.py`

### What was done

For overview and process queries, the system now builds a two-level context:

1. **Summary level**: Section title hierarchy + first sentence of each subsection → gives the LLM a map of what's available
2. **Detail level**: Full content of top-ranked pages/chunks (budget-constrained)

### Output format

```
## Document Structure

**PSLV-C50**
- 4.3 Propulsion System
  - **4.3.1 Vikas Engine**
    → The Vikas engine is a liquid-fueled rocket engine...

## Detailed Content

...
```

### Key implementation details

- `extract_section_hierarchy()` reads metadata for books with expanded pages
- Builds a deduplicated section tree from metadata `sections` arrays
- Leaf sections are bolded with a first-sentence preview
- Budget is adjusted: `detail_budget = max(1200, effective_max_chars - hierarchy_chars)`
- Only activates for `overview` and `process` query types (from query classification)

### Verification

- `test_hierarchy.py` — 14 tests covering: first sentence extraction (decimals, abbreviations, empty), single/multi-book hierarchy, deduplication, missing metadata, empty sections

---

## 5.2 Dynamic System Prompt per Query Type

**Commit**: `059df05`
**Files**: `partb/retrieval/prompts.py`, `partb/retrieval/pipeline.py`

### What was done

The system prompt is now generated dynamically based on the query type classification. Each query type gets type-specific instructions appended to the base prompt.

### Type-specific instructions

| Type | Added Instruction |
|---|---|
| `spec_lookup` | Present value and unit first, then describe context. Use pipe tables for multi-parameter data. |
| `process` | Explain each step in sequence. Use numbered lists for step-by-step processes. |
| `comparison` | Present comparisons using structured format (table or bullet pairs). Highlight differences clearly. |
| `overview` | Start with a one-sentence summary, then expand with details organized by section. |
| `general` / None | No additional instructions (backward-compatible) |

### Key implementation details

- `get_system_prompt(mode, query_type=None)` — new optional parameter `query_type`
- When `query_type` is provided and found in `QUERY_TYPE_PROMPTS`, the type-specific block is appended after the mode-style block
- Backward compatible: callers without `query_type` work unchanged
- When query classification is disabled (`query_type = "general"`), no type-specific instructions are added

### Verification

- `test_dynamic_prompt.py` — 12 tests covering: all 4 query types, general/None/unknown fallback, all 3 modes, backward compatibility, base prompt always present

---

## 5.4 Citation Format Enhancement

**Commit**: `ae924d6`
**Files**: `partb/retrieval/prompts.py`, `partb/retrieval/pipeline.py`

### What was done

Enhanced the citation format to include the section name, making citations more informative for navigation and debugging.

### Format change

| Context | Before | After |
|---|---|---|
| With section | `[Book: X \| Page: Y]` | `[Book: X \| § Section Name \| Page: Y]` |
| No section | `[Book: X \| Page: Y]` | `[Book: X \| Page: Y]` (unchanged) |
| Table | `[Table \| Book: X \| Page: Y]` | `[Table \| Book: X \| § Section Name \| Page: Y]` |
| Spec data | `Specification Data [Section] [Book: X \| Page: Y]:` | `Specification Data: [Book: X \| § Section \| Page: Y]:` |

### Key implementation details

- Uses the `section_path` from chunk metadata to extract the section name
- Falls back to old format when `section_path` is empty or unavailable
- Uses `§` symbol as a compact section indicator
- Updated in all 7 citation locations:
  1. `prompts.py` — system prompt guideline
  2. `pipeline.py:format_table_for_llm()` — source_label + raw fallback
  3. `pipeline.py:_build_context_greedy()` — text chunk + table fallback
  4. `pipeline.py:build_context()` — chunk label + table fallback (non-greedy path)

### Verification

- `test_citation.py` — 10 tests covering: all citation paths, section-bearing and section-less chunks, all context assembly functions

---

## Summary of Completed Features

| # | Feature | Priority | Effort | Status | Tests |
|---|---|---|---|---|---|
| 1.2 | Query Classification & Routing | P2 | 1 day | ✅ Done | 12+ |
| 2.1 | Hybrid Search (Vector + BM25) | P1 | 3-5 days | ✅ Done | — |
| 2.3 | MMR Diversity Selection | P1 | 1-2 days | ✅ Done | — |
| 2.4 | Adaptive Retrieval Depth | P2 | 2-3 days | ✅ Done | — |
| 2.5 | Proposition-to-Section Mapping | P3 | 1 day | ✅ Done | 19 (integration) |
| 2.6 | Weighted Fusion of Prop & Section Scores | P3 | 1-2 days | ✅ Done | 19 (integration) |
| 3.1 | Reranker Upgrade (Jina v3) | P1 | 1 day | ✅ Done | — |
| 4.1a | Greedy Context Budget | P2 | 2-3 days | ✅ Done | 15+ |
| 4.3a | Markdown Table Preservation | P3 | 1-2 days | ✅ Done | 15+ |
| 4.4 | Hierarchical Context | P3 | 3-5 days | ✅ Done | 14 |
| 5.2 | Dynamic System Prompt | P3 | 1 day | ✅ Done | 12 |
| 5.4 | Citation Format Enhancement | P3 | 0.5 day | ✅ Done | 10 |

---

*Generated: 2026-07-07 | Based on completed code changes to KRUTRIM RAG Part B retrieval pipeline*
