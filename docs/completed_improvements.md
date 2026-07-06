# KRUTRIM RAG — Completed Retrieval Pipeline Improvements

> **Last updated**: 2026-07-06
> **Scope**: Part B retrieval pipeline (`partb/retrieval/pipeline.py` + supporting modules)

---

## Table of Contents

1. [Query Classification & Routing (1.2)](#12-query-classification--routing)
2. [Greedy Context Budget Allocation (4.1a)](#41a-greedy-context-budget-allocation)
3. [Markdown Table Preservation (4.3a)](#43a-markdown-table-preservation)
4. [Hierarchical Context — Summary + Detail (4.4)](#44-hierarchical-context--summary--detail)
5. [Dynamic System Prompt per Query Type (5.2)](#52-dynamic-system-prompt-per-query-type)
6. [Citation Format Enhancement (5.4)](#54-citation-format-enhancement)
7. [MMR Diversity Selection (2.3)](#23-mmr-diversity-selection)
8. [Reranker Upgrade — Jina Reranker v3 (3.1)](#31-reranker-upgrade--jina-reranker-v3)

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

## 3.1 Reranker Upgrade — Jina Reranker v3

**Commit**: `3c69fca`
**Files**: `partb/service/`

### What was done

Upgraded from `BAAI/bge-reranker-base` to `jina-reranker-v3` for the listwise reranking approach. This model provides better cross-attention quality and supports listwise scoring (scores all candidates relative to each other, not just independently).

### Key benefits

- **Listwise scoring**: Scores candidates as a ranked list rather than independently, better reflecting relative relevance
- **Quality**: Superior cross-attention for technical/scientific content
- **Multilingual support**: Handles mixed-language content (English + transliterated terms)

### Configuration

Configured via:
- `partb/service/base_model.py` — Base reranker service class
- `partb/service/medium_model.py` — Medium mode config
- `partb/service/deep_model.py` — Deep mode config

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
| 2.3 | MMR Diversity Selection | P1 | 1-2 days | ✅ Done | — |
| 3.1 | Reranker Upgrade (Jina v3) | P1 | 1 day | ✅ Done | — |
| 4.1a | Greedy Context Budget | P2 | 2-3 days | ✅ Done | 15+ |
| 4.3a | Markdown Table Preservation | P3 | 1-2 days | ✅ Done | 15+ |
| 4.4 | Hierarchical Context | P3 | 3-5 days | ✅ Done | 14 |
| 5.2 | Dynamic System Prompt | P3 | 1 day | ✅ Done | 12 |
| 5.4 | Citation Format Enhancement | P3 | 0.5 day | ✅ Done | 10 |

---

*Generated: 2026-07-06 | Based on completed code changes to KRUTRIM RAG Part B retrieval pipeline*
