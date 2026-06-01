"""
processing/ingest_neo4j.py
---------------------------
STEP 5 of the new processing pipeline.

Replaces neo4j_server/master_server.py + neo4j_worker.py entirely.
Single script. No HTTP server. No worker polling. No load balancing needed.
Estimated runtime: 5-15 minutes for a 500-page book on CPU.

Reads  → data/checkpoints/{book_id}_ready.json
Uses   → portable/gliner/   (offline GLiNER model)
         portable/nltk_data/ (offline NLTK punkt tokenizer)
Target → bolt://localhost:7687  (hardcoded, no env vars)
Auth   → ("neo4j", "sac@1234") (hardcoded, no env vars)
Writes → data/neo4j/{book_id}_neo4j_log.json

FIVE LAYERS BUILT IN ORDER
────────────────────────────
Layer 1 — Document Hierarchy         (pure Python, deterministic)
  MERGE Book → Chapter → Section → Subsection nodes
  MERGE HAS_CHAPTER / HAS_SECTION / HAS_SUBSECTION / HAS_CHUNK edges
  MERGE NEXT_SECTION edges (sequential navigation)

Layer 2 — Specification Nodes        (regex, deterministic)
  Detects <subject> <number> <unit> patterns in chunk text
  MERGE Entity node for subject
  MERGE Spec node {property, value, unit, raw}
  MERGE HAS_SPECIFICATION edge

Layer 3 — Entity-Section Links       (GLiNER)
  Runs GLiNER on full chunk content
  MERGE Entity nodes {name, type}
  MERGE MENTIONED_IN edge (Entity → Section)
  Replaces old page-level CO_OCCURS_WITH

Layer 4 — Sentence Co-occurrence     (GLiNER per sentence)
  Splits chunk into sentences via NLTK
  Runs GLiNER on each sentence
  Entity pairs in same sentence → accumulated in memory
  MERGE SENTENCE_CO_OCCURS {count, sections} edges at end
  Sentence-level = real signal. Page-level was noise.

Layer 5 — Table Nodes                (pure Python, deterministic)
  Reads structured_json from triple_rep.py output
  MERGE Table node {title, headers, row_count}
  MERGE TableRow nodes {parameter, value, unit}
  MERGE HAS_TABLE (Section → Table)
  MERGE HAS_ROW (Table → TableRow)

Called by pipeline_controller.py:
    from processing.ingest_neo4j import run_neo4j_ingestion
    result = run_neo4j_ingestion(book_id, ready_path, str(BASE_DIR), neo4j_cb)
    # result = {"sections_written": N, "entities_written": M, ...}
"""

import re
import concurrent.futures
from parta.logger import time_it, async_time_it, logger
import json
import time
import uuid
import warnings
import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED CONFIGURATION — no environment variables per project rules
# ─────────────────────────────────────────────────────────────────────────────
NEO4J_URI  = "neo4j+s://95a8070a.databases.neo4j.io"
NEO4J_AUTH = ("95a8070a", "39TVuQIDdPNbNnVNgiWGzi_SVl17V-8hetw54nLyI0M")

# Worker count for knowledge-graph chunk processing.
# Keep 1 if GLiNER is unstable on your CPU; increase to 2/4 for faster graph injection.
NEO4J_WORKERS = 1

import json
import collections
from pathlib import Path

# GLiNER entity labels — dynamically loaded
ENTITY_LABELS = [
    "equipment",
    "metric",
    "organization",
    "location",
    "identifier",
    "concept",
    "person",
]

_json_path = Path(__file__).resolve().parent.parent.parent / "partb" / "unie_synthetic.json"
if _json_path.exists():
    try:
        with open(_json_path, 'r', encoding='utf-8') as f:
            _data = json.load(f)
        _all_labels = [
            item[2] for record in _data 
            for item in record.get('ner', []) 
            if '<>' not in item[2] and item[2].lower() != "match"
        ]
        _counter = collections.Counter(_all_labels)
        ENTITY_LABELS = [k for k, v in _counter.most_common(30)]
    except Exception as e:
        print(f"Warning: Failed to load labels from JSON: {e}")

# GLiNER confidence threshold
GLINER_THRESHOLD = 0.45

# Max characters per GLiNER batch window (same as original worker)
GLINER_MAX_CHARS = 500

# Minimum sentence length to run GLiNER on (avoids wasting time on fragments)
MIN_SENTENCE_CHARS = 20

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — SPECIFICATION REGEX
# ─────────────────────────────────────────────────────────────────────────────
# Matches patterns like:
#   "thrust 799 kN"
#   "chamber pressure of 58.5 bar"
#   "operating at 17,000 RPM"
#   "data rate 256 kbps"
#   "temperature 3200 °C"

SPEC_UNITS = (
    r"rpm|kn|kpa|mpa|bar|kg|kgs|g|mg|°c|°f|k|ghz|mhz|khz|hz|"
    r"kbps|mbps|gbps|bps|ms|ns|us|s|sec|min|h|hr|"
    r"mm|cm|m|km|nm|um|in|ft|"
    r"v|kv|mv|w|kw|mw|a|ma|ka|"
    r"n|nm|j|kj|mj|"
    r"pa|hpa|atm|psi|torr|"
    r"l|ml|cc|m3|"
    r"t|ton|lb|lbf|"
    r"deg|rad|"
    r"percent|%"
)

SPEC_PATTERN = re.compile(
    r"([\w][\w\s\-\/]{1,35}?)"          # subject: 2-36 chars, word chars + space/dash
    r"\s*(?:of|at|is|was|:|=)?\s*"       # optional connector
    r"([\d][,\d]*(?:\.\d+)?)"            # numeric value (supports commas like 17,000)
    r"\s*"
    r"(" + SPEC_UNITS + r")\b",          # unit from allowed list
    re.IGNORECASE,
)

# Subjects to ignore — too generic to be useful entities
SPEC_SUBJECT_IGNORE = {
    "a", "an", "the", "this", "that", "it", "he", "she", "they",
    "which", "where", "when", "about", "after", "with", "from",
    "total", "each", "per", "average", "maximum", "minimum", "typical",
    "approximately", "about", "around", "nearly", "roughly",
}


@time_it
def _extract_specifications(text: str) -> List[Dict]:
    """
    Runs the spec regex on a text block.
    Returns list of {subject, property_hint, value, unit, raw} dicts.
    Cleans up the subject string before returning.
    """
    specs = []
    seen  = set()

    for match in SPEC_PATTERN.finditer(text):
        subject_raw = match.group(1).strip()
        value_raw   = match.group(2).strip()
        unit_raw    = match.group(3).strip().lower()
        raw_text    = match.group(0).strip()

        # Clean subject
        subject = re.sub(r"\s+", " ", subject_raw).strip(" -:/,.")
        subject_lower = subject.lower()

        # Skip ignored subjects
        if subject_lower in SPEC_SUBJECT_IGNORE:
            continue
        if len(subject) < 3:
            continue
        # Skip subjects that start with a number
        if re.match(r"^\d", subject):
            continue

        # Parse numeric value — remove commas
        try:
            value_float = float(value_raw.replace(",", ""))
        except ValueError:
            continue

        # Deduplicate
        dedup_key = (subject_lower, value_raw, unit_raw)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        specs.append({
            "subject":       subject,
            "value":         value_float,
            "unit":          unit_raw,
            "raw":           raw_text,
        })

    return specs


# ─────────────────────────────────────────────────────────────────────────────
# GLINER SETUP — reused directly from neo4j_worker.py
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _load_gliner(base_dir: Path):
    """
    Loads GLiNER from portable/gliner/ — same logic as original worker.
    Returns the loaded model.
    """
    from gliner import GLiNER

    model_dir = base_dir / "portable" / "gliner"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"[NEO4J] GLiNER model not found at: {model_dir}\n"
            f"        Place offline GLiNER model in portable/gliner/"
        )

    print(f"[NEO4J] Loading GLiNER from {model_dir}...")
    model = GLiNER.from_pretrained(
        str(model_dir),
        local_files_only=True,
    ).to("cpu")
    print("[NEO4J] ✅ GLiNER loaded.")
    return model


@time_it
def _load_nltk(base_dir: Path):
    """
    Loads NLTK sent_tokenize with offline data path.
    Falls back to regex splitter if punkt not available.
    Same pattern as propositions.py.
    """
    import nltk

    nltk_data_path = base_dir / "portable" / "nltk_data"
    if nltk_data_path.exists():
        if str(nltk_data_path) not in nltk.data.path:
            nltk.data.path.insert(0, str(nltk_data_path))

    try:
        nltk.data.find("tokenizers/punkt")
        from nltk.tokenize import sent_tokenize
        return sent_tokenize
    except LookupError:
        try:
            nltk.data.find("tokenizers/punkt_tab")
            from nltk.tokenize import sent_tokenize
            return sent_tokenize
        except LookupError:
            print("[NEO4J] ⚠  NLTK punkt not found. Using regex splitter.")
            return _regex_sent_tokenize


@time_it
def _regex_sent_tokenize(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


@time_it
def _normalize_entity_name(name: str) -> str:
    """Lowercase and collapse whitespace — same as original worker."""
    return " ".join(name.strip().lower().split())


@time_it
def _run_gliner_on_text(
    gliner_model,
    text: str,
) -> List[Dict]:
    """
    Runs GLiNER on a text string using windowed batching.
    Windows of GLINER_MAX_CHARS with sentence boundaries respected.
    Returns list of {name, type} dicts (deduplicated).
    Same windowing logic as original neo4j_worker.py.
    """
    text = text.replace("\n", " ").strip()
    if not text:
        return []

    # Split into sentence-aware windows
    sentences = re.split(r"(?<=[.!?])\s+", text)
    batches   = []
    current   = ""

    for sent in sentences:
        if len(sent) > GLINER_MAX_CHARS:
            if current:
                batches.append(current.strip())
                current = ""
            for i in range(0, len(sent), GLINER_MAX_CHARS):
                chunk = sent[i: i + GLINER_MAX_CHARS]
                if len(chunk) > 10:
                    batches.append(chunk)
            continue
        if len(current) + len(sent) < GLINER_MAX_CHARS:
            current += " " + sent
        else:
            batches.append(current.strip())
            current = sent

    if current:
        batches.append(current.strip())

    # Run GLiNER on each batch
    entities = []
    seen     = set()

    for batch in batches:
        if len(batch) < 5:
            continue
        try:
            preds = gliner_model.predict_entities(
                batch, ENTITY_LABELS, threshold=GLINER_THRESHOLD
            )
        except Exception:
            continue

        for p in preds:
            norm = _normalize_entity_name(p["text"])
            key  = (norm, p["label"])
            if key in seen:
                continue
            seen.add(key)
            entities.append({
                "name": norm,
                "type": p["label"],
                "raw":  p["text"],
            })

    return entities


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — DOCUMENT HIERARCHY
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _build_hierarchy(
    session,
    chunks:     List[dict],
    book_id:    str,
    book_title: str,
):
    """
    Pass 1: builds the full document hierarchy from section_paths.

    Creates:
      (Book)──[HAS_CHAPTER]──►(Chapter)
      (Chapter)──[HAS_SECTION]──►(Section)
      (Section)──[HAS_SUBSECTION]──►(Subsection)
      (Section)──[NEXT_SECTION]──►(Section)  ← sequential navigation

    All writes are MERGE — safe to re-run on same book.
    """
    print(f"[NEO4J] Layer 1: Building document hierarchy...")

    # MERGE Book node
    session.run(
        """
        MERGE (b:Book {id: $bid})
        ON CREATE SET b.title = $title, b.ingested_at = datetime()
        ON MATCH  SET b.title = $title
        """,
        bid=book_id, title=book_title,
    )

    # Collect all unique section paths across all chunks
    # section_path = ["Ch3 Propulsion", "3.2 Vikas Engine", "3.2.1 Fuel"]
    seen_paths = set()
    ordered_sections = []  # for NEXT_SECTION edges

    for chunk in chunks:
        path  = chunk.get("section_path", [])
        level = chunk.get("level", 0)

        if not path or level == 0:
            continue

        path_key = ">>".join(path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        ordered_sections.append((level, path, chunk.get("chunk_id")))

        # Build nodes for each level of this path
        for depth in range(len(path)):
            sub_path = path[:depth + 1]
            node_name = sub_path[-1]
            parent_name = sub_path[-2] if depth > 0 else None

            if depth == 0:
                # Chapter level
                session.run(
                    """
                    MATCH (b:Book {id: $bid})
                    MERGE (c:Chapter {name: $name, book_id: $bid})
                    ON CREATE SET c.level = 1
                    MERGE (b)-[:HAS_CHAPTER]->(c)
                    """,
                    bid=book_id, name=node_name,
                )
            elif depth == 1:
                # Section level
                session.run(
                    """
                    MATCH (c:Chapter {name: $parent, book_id: $bid})
                    MERGE (s:Section {name: $name, book_id: $bid})
                    ON CREATE SET s.level = 2
                    MERGE (c)-[:HAS_SECTION]->(s)
                    """,
                    bid=book_id, name=node_name, parent=parent_name,
                )
            elif depth == 2:
                # Subsection level
                session.run(
                    """
                    MATCH (s:Section {name: $parent, book_id: $bid})
                    MERGE (ss:Subsection {name: $name, book_id: $bid})
                    ON CREATE SET ss.level = 3
                    MERGE (s)-[:HAS_SUBSECTION]->(ss)
                    """,
                    bid=book_id, name=node_name, parent=parent_name,
                )

    # Build NEXT_SECTION edges — connect sections sequentially
    # Sort by level-2 sections only (chapter-level sections)
    level2_sections = [
        path[1] for level, path, _ in ordered_sections
        if level >= 2 and len(path) >= 2
    ]
    # Deduplicate while preserving order
    seen = set()
    level2_ordered = []
    for s in level2_sections:
        if s not in seen:
            seen.add(s)
            level2_ordered.append(s)

    for i in range(len(level2_ordered) - 1):
        session.run(
            """
            MATCH (a:Section {name: $a, book_id: $bid})
            MATCH (b:Section {name: $b, book_id: $bid})
            MERGE (a)-[:NEXT_SECTION]->(b)
            """,
            bid=book_id,
            a=level2_ordered[i],
            b=level2_ordered[i + 1],
        )

    print(f"[NEO4J] ✅ Layer 1 done — "
          f"{len(seen_paths)} sections, "
          f"{len(level2_ordered)-1} NEXT_SECTION edges")


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — SPECIFICATION NODES
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _write_specifications(session, specs: List[Dict], section_name: str, book_id: str):
    """
    Writes Spec nodes and HAS_SPECIFICATION edges for one chunk's specs.
    """
    for spec in specs:
        session.run(
            """
            MERGE (e:Entity {name: $subject, book_id: $bid})
            ON CREATE SET e.type = 'equipment', e.source = 'spec_regex'

            MERGE (sp:Spec {
                subject: $subject,
                unit:    $unit,
                book_id: $bid
            })
            ON CREATE SET
                sp.value    = $value,
                sp.raw      = $raw,
                sp.section  = $section
            ON MATCH SET
                sp.value    = $value,
                sp.raw      = $raw

            MERGE (e)-[:HAS_SPECIFICATION]->(sp)
            """,
            bid=book_id,
            subject=spec["subject"].lower(),
            value=spec["value"],
            unit=spec["unit"],
            raw=spec["raw"],
            section=section_name,
        )


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — ENTITY-SECTION LINKS
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _write_entity_section_links(
    session,
    entities:     List[Dict],
    section_name: str,
    book_id:      str,
):
    """
    Writes Entity nodes and MENTIONED_IN edges for one chunk.
    Operates at section level — far more meaningful than page level.
    """
    for ent in entities:
        # Try to match section or subsection
        session.run(
            """
            MERGE (e:Entity {name: $name, book_id: $bid})
            ON CREATE SET e.type = $etype, e.raw = $raw
            ON MATCH  SET
                e.type = COALESCE(e.type, $etype),
                e.raw  = COALESCE(e.raw,  $raw)

            WITH e
            OPTIONAL MATCH (s:Section {name: $sname, book_id: $bid})
            OPTIONAL MATCH (ss:Subsection {name: $sname, book_id: $bid})
            WITH e,
                 COALESCE(s, ss) AS section_node
            WHERE section_node IS NOT NULL
            MERGE (e)-[:MENTIONED_IN]->(section_node)
            """,
            bid=book_id,
            name=ent["name"],
            etype=ent["type"],
            raw=ent.get("raw", ent["name"]),
            sname=section_name,
        )


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — SENTENCE CO-OCCURRENCE (accumulated in memory, written in batch)
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _write_cooccurrence_batch(
    session,
    cooccurrence: Dict[Tuple, Dict],
    book_id: str,
):
    """
    Writes all accumulated SENTENCE_CO_OCCURS edges in one pass.

    cooccurrence keys: (entity_a_name, entity_b_name)  (always sorted)
    cooccurrence values: {count: int, sections: set}
    """
    print(f"[NEO4J] Layer 4: Writing {len(cooccurrence)} "
          f"sentence co-occurrence edges...")

    written = 0
    for (name_a, name_b), data in cooccurrence.items():
        sections_list = list(data["sections"])[:20]  # cap at 20 section names

        session.run(
            """
            MATCH (a:Entity {name: $na, book_id: $bid})
            MATCH (b:Entity {name: $nb, book_id: $bid})
            MERGE (a)-[r:SENTENCE_CO_OCCURS]->(b)
            ON CREATE SET
                r.count    = $cnt,
                r.sections = $secs
            ON MATCH SET
                r.count    = r.count + $cnt,
                r.sections = r.sections + $secs
            """,
            bid=book_id,
            na=name_a,
            nb=name_b,
            cnt=data["count"],
            secs=sections_list,
        )
        written += 1

    print(f"[NEO4J] ✅ Layer 4 done — {written} co-occurrence edges written")


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5 — TABLE NODES
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _write_table_nodes(
    session,
    chunk:   dict,
    book_id: str,
):
    """
    Writes Table and TableRow nodes for one table chunk.
    Links Table to its parent Section or Subsection.
    """
    structured = chunk.get("structured_json", {})
    headers    = structured.get("headers", [])
    rows       = structured.get("rows", [])

    if not headers or not rows:
        return  # Unparseable table — skip silently

    section_path = chunk.get("section_path", [])
    section_name = section_path[-1] if section_path else "Unknown"
    parent_section = (
        section_path[-2] if len(section_path) >= 2 else section_name
    )

    # Unique table ID
    table_id = chunk.get("chunk_id", str(uuid.uuid4()))
    table_title = f"{section_name} — Table"

    # MERGE Table node linked to parent Section/Subsection
    session.run(
        """
        MERGE (t:Table {id: $tid, book_id: $bid})
        ON CREATE SET
            t.title     = $title,
            t.headers   = $headers,
            t.row_count = $rcount,
            t.section   = $sname
        ON MATCH SET
            t.title     = $title,
            t.row_count = $rcount

        WITH t
        OPTIONAL MATCH (s:Section    {name: $psname, book_id: $bid})
        OPTIONAL MATCH (ss:Subsection{name: $psname, book_id: $bid})
        WITH t, COALESCE(s, ss) AS parent
        WHERE parent IS NOT NULL
        MERGE (parent)-[:HAS_TABLE]->(t)
        """,
        bid=book_id,
        tid=table_id,
        title=table_title,
        headers=headers,
        rcount=len(rows),
        sname=section_name,
        psname=parent_section,
    )

    # MERGE TableRow nodes
    for row_idx, row in enumerate(rows):
        row_id = f"{table_id}_row_{row_idx}"

        # Build row properties — all header values stored flat
        # Also pick out parameter/value/unit for the most common pattern
        lowered = {k.lower().strip(): v for k, v in row.items()}
        param = (
            lowered.get("parameter") or
            lowered.get("item")      or
            lowered.get("name")      or
            lowered.get("description") or
            next(iter(row.values()), "")
        )
        value = (
            lowered.get("value")  or
            lowered.get("values") or
            lowered.get("data")   or
            ""
        )
        unit = lowered.get("unit") or lowered.get("units") or ""

        session.run(
            """
            MERGE (r:TableRow {id: $rid, book_id: $bid})
            ON CREATE SET
                r.parameter  = $param,
                r.value      = $value,
                r.unit       = $unit,
                r.row_data   = $raw_row,
                r.row_index  = $ridx

            WITH r
            MATCH (t:Table {id: $tid, book_id: $bid})
            MERGE (t)-[:HAS_ROW]->(r)
            """,
            bid=book_id,
            rid=row_id,
            tid=table_id,
            param=str(param),
            value=str(value),
            unit=str(unit),
            raw_row=json.dumps(row),
            ridx=row_idx,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def _neo4j_chunk_worker(args) -> Dict:
    """Worker unit for graph extraction. Does CPU/model work only; Neo4j writes stay in main thread."""
    idx, chunk, book_id, gliner_model, sent_tokenize = args
    content = chunk.get("content", "").strip()
    chunk_type = chunk.get("type", "text")
    section_path = chunk.get("section_path", [])
    section_name = section_path[-1] if section_path else "Unknown"
    out = {"idx": idx, "chunk": chunk, "section_name": section_name, "specs": [], "entities": [], "pairs": []}
    if not content:
        return out

    if chunk_type == "text":
        out["specs"] = _extract_specifications(content)

    out["entities"] = _run_gliner_on_text(gliner_model, content)

    if chunk_type == "text":
        try:
            sentences = sent_tokenize(content)
        except Exception:
            sentences = _regex_sent_tokenize(content)
        for sentence in sentences:
            if len(sentence) < MIN_SENTENCE_CHARS:
                continue
            sent_entities = _run_gliner_on_text(gliner_model, sentence)
            sent_names = [e["name"] for e in sent_entities]
            for i in range(len(sent_names)):
                for j in range(i + 1, len(sent_names)):
                    a, b = sent_names[i], sent_names[j]
                    out["pairs"].append((min(a, b), max(a, b), section_name))
    return out


@time_it
def _process_chunks(
    session,
    chunks:       List[dict],
    book_id:      str,
    gliner_model,
    sent_tokenize,
    progress_callback,
) -> Dict:
    """
    Pass 2: worker-style chunk processing for Layers 2, 3, 4, 5.
    Workers extract specs/entities/co-occurrence; main thread writes to Neo4j
    so the Neo4j session is never shared unsafely.
    """
    total = len(chunks)
    entities_seen = specs_seen = tables_seen = 0
    cooccurrence: Dict[Tuple, Dict] = defaultdict(lambda: {"count": 0, "sections": set()})
    worker_count = max(1, int(NEO4J_WORKERS))
    logger.info("[NEO4J] Starting graph worker pool | workers=%s | chunks=%s | book=%s", worker_count, total, book_id)

    def handle_result(result: Dict):
        nonlocal entities_seen, specs_seen, tables_seen
        chunk = result["chunk"]
        section_name = result["section_name"]
        if result["specs"]:
            _write_specifications(session, result["specs"], section_name, book_id)
            specs_seen += len(result["specs"])
        if result["entities"]:
            _write_entity_section_links(session, result["entities"], section_name, book_id)
            entities_seen += len(result["entities"])
        for a, b, sec in result["pairs"]:
            cooccurrence[(a, b)]["count"] += 1
            cooccurrence[(a, b)]["sections"].add(sec)
        if chunk.get("type") == "table" and chunk.get("content", "").strip():
            _write_table_nodes(session, chunk, book_id)
            tables_seen += 1

    def report(done: int):
        pct = 82 + int((done / max(total, 1)) * 13)
        msg = f"Graph workers processed: {done}/{total} | Entities: {entities_seen} | Specs: {specs_seen} | Tables: {tables_seen}"
        if progress_callback:
            progress_callback(percent=min(pct, 94), stage="Graph Ingestion", message=msg, extra={"chunks_done": done, "total_chunks": total, "workers": worker_count})
        logger.info("[NEO4J] %s", msg)

    args_iter = [(idx, chunk, book_id, gliner_model, sent_tokenize) for idx, chunk in enumerate(chunks)]
    done = 0
    if worker_count == 1:
        for args in args_iter:
            handle_result(_neo4j_chunk_worker(args))
            done += 1
            if done % 25 == 0 or done == total:
                report(done)
    else:
        # Thread workers keep one shared loaded model in memory. If your GLiNER build is not
        # thread-safe, set RAG_NEO4J_WORKERS=1.
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="neo4j-kg-worker") as ex:
            futures = [ex.submit(_neo4j_chunk_worker, args) for args in args_iter]
            for fut in concurrent.futures.as_completed(futures):
                handle_result(fut.result())
                done += 1
                if done % 25 == 0 or done == total:
                    report(done)

    logger.info("[NEO4J] Graph worker pool complete | chunks=%s | entities=%s | specs=%s | tables=%s | cooc=%s", total, entities_seen, specs_seen, tables_seen, len(cooccurrence))
    return {
        "entities_written":     entities_seen,
        "specs_written":        specs_seen,
        "tables_written":       tables_seen,
        "cooccurrence":         cooccurrence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT — called by pipeline_controller.py
# ─────────────────────────────────────────────────────────────────────────────

@time_it
def run_neo4j_ingestion(
    book_id:           str,
    ready_path:        str,
    base_dir:          str,
    progress_callback = None,
) -> Dict:
    """
    Main entry point called by pipeline_controller.py

    Args:
        book_id    : e.g. "PSLV-C50"
        ready_path : path to {book_id}_ready.json
        base_dir   : project root
        progress_callback : optional fn(percent, stage, message, extra=None)

    Returns:
        dict with keys:
            sections_written, entities_written, specs_written,
            tables_written, cooccurrence_edges

    Raises:
        FileNotFoundError if ready_path or gliner model missing
        RuntimeError if Neo4j connection fails
    """
    from neo4j import GraphDatabase

    ready_file = Path(ready_path)
    base       = Path(base_dir)

    if not ready_file.exists():
        raise FileNotFoundError(
            f"[NEO4J] _ready.json not found: {ready_path}\n"
            f"        Run chunk.py and triple_rep.py first."
        )

    if progress_callback:
        progress_callback(
            percent=81,
            stage="Graph Ingestion",
            message="Connecting to Neo4j and loading models...",
        )

    # ── Connect to Neo4j ──────────────────────────────────────────────────────
    logger.info("[NEO4J] Starting graph ingestion | book=%s | ready_path=%s", book_id, ready_path)
    print(f"\n[NEO4J] Connecting to {NEO4J_URI}...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, max_connection_lifetime=200, keep_alive=True)
        driver.verify_connectivity()
        print("[NEO4J] ✅ Neo4j connected.")
        logger.info("[NEO4J] Neo4j connected | uri=%s", NEO4J_URI)
    except Exception as e:
        raise RuntimeError(
            f"[NEO4J] Cannot connect to Neo4j at {NEO4J_URI}.\n"
            f"        Is Neo4j running? Error: {e}"
        )

    # ── Load models ───────────────────────────────────────────────────────────
    gliner_model  = _load_gliner(base)
    sent_tokenize = _load_nltk(base)

    # ── Load data ─────────────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(
            percent=82,
            stage="Graph Ingestion",
            message=f"Reading chunks for {book_id}...",
        )

    with open(ready_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise RuntimeError(f"[NEO4J] No chunks found in {ready_path}")

    book_title = chunks[0].get("book_title", book_id)
    text_chunks  = [c for c in chunks if c.get("type") == "text"]
    table_chunks = [c for c in chunks if c.get("type") == "table"]

    print(f"[NEO4J] Book: '{book_title}'")
    print(f"[NEO4J] {len(chunks)} chunks — "
          f"{len(text_chunks)} text, {len(table_chunks)} tables")
    logger.info("[NEO4J] Loaded chunks | book=%s | total=%s | text=%s | tables=%s", book_id, len(chunks), len(text_chunks), len(table_chunks))

    t_start = time.perf_counter()

    # ── All writes in one session ─────────────────────────────────────────────
    with driver.session() as session:

        # Layer 1 — Document Hierarchy
        if progress_callback:
            progress_callback(
                percent=82,
                stage="Graph Ingestion",
                message="Building document hierarchy...",
            )
        _build_hierarchy(session, chunks, book_id, book_title)

        # Layers 2, 3, 4, 5 — per chunk processing
        if progress_callback:
            progress_callback(
                percent=83,
                stage="Graph Ingestion",
                message=f"Processing {len(chunks)} chunks for entities, "
                        f"specs, co-occurrence and tables...",
            )

        stats = _process_chunks(
            session, chunks, book_id,
            gliner_model, sent_tokenize,
            progress_callback,
        )

        # Layer 4 — flush co-occurrence edges
        if stats["cooccurrence"]:
            _write_cooccurrence_batch(
                session, stats["cooccurrence"], book_id
            )

    # ── Write log ─────────────────────────────────────────────────────────────
    elapsed  = time.perf_counter() - t_start
    cooc_ct  = len(stats["cooccurrence"])

    log = {
        "book_id":             book_id,
        "book_title":          book_title,
        "total_chunks":        len(chunks),
        "sections_written":    len(set(
            ">>".join(c.get("section_path", []))
            for c in chunks if c.get("section_path")
        )),
        "entities_written":    stats["entities_written"],
        "specs_written":       stats["specs_written"],
        "tables_written":      stats["tables_written"],
        "cooccurrence_edges":  cooc_ct,
        "elapsed_seconds":     round(elapsed, 1),
    }

    log_dir = base / "data" / "neo4j"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{book_id}_neo4j_log.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    logger.info("[NEO4J] Wrote graph ingestion log | path=%s", log_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n[NEO4J] ═══ Graph Ingestion Complete ═══")
    print(f"         Sections  : {log['sections_written']}")
    print(f"         Entities  : {stats['entities_written']}")
    print(f"         Specs     : {stats['specs_written']}")
    print(f"         Tables    : {stats['tables_written']}")
    print(f"         Co-occurs : {cooc_ct}")
    print(f"         Time      : {elapsed:.1f}s")
    logger.info("[NEO4J] Graph ingestion complete | book=%s | sections=%s | entities=%s | specs=%s | tables=%s | cooc=%s | elapsed=%.1fs", book_id, log['sections_written'], stats['entities_written'], stats['specs_written'], stats['tables_written'], cooc_ct, elapsed)

    if progress_callback:
        progress_callback(
            percent=95,
            stage="Graph Ingestion",
            message=(
                f"Graph complete — {stats['entities_written']} entities, "
                f"{stats['specs_written']} specs, "
                f"{stats['tables_written']} tables, "
                f"{cooc_ct} co-occurrence edges in {elapsed:.1f}s."
            ),
        )

    driver.close()
    return log


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE MODE — run directly for debugging
# python processing/ingest_neo4j.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    BASE_DIR = Path(__file__).resolve().parent.parent

    def _print_callback(percent, stage, message, extra=None):
        print(f"  [{percent}%] {stage}: {message}")

    checkpoint_dir = BASE_DIR / "data" / "checkpoints"
    ready_files    = sorted(checkpoint_dir.glob("*_ready.json"))

    if not ready_files:
        print("[NEO4J] No *_ready.json files found.")
        print("        Run chunk.py and triple_rep.py first.")
        sys.exit(1)

    for rf in ready_files:
        book_id = rf.stem.replace("_ready", "")
        print(f"\n{'='*60}")
        print(f"  Ingesting: {book_id}")
        print(f"{'='*60}")

        try:
            result = run_neo4j_ingestion(
                book_id    = book_id,
                ready_path = str(rf),
                base_dir   = str(BASE_DIR),
                progress_callback = _print_callback,
            )
            print(f"\n  ✅ Done: {result}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
