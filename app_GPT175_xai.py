import os
os.environ["ANONYMOUS_TELEMETRY"] = "False"

from pathlib import Path
import os, re, json, hashlib
import streamlit as st

# LangChain
from langchain_core.documents import Document
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_text_splitters import CharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Optional cross-encoder re-ranker (local)
try:
    from sentence_transformers import CrossEncoder
    HAS_RERANKER = True
except Exception:
    HAS_RERANKER = False

# -----------------------
# Paths & Static Config
# -----------------------
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
MD_PATH     = DATA_DIR / "article_175_normalized.md"  # New Markdown source
JSONL_PATH  = DATA_DIR / "article_175_chunks.jsonl"  # New JSONL source
PROMPT_PATH = DATA_DIR / "gpt175_prompt.txt"
SYN_PATH    = DATA_DIR / "synonyms.json"


st.set_page_config(page_title="GPT-175 — NYC Radiation Control Assistant", layout="wide")
st.title("🩺 GPT-175 — NYC Radiation Control Code Assistant")
st.caption("Unofficial assistant for NYC radiation-control code (Article 175)")

# -----------------------
# Provider / Model UI
# -----------------------
provider = st.selectbox("Model provider", ["OpenAI", "xAI"], index=0)
if provider == "OpenAI":
    model_name = st.selectbox("OpenAI model", ["gpt-5", "gpt-4o-mini"], index=0)
else:
    model_name = st.selectbox("xAI model", ["grok-4-fast-reasoning", "grok-2"], index=0)

# Provider-scoped Chroma dir
CHROMA_DIR = DATA_DIR / f"chroma_store_{provider.lower()}"
BUILD_SIG  = CHROMA_DIR / ".build_signature"

# -----------------------
# API Key (provider-aware)
# -----------------------
def get_api_key(provider: str) -> str | None:
    if provider == "OpenAI":
        if key := os.getenv("OPENAI_API_KEY"):
            return key
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
        return st.text_input("Enter your OpenAI API key", type="password")
    else:
        if key := os.getenv("XAI_API_KEY"):
            return key
        if "XAI_API_KEY" in st.secrets:
            return st.secrets["XAI_API_KEY"]
        return st.text_input("Enter your xAI API key", type="password")

api_key = get_api_key(provider)
if not api_key:
    st.error(f"{provider} API key not set. Add it via env var or .streamlit/secrets.toml.")
    st.stop()

# Quick probe (provider-aware)
try:
    if provider == "OpenAI":
        _probe = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
    else:
        _probe = ChatOpenAI(model=model_name, api_key=api_key, openai_api_base="https://api.x.ai/v1", temperature=0)
    _ = _probe.invoke("ok").content
    st.caption(f"{provider} model probe OK.")
except Exception as e:
    st.error(f"{provider} model probe failed: {e}")
    st.stop()

# -----------------------
# Load system prompt & synonyms
# -----------------------
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else (
    "You are GPT175, a strict, citation-first assistant for Article 175. "
    "Answer NOT ONLY from the provided context and DO include citations."
)
SYN = {}
if SYN_PATH.exists():
    try:
        SYN = json.loads(SYN_PATH.read_text(encoding="utf-8"))
    except Exception:
        SYN = {}

# Q&A Memory for Gradual Learning (store/load verified examples)
QA_MEMORY_PATH = DATA_DIR / "qa_memory.json"
qa_memory = []
if QA_MEMORY_PATH.exists():
    qa_memory = json.loads(QA_MEMORY_PATH.read_text(encoding="utf-8"))

def expand_query(q: str) -> str:
    q_low = q.lower()
    expansions = []
    for cat in ["device_synonyms", "concept_synonyms"]:
        for k, vals in SYN.get(cat, {}).items():
            if any(term.lower() in q_low for term in [k] + vals):
                expansions.extend(vals)
    for m in re.findall(r"175\.\d+", q):
        expansions.append(f"§{m}")
    if "how long" in q_low or "duration" in q_low:
        expansions.extend(["week", "continuously", "greater than", "fixed installation"])
    if expansions:
        q = q.strip() + " " + " ".join(sorted(set(expansions)))
    return q

# -----------------------
# Embeddings (provider-aware)
# -----------------------
def make_embeddings(provider: str, api_key: str):
    if provider == "OpenAI":
        try:
            return OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
        except Exception:
            pass
    # xAI (or fallback): use local HF with increased timeout and retries
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
        # model_kwargs={"timeout": 30, "retry": 5},  # Increased timeout + retries
        # local_files_only=False  # Use if manually downloaded
    )

# Embed Q&A for retrieval (separate small vector store)
# -----------------------
# QA memory store (isolated per provider+model)
# -----------------------
qa_vs = None
if qa_memory:
    qa_docs = [Document(page_content=f"Q: {qa['query']}\nA: {qa['answer']}", metadata={"source": "memory"}) for qa in qa_memory]
    qa_emb = make_embeddings(provider, api_key)

    embed_tag = type(qa_emb).__name__  # e.g., OpenAIEmbeddings or HuggingFaceEmbeddings
    qa_dir = DATA_DIR / f"chroma_qa_{provider.lower()}_{embed_tag}"
    qa_dir.mkdir(parents=True, exist_ok=True)

    qa_collection = f"qa_memory_{provider.lower()}_{embed_tag}"

    try:
        from chromadb.config import Settings
        qa_vs = Chroma.from_documents(
            qa_docs,
            qa_emb,
            collection_name=qa_collection,
            persist_directory=str(qa_dir),
            client_settings=Settings(anonymized_telemetry=False),
        )
    except Exception as e:
        # If you changed models mid-run and the folder has old dims, uncomment the cleanup:
        # import shutil
        # shutil.rmtree(qa_dir, ignore_errors=True)
        # qa_vs = Chroma.from_documents(
        #     qa_docs, qa_emb, collection_name=qa_collection,
        #     persist_directory=str(qa_dir),
        #     client_settings=Settings(anonymized_telemetry=False),
        # )
        raise


# -----------------------
# Metadata helpers (fix Chroma scalar constraint)
# -----------------------
def to_scalar_meta(meta: dict) -> dict:
    out = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = "" if v is None else v
        elif k == "pages":
            out[k] = ",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)
        else:
            out[k] = str(v)
    return out

def meta_pages(meta) -> list[int]:
    start = meta.get("page_start")
    end = meta.get("page_end")
    try:
        if start is not None:
            start_i = int(start)
            end_i = int(end) if end is not None else start_i
            if end_i >= start_i:
                return list(range(start_i, end_i + 1))
    except Exception:
        pass
    v = meta.get("pages")
    if isinstance(v, list):
        out = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out
    if isinstance(v, str):
        return [int(x) for x in re.findall(r"\d+", v)]
    return []


# -----------------------
# Change-aware Vector Store build
# -----------------------
def file_md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def current_signature(provider: str, embed_tag: str) -> str:
    parts = [
        ("md", file_md5(MD_PATH)) if MD_PATH.exists() else ("md", "missing"),
        ("jsonl", file_md5(JSONL_PATH)) if JSONL_PATH.exists() else ("jsonl", "missing")
    ]
    if PROMPT_PATH.exists():
        parts.append(("prompt", file_md5(PROMPT_PATH)))
    parts.append(("provider", provider))
    parts.append(("embed", embed_tag))
    return "|".join(f"{k}:{v}" for k, v in parts)

def bm25_retrieve_docs(bm25, query: str):
    for method in ("get_relevant_documents", "invoke", "_get_relevant_documents"):
        if hasattr(bm25, method):
            return getattr(bm25, method)(query)
    return []

@st.cache_resource(show_spinner=False)
def build_or_load_vs_and_bm25(provider: str, api_key: str):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    emb = make_embeddings(provider, api_key)
    embed_tag = type(emb).__name__
    sig = current_signature(provider, embed_tag)
    need_rebuild = True
    if BUILD_SIG.exists() and (CHROMA_DIR / "chroma.sqlite3").exists():
        if BUILD_SIG.read_text() == sig:
            need_rebuild = False

    if need_rebuild:
        # Clear existing store
        for p in CHROMA_DIR.glob("*"):
            try: p.unlink()
            except Exception: pass
        # Load pre-chunked JSONL
        raw_docs = []
        with JSONL_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                raw_docs.append(Document(
                    page_content=chunk["content"],
                    metadata={
                        "article": chunk["article"],
                        "section": chunk["section"],
                        "title": chunk["title"],
                        "heading_path": chunk["heading_path"],
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"]
                    }
                ))
        docs_for_chroma = [
            Document(page_content=d.page_content, metadata=to_scalar_meta(d.metadata))
            for d in raw_docs
        ]
        vs = Chroma.from_documents(
            docs_for_chroma,
            emb,
            persist_directory=str(CHROMA_DIR),
            client_settings=Settings(anonymized_telemetry=False),
        )
        BUILD_SIG.write_text(sig)
    else:
        vs = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=emb)

    # Build BM25 over JSONL corpus
    all_docs = vs.get(include=["metadatas", "documents"])
    bm25_docs = [
        Document(page_content=txt, metadata=meta)
        for txt, meta in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    bm25 = BM25Retriever.from_documents(bm25_docs)
    bm25.k = 15
    return vs, bm25

vs, bm25 = build_or_load_vs_and_bm25(provider, api_key)

# -----------------------
# Optional Cross-Encoder
# -----------------------
@st.cache_resource
def load_reranker():
    if not HAS_RERANKER:
        return None
    try:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None

reranker = load_reranker()

def reciprocal_rank_fusion(results_lists, k=40, weight=60):
    scores = {}
    for docs in results_lists:
        for rank, d in enumerate(docs):
            key = (d.page_content[:64], tuple(sorted(meta_pages(d.metadata))), d.metadata.get("section"))
            scores[key] = scores.get(key, 0) + 1.0 / (weight + rank + 1)
    exemplars = {}
    for docs in results_lists:
        for d in docs:
            key = (d.page_content[:64], tuple(sorted(meta_pages(d.metadata))), d.metadata.get("section"))
            if key not in exemplars:
                exemplars[key] = d
    ranked = sorted(exemplars.items(), key=lambda kv: scores[kv[0]], reverse=True)[:k]
    return [d for _, d in ranked]

def mmr_diversify(query, docs, top_k=8):
    seen_sections, seen_pages, out = set(), set(), []
    for d in docs:
        sec = d.metadata.get("section")
        pages = tuple(meta_pages(d.metadata))
        if (sec and sec not in seen_sections) or not sec:
            out.append(d)
            if sec: seen_sections.add(sec)
            seen_pages.update(pages)
        if len(out) >= top_k:
            break
    if len(out) < top_k:
        for d in docs:
            if d in out:
                continue
            pages = tuple(meta_pages(d.metadata))
            if not set(pages).issubset(seen_pages):
                out.append(d)
                seen_pages.update(pages)
            if len(out) >= top_k:
                break
    return out[:top_k]

def gate_docs(docs, max_n=12):
    exact = [d for d in docs if re.search(r"§\s*175\.", d.page_content)]
    return (exact or docs)[:max_n]

FEW_SHOT_EXAMPLES = """
Example Query: "For how long a portable/mobile x-ray unit must be used in one room to be categorized as a fixed x-ray unit? does this include mobile fluoro units like c-arms? what would be the difference in terms of physics testing, if a mobile x-ray unit be used in one fixed room or carried around in multiple areas?"
Example Response:
Direct answer: A mobile/portable radiographic x-ray system used continuously for >1 week in the same location is deemed a fixed radiographic installation and must meet fixed-installation operator-protection standards; Article 175 does not state an analogous “>1 week” rule for fluoroscopic (e.g., C-arm) units. 
Definitions
•   Mobile / Portable / Stationary — Mobile: wheeled base; Portable: hand-carried; Stationary: installed in a fixed location. (§175.08 “X-ray equipment” definitions.) 
Operator protection & “fixed” threshold (Radiography)
•   Threshold that triggers “fixed” — Mobile/portable radiographic units (excluding dental/podiatric) used continuously >1 week in the same location → “fixed radiographic installation”. (§175.46(h)(2).) 
•   Once deemed fixed — Must meet operator’s-booth standards for fixed radiographic equipment. (§175.46(h)(2) referencing fixed standards.) 
Fluoroscopy / C-arms
•   No 1-week rule stated — Article 175 does not specify that mobile C-arms become “fixed” after a duration of use in one room; instead, it sets mobile-fluoro technical conditions (e.g., image intensification required; beam must be intercepted; minimum SSD). (Not specified banner for a “fixed by time” rule; nearest related fluoroscopy provisions below.) 
•   Mobile-fluoro additional requirements — In absence of tabletop, maintain SSD ≥30 cm; image intensification required; beam operation interlocked to receptor; broader source-skin-distance rules: stationary ≥38 cm, mobile/portable ≥30 cm (with limited surgical exceptions). (§175.53(e), (p).) 
Usage restrictions (when mobiles are allowed)
•   Routine clinical use limited — By July 1, 2022, portable/mobile x-ray may be used only when transferring the patient to a stationary installation is impractical; routine use is prohibited except: hospital ER/trauma and non-ambulatory inpatients, plus patient homes/LTCFs. (§175.04(c).) 
Physics/QA & surveys — what changes if “fixed” vs roaming
•   Initial QA & room survey (registration stage) — Facilities mandated to have QA must submit a QMP initial QA report covering all tests required at any frequency for each unit, and a radiation-protection survey for each room housing a radiographic unit; this becomes applicable when a mobile radiographic unit is deemed fixed in that room. (§175.40(d)(2).) 
•   General QA program (diagnostic) — Facilities (except dental/podiatric/vet) must maintain a facility- and equipment-specific QA manual, perform QA tests per the manual, correct deficiencies, and keep records; this applies regardless of whether a unit is mobile or fixed. (§175.12(b)(1)–(3).) 
•   Fluoroscopy acceptance testing — All fluoroscopic units require QMP acceptance testing (covering all tests required at any frequency) before clinical use, except mini C-arms with II <6 in. (§175.53(a).) 
•   Operator-protection differences —
o   Fixed radiographic: exposure control permanently mounted in a protected area (operator’s booth). (§175.46(a).) 
o   Mobile/portable radiographic (not deemed fixed): operator must stand ≥2 m from patient or behind ≥2.1 m-high protective barrier, not in primary beam; personnel monitoring and ≥0.25 mm Pb protective garment required. (§175.46(h)(1),(3).) 
o   Fluoroscopy (including C-arms): follow §175.53 technical and dose-management provisions irrespective of room permanence (e.g., AKR/cumulative air-kerma display for post-2006 equipment; last-image-hold; resolution tests; patient-dose management for FGI). (§175.53.) 
Dosimetry/monitoring (when mobiles are used)
•   Radiography (mobile/portable) — Personnel monitoring required for all operators of mobile/portable radiographic units (excluding dental/podiatric). (§175.46(h)(3).) 
•   Fluoroscopy (any location) — All individuals working with medical fluoroscopic equipment must be monitored; badges processed monthly (quarterly allowed only for mini-C-arm-only workers). (§175.17(e)(4)–(5).) 
________________________________________
Not specified in Article 175: A time-based rule that converts mobile fluoroscopy/C-arms into “fixed” status. Nearest related sections: operator’s-booth & mobile radiography rules (§175.46(h)), and fluoroscopy equipment requirements (§175.53). 
Disclaimer: I am not legal counsel. This summary cites the regulation’s text; final compliance determinations rest with the NYC DOH.
"""

# -----------------------
# Prompt (strict, citation-first)
# -----------------------
STRICT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\nFollow these examples for structure and detail:\n" + FEW_SHOT_EXAMPLES),
    ("system",
     "Additional Rules: Answer ONLY from the supplied context. If the answer is not explicitly present, say "
     "'Not specified in Article 175. Closest related sections: …' and list them.\n"
     "Required format:\n"
     "1) Start with a one-sentence answer (or 'Not specified…').\n"
     "2) Then provide grouped bullets by theme. Each REQUIREMENT bullet MUST end with a citation like "
     "‘(e.g., Dosimetry: - Req1 (§{{section}}, p.{{page}}))’. Use the section and page numbers from context metadata.\n"
     "3) If sections overlap, explicitly cross-reference them (e.g., Fluoro-CT: Merge §175.53 + §175.55).\n"
     "4) Normalize terms per synonyms (e.g., mini C-arm as small-SID fluoro)."
    ),
    ("system", "Context:\n{context}"),
    ("human", "{input}")
])


# -----------------------
# LLM Factory (provider-aware)
# -----------------------
def make_default_llm(provider: str, api_key: str, model_id: str):
    kwargs = dict(model=model_id, api_key=api_key, temperature=0, top_p=1, timeout=60, max_retries=1)
    if provider == "xAI":
        kwargs["openai_api_base"] = "https://api.x.ai/v1"
    return ChatOpenAI(**kwargs)

# Version-safe coercion for retriever outputs
def ensure_docs(items):
    if not items:
        return []
    if isinstance(items, list) and items and isinstance(items[0], Document):
        return items
    if isinstance(items, list) and items and isinstance(items[0], str):
        return [Document(page_content=s, metadata={}) for s in items]
    if isinstance(items, str):
        return [Document(page_content=items, metadata={})]
    return items

# -----------------------
# UI (ASK + ANSWER)
# -----------------------
question = st.text_area(
    "Ask your question about Article 175:",
    placeholder="e.g., For how long must a mobile X-ray be in one room to be considered fixed? Does this include mobile C-arms?"
)

if question:
    with st.spinner("Retrieving context…"):
        expanded_q = expand_query(question)

        # Embedding retrieval (no fetch_k for broad compatibility)
        emb_retriever = vs.as_retriever(search_kwargs={"k": 30})
        emb_docs = ensure_docs(emb_retriever.invoke(expanded_q))
        bm25.k = 15
        # BM25 retrieval (version-safe)
        bm25_docs = ensure_docs(bm25_retrieve_docs(bm25, expanded_q))

        # Hybrid fuse + optional re-rank + gating + diversify
        fused = reciprocal_rank_fusion([emb_docs, bm25_docs], k=40)
        pre = fused[:40]
        if reranker and pre:
            pairs = [[expanded_q, d.page_content] for d in pre]
            try:
                scores = load_reranker().predict(pairs)
                pre = [d for d, _ in sorted(zip(pre, scores), key=lambda x: x[1], reverse=True)]
            except Exception:
                pass
        gated = gate_docs(pre, max_n=18)
        docs  = mmr_diversify(expanded_q, gated, top_k=8)

    # annotate pages for better citations
    for d in docs:
        pages_list = meta_pages(d.metadata)
        # d.page_content = f"(pages {','.join(map(str, pages_list))}) " + d.page_content

    with st.spinner("Answering…"):
        llm = make_default_llm(provider, api_key, model_name)
        doc_chain = create_stuff_documents_chain(llm=llm, prompt=STRICT_PROMPT)

        memory_docs = []
        if qa_vs:
            memory_retriever = qa_vs.as_retriever(search_kwargs={"k": 3})
            memory_docs = memory_retriever.invoke(question)  # List of Documents

        # Combine memory and docs as list of Documents
        full_context = memory_docs + docs

        out = doc_chain.invoke({"input": question, "context": full_context})
        answer_text = out.get("text", "") if isinstance(out, dict) else (out or "")

        # Post-process: ensure citations + show consulted sources
        cited = bool(re.search(r"§\s*175\.\d+.*p\.\s*\d+", answer_text))
        unique_sources, seen = [], set()
        for d in full_context:
            sec = d.metadata.get("section") or "Unlabeled"
            for p in meta_pages(d.metadata):
                key = (sec, p)
                if key not in seen:
                    seen.add(key)
                    unique_sources.append(f"{sec}, p.{p}")
        if not cited:
            answer_text += "\n\n_Notes: The answer above is constrained to the provided context._"
        if unique_sources:
            answer_text += "\n\n**Sources consulted:** " + "; ".join(unique_sources)
        answer_text += "\n\n**Disclaimer**: I am not legal counsel. This summary cites the regulation’s text; final compliance determinations rest with the NYC DOH."

        st.markdown("### 🧾 Answer")
        st.markdown(answer_text)

        with st.expander("Retrieved context (top chunks)"):
            for i, d in enumerate(docs, 1):
                pages_list = meta_pages(d.metadata)
                sec = d.metadata.get("section") or "Unlabeled"
                st.markdown(f"**Chunk {i} — {sec}, pages {','.join(map(str, pages_list))}**\n\n{d.page_content[:1200]}…")

    if st.button("Save this Q&A as Verified Example (for future learning)"):
        qa_memory.append({"query": question, "answer": answer_text})
        QA_MEMORY_PATH.write_text(json.dumps(qa_memory, indent=2))
        st.success("Saved! Model will 'learn' from this next time.")