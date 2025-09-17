# streamlit_rag_chatbot_chroma.py
"""
Streamlit RAG chatbot (Chroma-based) — updated

Changes:
- Ensure 'source' metadata is always present for ingested chunks (no [null] entries).
- Streaming-style typing effect when showing assistant responses.
- Cleaner markdown formatting for answers and sources.
- Minimal changes to preserve original behavior.
"""

import os
import re
import io
import json
import time
import requests
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from typing import List, Tuple

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# ------------------------
# Config + env
# ------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env. Put OPENAI_API_KEY=sk-...in your .env file.")

PERSIST_DIR = "chroma_store"
CHAT_HISTORY_DIR = "chat_histories"
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# params
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
SIMILARITY_THRESHOLD = 0.18  # adjust per your data
STREAM_DELAY = 0.02  # seconds between tokens for typing effect

# ------------------------
# Utilities
# ------------------------
def safe_collection_name(user_id: str) -> str:
    return "user_" + re.sub(r"[^a-zA-Z0-9]", "_", (user_id or "guest")).lower()

def pdf_bytes_to_text(b: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(b))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    except Exception:
        try:
            return b.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def csv_bytes_to_text(b: bytes) -> str:
    try:
        df = pd.read_csv(io.BytesIO(b))
        return df.to_csv(index=False)
    except Exception:
        try:
            return b.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def html_bytes_to_text(b: bytes) -> str:
    try:
        soup = BeautifulSoup(b, "html.parser")
        for tag in soup(["script","style","noscript"]):
            tag.decompose()
        texts = [tag.get_text(separator=" ", strip=True) for tag in soup.find_all(["h1","h2","h3","p","li"])]
        joined = "\n\n".join([t for t in texts if t])
        return joined if joined else soup.get_text(separator="\n")
    except Exception:
        try:
            return b.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def fetch_url_text(url: str, timeout: int = 20) -> Tuple[str, str]:
    try:
        r = requests.get(url, headers={"User-Agent":"RAG-Chatbot/1.0"}, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        return (f"[Error fetching URL: {e}]", url)
    ctype = r.headers.get("content-type","").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return (pdf_bytes_to_text(r.content), r.url)
    if "csv" in ctype or url.lower().endswith(".csv"):
        return (csv_bytes_to_text(r.content), r.url)
    return (html_bytes_to_text(r.content), r.url)

def extract_text_from_uploaded(uploaded_file) -> str:
    raw = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return pdf_bytes_to_text(raw)
    if name.endswith(".csv"):
        return csv_bytes_to_text(raw)
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def build_docs_from_text_and_source(text: str, source: str) -> List[Document]:
    """
    Create chunked Documents and ensure metadata['source'] is always a non-null string.
    """
    source_label = source or "unknown_source"
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text or "")
    docs = []
    for chunk in chunks:
        if not chunk or not chunk.strip():
            continue
        md = {"source": source_label}
        docs.append(Document(page_content=chunk, metadata=md))
    return docs

# ------------------------
# Similarity util
# ------------------------
def cosine_similarity(a: List[float], b: List[float]) -> float:
    import numpy as np
    a = np.array(a, dtype="float32")
    b = np.array(b, dtype="float32")
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(float(a.dot(b)) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ------------------------
# Vector store & history helpers
# ------------------------
def get_embeddings_client():
    try:
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    except TypeError:
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

def load_vectorstore_for_user(user_id: str, embeddings: OpenAIEmbeddings) -> Chroma:
    return Chroma(collection_name=safe_collection_name(user_id), embedding_function=embeddings, persist_directory=PERSIST_DIR)

def persist_vectorstore(vs: Chroma):
    try:
        vs.persist()
    except Exception:
        pass

def save_chat_history(user_id: str, history):
    path = os.path.join(CHAT_HISTORY_DIR, f"{safe_collection_name(user_id)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_chat_history(user_id: str):
    path = os.path.join(CHAT_HISTORY_DIR, f"{safe_collection_name(user_id)}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def delete_user_memory(user_id: str):
    try:
        vs = Chroma(collection_name=safe_collection_name(user_id), embedding_function=get_embeddings_client(), persist_directory=PERSIST_DIR)
        if hasattr(vs, "delete_collection"):
            try:
                vs.delete_collection()
            except Exception:
                pass
    except Exception:
        pass
    hist_path = os.path.join(CHAT_HISTORY_DIR, f"{safe_collection_name(user_id)}.json")
    if os.path.exists(hist_path):
        try:
            os.remove(hist_path)
        except Exception:
            pass
    if "chat_history" in st.session_state:
        st.session_state.chat_history = []

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot — context + fallback (streaming + clean sources)")

# Sidebar controls
with st.sidebar:
    st.header("User & Controls")
    user_id = st.text_input("Enter your email or user id:", value="guest")
    if not user_id:
        st.stop()

    st.subheader("Ingest documents")
    uploaded_files = st.file_uploader("Upload files (PDF / CSV / TXT / MD)", accept_multiple_files=True)
    st.markdown("Or paste links (one per line)")
    links_text = st.text_area("Paste https links here (one per line)")

    if st.button("Ingest selected files & links"):
        with st.spinner("Ingesting and creating embeddings..."):
            embeddings = get_embeddings_client()
            vs = load_vectorstore_for_user(user_id, embeddings)
            total_added = 0

            # files
            for f in uploaded_files or []:
                try:
                    text = extract_text_from_uploaded(f)
                    docs = build_docs_from_text_and_source(text, source=f.name or "uploaded_file")
                    if docs:
                        vs.add_documents(docs)
                        total_added += len(docs)
                except Exception as e:
                    st.warning(f"Failed to ingest {f.name}: {e}")

            # links (single page fetch)
            for line in (links_text or "").splitlines():
                url = line.strip()
                if not url:
                    continue
                try:
                    text, final_url = fetch_url_text(url)
                    # ensure final_url is a valid string
                    src = final_url or url or "url_source"
                    docs = build_docs_from_text_and_source(text, source=src)
                    if docs:
                        vs.add_documents(docs)
                        total_added += len(docs)
                except Exception as e:
                    st.warning(f"Failed to ingest {url}: {e}")

            persist_vectorstore(vs)
            st.success(f"Ingestion finished — added {total_added} chunks.")

    st.markdown("---")
    if st.button("🗑️ Reset Memory"): 
        delete_user_memory(user_id)
        st.success("Memory cleared.")

# ------------------------
# Init
# ------------------------
embeddings = get_embeddings_client()
vectorstore = load_vectorstore_for_user(user_id, embeddings)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history(user_id)

# LLM init - keep non-streaming model call; we'll simulate streaming display below
try:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4.1-mini", temperature=0)
except TypeError:
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4.1-mini", temperature=0)

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)

# Show history
st.subheader(f"Chat — {user_id}")
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(msg["user"])
    with st.chat_message("assistant"):
        # display saved markdown (we saved answer as string before)
        st.markdown(msg["bot"])
        if msg.get("sources"):
            with st.expander("Sources (saved)"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

# Chat input
prompt = st.chat_input("Ask a question...")
if prompt:
    # exact-match cache to avoid duplicate LLM calls
    cached = next((h for h in st.session_state.chat_history if h["user"].strip().lower() == prompt.strip().lower()), None)
    if cached:
        answer, sources = cached["bot"], cached.get("sources", [])
    else:
        # always include chat_history for the chain
        hist_for_chain = [(h["user"], h["bot"]) for h in st.session_state.chat_history]

        # 1) retrieve local top-K candidate chunks
        try:
            docs = vectorstore.similarity_search(prompt, k=TOP_K)
        except Exception:
            # fallback if method signature differs
            try:
                docs = retriever.get_relevant_documents(prompt)
            except Exception:
                docs = []

        # 2) compute embeddings for query and for candidate docs (re-embed top-k docs)
        try:
            # query embedding
            try:
                q_emb = embeddings.embed_query(prompt)
            except Exception:
                q_emb = embeddings.embed_documents([prompt])[0]

            doc_texts = [d.page_content for d in docs]
            if doc_texts:
                try:
                    doc_embs = embeddings.embed_documents(doc_texts)
                except Exception:
                    # fallback: embed each document separately
                    doc_embs = [embeddings.embed_query(t) for t in doc_texts]
            else:
                doc_embs = []

            sims = [cosine_similarity(q_emb, de) for de in doc_embs] if doc_embs else []
            max_sim = max(sims) if sims else 0.0
        except Exception:
            max_sim = 0.0
            sims = []

        # prepare default values
        answer = ""
        sources = []

        # 3) Decide: fallback to model knowledge or use RAG chain
        if max_sim < SIMILARITY_THRESHOLD:
            # Model-only fallback; instruct model to format answer
            system_prompt = (
                "You are a helpful assistant. The user asked a question that does not match the provided documents. "
                "Answer from your model knowledge only. Format the answer into clear sections or bullet points. "
                "Keep the answer concise."
            )
            try:
                # try different invocation methods across LangChain versions
                try:
                    resp = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])
                    # resp might be an object with different attrs; try to read content robustly
                    answer = getattr(resp, "content", None) or (resp[0].get("content") if isinstance(resp, (list,tuple)) and resp else None) or str(resp)
                except Exception:
                    # fallback to calling llm with single string prompt
                    resp = llm.invoke(f"{system_prompt}\n\nUser: {prompt}\nAssistant:")
                    answer = getattr(resp, "content", None) or str(resp)
            except Exception as e:
                answer = f"Error: {e}"
            sources = ["Model Knowledge (no relevant context found)"]
        else:
            # use retrieval chain
            try:
                res = qa_chain({"question": prompt, "chat_history": hist_for_chain})
                answer = res.get("answer") or res.get("result") or res.get("output_text") or ""
                sdocs = res.get("source_documents") or []
                # normalize sources: never return null
                norm_sources = []
                for sd in sdocs:
                    md = getattr(sd, "metadata", None) or {}
                    src = md.get("source") if isinstance(md, dict) else None
                    if not src:
                        # try other attributes
                        src = getattr(sd, "source", None) or getattr(sd, "id", None) or "Unknown"
                    norm_sources.append(str(src))
                # deduplicate while preserving order
                seen = set()
                final_sources = []
                for s in norm_sources:
                    if s not in seen:
                        seen.add(s)
                        final_sources.append(s)
                sources = final_sources
            except Exception as e:
                answer = f"Error during retrieval/LLM call: {e}"
                sources = []

    # Save to history
    entry = {"user": prompt, "bot": answer, "sources": sources}
    st.session_state.chat_history.append(entry)
    save_chat_history(user_id, st.session_state.chat_history)

    # Display messages with streaming typing effect (simulated)
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        # Format answer into markdown sections if not already formatted
        display_text = answer.strip()
        # Simple normalization: if answer is long single-line, try to insert Markdown bullets for sentences.
        # But we mainly let model format it. We'll just stream the final markdown string.
        placeholder = st.empty()
        stream_text = ""
        # Stream by words for typing effect
        for token in display_text.split():
            stream_text += token + " "
            # show a caret to indicate typing
            placeholder.markdown(stream_text + "▌")
            time.sleep(STREAM_DELAY)
        # final write (remove caret)
        placeholder.markdown(stream_text.strip())

        # Sources expander
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- {s}")

