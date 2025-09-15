# streamlit_rag_chatbot_chroma.py
"""
Complete Streamlit RAG chatbot (single-file)

Features:
- Per-user Chroma vectorstore (collection name sanitized from user id)
- Embeddings: OpenAI text-embedding-3-small
- LLM: gpt-4.1-mini
- Chunking with overlap
- Similarity check using re-embedding of top-k retrieved chunks (cosine similarity)
- Optional Bing Web Search fallback when local similarity is low (BING_SEARCH_API_KEY)
- Persistent per-user chat history (JSON)
- Reset Memory button
- Robust handling for different langchain parameter names
"""

import os
import re
import io
import json
import requests
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from typing import List, Tuple

# LangChain / OpenAI classes (the wrapper you have in repo)
# These imports reflect the style used earlier in your file.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# ------------------------
# Config + env
# ------------------------
load_dotenv()
# prefer Streamlit secrets in cloud; else fallback to .env / environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY")  # optional

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
SIMILARITY_THRESHOLD = 0.18  # adjust per your data (0.18 is a starting point)
WEB_RESULTS_TO_USE = 3

# ------------------------
# Utilities (text extraction)
# ------------------------
def safe_collection_name(user_id: str) -> str:
    if not user_id:
        user_id = "guest"
    return "user_" + re.sub(r"[^a-zA-Z0-9]", "_", user_id).lower()

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
        texts = []
        for tag in soup.find_all(["h1","h2","h3","p","li"]):
            t = tag.get_text(separator=" ", strip=True)
            if t:
                texts.append(t)
        return "\n\n".join(texts) if texts else soup.get_text(separator="\n")
    except Exception:
        try:
            return b.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def fetch_url_text(url: str, timeout: int = 20) -> Tuple[str, str]:
    """Fetch a URL and return (text, final_url). Handles PDF/CSV/HTML heuristics."""
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text or "")
    return [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks if chunk.strip()]

# ------------------------
# Similarity util
# ------------------------
def cosine_similarity(a: List[float], b: List[float]) -> float:
    import numpy as np
    a = np.array(a, dtype="float32")
    b = np.array(b, dtype="float32")
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ------------------------
# Bing web-search fallback helpers
# ------------------------
def bing_web_search(query: str, top_n: int = 3) -> List[Tuple[str,str,str]]:
    key = BING_SEARCH_API_KEY
    if not key:
        return []
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": key}
    params = {"q": query, "count": top_n, "textDecorations": False, "textFormat": "Raw"}
    try:
        r = requests.get(endpoint, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = []
        for it in data.get("webPages", {}).get("value", [])[:top_n]:
            items.append((it.get("name"), it.get("snippet"), it.get("url")))
        return items
    except Exception:
        return []

def fetch_and_build_web_docs(search_results: List[Tuple[str,str,str]], limit_chars=15000) -> List[Document]:
    docs = []
    for title, snippet, url in search_results:
        try:
            text, final_url = fetch_url_text(url)
            txt = (text or "")[:limit_chars]
            docs.extend(build_docs_from_text_and_source(txt, source=final_url or url))
        except Exception:
            if snippet:
                docs.extend(build_docs_from_text_and_source(snippet, source=url))
    return docs

# ------------------------
# Vector store & history helpers
# ------------------------
def get_embeddings_client():
    # support both param names
    try:
        emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    except TypeError:
        emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    return emb

def load_vectorstore_for_user(user_id: str, embeddings: OpenAIEmbeddings) -> Chroma:
    collection_name = safe_collection_name(user_id)
    vs = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=PERSIST_DIR)
    return vs

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
    collection_name = safe_collection_name(user_id)
    try:
        # init a client and delete collection if supported
        vs = Chroma(collection_name=collection_name, embedding_function=get_embeddings_client(), persist_directory=PERSIST_DIR)
        if hasattr(vs, "delete_collection"):
            try:
                vs.delete_collection()
            except Exception:
                pass
    except Exception:
        pass
    hist_path = os.path.join(CHAT_HISTORY_DIR, f"{collection_name}.json")
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
st.set_page_config(page_title="RAG Chatbot (Chroma) — similarity + fallback", layout="wide")
st.title("📚 RAG Chatbot — text-embedding-3-small + gpt-4.1-mini + similarity check")

# Sidebar controls
with st.sidebar:
    st.header("User & Controls")
    user_id = st.text_input("Enter your email or user id:", value="guest")
    if not user_id:
        st.warning("Please enter your email/user-id to continue.")
        st.stop()

    st.markdown("---")
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
            if uploaded_files:
                for f in uploaded_files:
                    try:
                        text = extract_text_from_uploaded(f)
                        docs = build_docs_from_text_and_source(text, source=f.name)
                        if docs:
                            vs.add_documents(docs)
                            total_added += len(docs)
                    except Exception as e:
                        st.warning(f"Failed to ingest {f.name}: {e}")

            # links (single page fetch)
            if links_text:
                for line in links_text.splitlines():
                    url = line.strip()
                    if not url:
                        continue
                    try:
                        text, final_url = fetch_url_text(url)
                        docs = build_docs_from_text_and_source(text, source=final_url or url)
                        if docs:
                            vs.add_documents(docs)
                            total_added += len(docs)
                    except Exception as e:
                        st.warning(f"Failed to ingest {url}: {e}")

            persist_vectorstore(vs)
            st.success(f"Ingestion finished — added {total_added} chunks.")

    st.markdown("---")
    if st.button("🗑️ Reset Memory for this user"):
        delete_user_memory(user_id)
        st.success("Memory cleared for this user.")

    st.markdown("---")
    st.caption("Notes:\n - Embeddings use text-embedding-3-small (OpenAI).\n - Responses use gpt-4.1-mini.\n - If local similarity is low and you configured BING_SEARCH_API_KEY, top web results will be fetched.")

# Initialize embeddings, vectorstore, LLM & chain
embeddings = get_embeddings_client()
vectorstore = load_vectorstore_for_user(user_id, embeddings)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history(user_id)  # list(dict) with keys user, bot, sources

# LLM init (support both arg names)
try:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4.1-mini", temperature=0)
except TypeError:
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4.1-mini", temperature=0)

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)

# Show prior conversation
st.subheader(f"Chat — {user_id}")
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(msg["user"])
    with st.chat_message("assistant"):
        st.write(msg["bot"])
        if msg.get("sources"):
            with st.expander("Sources (saved)"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

# Chat input (persistent)
prompt = st.chat_input("Ask a question about your ingested documents or links...")
if prompt:
    # exact-match cache to avoid duplicate LLM calls
    cached = None
    for h in st.session_state.chat_history:
        if h["user"].strip().lower() == prompt.strip().lower():
            cached = h
            break

    if cached:
        answer = cached["bot"]
        sources = cached.get("sources", [])
    else:
        # always include chat_history for the chain
        hist_for_chain = [(h["user"], h["bot"]) for h in st.session_state.chat_history]

        # 1) retrieve local top-K candidate chunks
        try:
            docs = vectorstore.similarity_search(prompt, k=TOP_K)
        except Exception:
            # fallback if method signature differs
            docs = retriever.get_relevant_documents(prompt) if hasattr(retriever, "get_relevant_documents") else []

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
        except Exception as e:
            # if embeddings call fails, fallback to running the chain directly (but ensure chat_history is present)
            max_sim = 0.0
            sims = []

        answer = ""
        sources = []

        # 3) Decide: if similarity insufficient -> try web fallback or chain with disclaimer
        if max_sim < SIMILARITY_THRESHOLD:
            # try web fallback (if Bing key present)
            if BING_SEARCH_API_KEY:
                # fetch search results
                web_results = bing_web_search(prompt, top_n=WEB_RESULTS_TO_USE)
                web_docs = []
                if web_results:
                    web_docs = fetch_and_build_web_docs(web_results, limit_chars=15000)

                # build compact context from local docs + web docs (first passages)
                context_parts = []
                # include local docs with score
                for d, s in zip(docs, sims):
                    snippet = (d.page_content or "")[:1200]
                    source_label = d.metadata.get("source", "local")
                    context_parts.append(f"[Local:{source_label}] (sim={s:.3f})\n{snippet}")
                # include web doc snippets
                for wd in web_docs[:TOP_K]:
                    snippet = (wd.page_content or "")[:1200]
                    context_parts.append(f"[Web:{wd.metadata.get('source','web')}]\n{snippet}")

                context_text = "\n\n---\n\n".join(context_parts) if context_parts else ""
                system_prompt = (
                    "You are an assistant that answers the user's question using the provided context as primary source. "
                    "If the context does not contain the answer, be honest and provide best-effort guidance and label it as model knowledge. "
                    "Cite sources in square brackets after sentences, using the source URLs or file names from the context.\n\n"
                    f"Context:\n{context_text}\n\n"
                )
                # Call the LLM directly with system + user (use robust extraction)
                try:
                    resp = llm.generate([{"role":"system","content":system_prompt},{"role":"user","content":prompt}])
                    # langchain ChatOpenAI.generate may return object with .generations
                    try:
                        answer = resp.generations[0][0].text
                    except Exception:
                        try:
                            answer = resp.generations[0].text
                        except Exception:
                            answer = str(resp)
                except Exception:
                    # fallback to chain (still pass chat_history)
                    try:
                        res = qa_chain({"question": prompt, "chat_history": hist_for_chain})
                        answer = res.get("answer") or res.get("result") or res.get("output_text") or ""
                        sdocs = res.get("source_documents") or []
                        sources = [getattr(sd, "metadata", {}).get("source", "Unknown") for sd in sdocs]
                    except Exception as ee:
                        answer = f"Error during retrieval/LLM call: {ee}"
                        sources = []
                # set sources combining local & web
                sources = []
                for d in docs:
                    sources.append(d.metadata.get("source", "local"))
                for wd in web_docs:
                    sources.append(wd.metadata.get("source", "web"))
            else:
                # no web API key: run the retrieval chain (still pass chat_history)
                try:
                    res = qa_chain({"question": prompt, "chat_history": hist_for_chain})
                    answer = res.get("answer") or res.get("result") or res.get("output_text") or ""
                    sdocs = res.get("source_documents") or []
                    sources = [getattr(sd, "metadata", {}).get("source", "Unknown") for sd in sdocs]
                    if not sources:
                        answer = answer + "\n\n⚠️ Note: No sufficiently similar local documents were found. This answer is based on the model's training data (not live web)."
                except Exception as e:
                    answer = f"Error during retrieval/LLM call: {e}"
                    sources = []
        else:
            # similarity high enough → use normal RAG chain (always pass chat_history)
            try:
                res = qa_chain({"question": prompt, "chat_history": hist_for_chain})
                answer = res.get("answer") or res.get("result") or res.get("output_text") or ""
                sdocs = res.get("source_documents") or []
                sources = [getattr(sd, "metadata", {}).get("source", "Unknown") for sd in sdocs]
            except Exception as e:
                answer = f"Error during retrieval/LLM call: {e}"
                sources = []

    # Save history & display
    entry = {"user": prompt, "bot": answer, "sources": sources}
    st.session_state.chat_history.append(entry)
    save_chat_history(user_id, st.session_state.chat_history)

    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- {s}")
