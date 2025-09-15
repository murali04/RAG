# streamlit_rag_chatbot_chroma.py
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

# LangChain imports (common API)
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# ------------------------
# Config & paths
# ------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env. Put OPENAI_API_KEY=sk-... in your .env file.")

PERSIST_DIR = "chroma_store"
CHAT_HISTORY_DIR = "chat_histories"
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# App settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4  # number of docs to retrieve for each query

# ------------------------
# Helper utilities
# ------------------------
def safe_collection_name(user_id: str) -> str:
    """Sanitize user id (email) into a safe collection name for Chroma."""
    if not user_id:
        user_id = "guest"
    safe_id = re.sub(r"[^a-zA-Z0-9]", "_", user_id)
    return f"user_{safe_id.lower()}"

def pdf_bytes_to_text(b: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(b))
        texts = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(texts)
    except Exception:
        # fallback decode if extraction fails
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
        for tag in soup(["script", "style", "noscript"]):
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

def fetch_url_text(url: str, timeout: int = 20):
    """Fetch URL and return (text, final_url). Handles PDFs & HTML & CSV."""
    try:
        headers = {"User-Agent": "RAG-Chatbot/1.0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        return f"Failed to fetch URL: {e}", url
    ctype = r.headers.get("content-type", "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return pdf_bytes_to_text(r.content), r.url
    if "csv" in ctype or url.lower().endswith(".csv"):
        return csv_bytes_to_text(r.content), r.url
    # assume html/text
    return html_bytes_to_text(r.content), r.url

def extract_text_from_uploaded(uploaded_file) -> str:
    """Return text for an uploaded Streamlit file (bytes). Supports PDF, CSV, TXT."""
    raw = uploaded_file.read()
    # detect extension
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return pdf_bytes_to_text(raw)
    if name.endswith(".csv"):
        return csv_bytes_to_text(raw)
    # default / txt / md
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def build_docs_from_text_and_source(text: str, source: str):
    """Split `text` into chunks and return list[Document] with metadata {'source': source}"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text or "")
    docs = [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks if chunk.strip()]
    return docs

# ------------------------
# Vectorstore + history helpers
# ------------------------
def load_vectorstore_for_user(user_id: str, embeddings: OpenAIEmbeddings):
    collection_name = safe_collection_name(user_id)
    # instantiate (loads persisted data if present)
    vs = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=PERSIST_DIR)
    return vs

def persist_vectorstore(vs: Chroma):
    try:
        vs.persist()
    except Exception:
        # some versions may not require or have persist; ignore errors
        pass

def save_chat_history(user_id: str, history):
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{safe_collection_name(user_id)}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_chat_history(user_id: str):
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{safe_collection_name(user_id)}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def delete_user_memory(user_id: str):
    """Delete user's Chroma collection and chat history file."""
    collection_name = safe_collection_name(user_id)
    try:
        vs = Chroma(collection_name=collection_name, embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY), persist_directory=PERSIST_DIR)
        # delete_collection may be supported depending on langchain/chroma versions
        if hasattr(vs, "delete_collection"):
            try:
                vs.delete_collection()
            except Exception:
                pass
    except Exception:
        pass

    # Remove chat history file
    hist_path = os.path.join(CHAT_HISTORY_DIR, f"{collection_name}.json")
    if os.path.exists(hist_path):
        try:
            os.remove(hist_path)
        except Exception:
            pass

    # Also try to remove persistent files for that collection directory if present (best-effort)
    # NOTE: chroma stores everything under persist_directory; removing specific files depends on chroma structure.
    if "chat_history" in st.session_state:
        st.session_state.chat_history = []

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="RAG Chatbot - Multi-user", layout="wide")
st.title("📚 Learn with Elumalai and Murali Multi-user RAG Chatbot - Keep Building Yourself")

# Sidebar: user, ingestion, controls
with st.sidebar:
    st.header("User & Controls")
    user_id = st.text_input("Enter your email or user id (used to store personal memory):", value="guest")
    if not user_id:
        st.warning("Please enter your email/user-id to continue.")
        st.stop()

    st.markdown("---")
    st.subheader("Ingest documents")
    uploaded_files = st.file_uploader("Upload files (PDF / CSV / TXT / MD)", accept_multiple_files=True)

    st.markdown("Or paste links (one per line)")
    links_text = st.text_area("Paste https links here (one per line)")

    if st.button("Ingest selected files & links"):
        # initialize embeddings & vectorstore
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) if "openai_api_key" not in OpenAIEmbeddings.__dict__ else OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        # Note: langchain versions differ for parameter name; attempt both safe ways:
        try:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        except Exception:
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            except Exception:
                embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

        vs = load_vectorstore_for_user(user_id, embeddings)

        total_added = 0
        # process uploaded files
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

        # process links
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
        st.success(f"Ingestion finished — added {total_added} chunks to your personal store.")

    st.markdown("---")
    if st.button("🗑️ Reset Memory for this user"):
        delete_user_memory(user_id)
        st.success(f"Memory reset for {user_id}")

    st.markdown("---")
    st.caption("Notes:\n• Ingesting files creates embeddings (one-time cost).\n• Each question uses the LLM (cost per query).")

# Initialize embeddings, vectorstore, and LLM for runtime
try:
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
except Exception:
    # Some versions use openai_api_key param name
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vectorstore = load_vectorstore_for_user(user_id, embeddings)

# Chat history load
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history(user_id)  # list of dicts: {"user":..., "bot":..., "sources":[...]}

# Create the ConversationalRetrievalChain
try:
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)  # adapt model as available
except TypeError:
    # some versions use openai_api_key kwarg
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)

# Display prior conversation
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
    # Simple exact-question cache: if same exact question exists, reuse answer (avoid LLM call)
    found_cached = None
    for h in st.session_state.chat_history:
        if h["user"].strip().lower() == prompt.strip().lower():
            found_cached = h
            break

    if found_cached:
        answer = found_cached["bot"]
        sources = found_cached.get("sources", [])
    else:
        # prepare chat_history tuple list for chain
        hist_for_chain = [(h["user"], h["bot"]) for h in st.session_state.chat_history]
        try:
            result = qa_chain({"question": prompt, "chat_history": hist_for_chain})
        except Exception as e:
            # fallback: try calling with single arg
            result = qa_chain({"question": prompt})
        # extract answer & sources with safe fallbacks
        answer = result.get("answer") or result.get("result") or result.get("output_text") or ""
        source_docs = result.get("source_documents") or result.get("source_documents", [])
        sources = []
        if source_docs:
            for d in source_docs:
                md = getattr(d, "metadata", {}) or {}
                s = md.get("source") or md.get("source_link") or md.get("filename") or "Unknown"
                sources.append(s)

    # Append to session history and persist
    history_entry = {"user": prompt, "bot": answer, "sources": sources}
    st.session_state.chat_history.append(history_entry)
    save_chat_history(user_id, st.session_state.chat_history)

    # Display the message we just added (Streamlit will also render above once saved)
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- {s}")