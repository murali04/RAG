# streamlit_rag_chatbot_chroma.py
"""
Streamlit RAG Chatbot with ChromaDB
- File + link ingestion
- Deep scraping supported
- Persistent memory per user
- Cosine similarity relevance check
- Fallback to model knowledge when context irrelevant
- Streaming response display
- Clean Markdown formatting
- Role-based access: User / Admin (hardcoded password "12345")
- Welcome message for users
"""

import os, io, re, json, requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import numpy as np

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# =========================
# CONFIG
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

PERSIST_DIR = "chroma_store"
CHAT_HISTORY_DIR = "chat_histories"
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
SIMILARITY_THRESHOLD = 0.25

# Hardcoded admin password
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "12345"

# =========================
# HELPERS
# =========================
def safe_collection_name(user_id: str) -> str:
    return "user_" + re.sub(r"[^a-zA-Z0-9]", "_", (user_id or "guest")).lower()

def pdf_bytes_to_text(b: bytes) -> str:
    reader = PdfReader(io.BytesIO(b))
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def csv_bytes_to_text(b: bytes) -> str:
    df = pd.read_csv(io.BytesIO(b))
    return df.to_csv(index=False)

def html_bytes_to_text(b: bytes) -> str:
    soup = BeautifulSoup(b, "html.parser")
    for tag in soup(["script", "style", "noscript"]): 
        tag.decompose()
    texts = [t.get_text(" ", strip=True) for t in soup.find_all(["h1","h2","h3","p","li"])]
    return "\n\n".join([t for t in texts if t]) or soup.get_text("\n")

def fetch_url_text(url: str):
    r = requests.get(url, headers={"User-Agent":"RAG-Chatbot/1.0"}, timeout=20)
    r.raise_for_status()
    ctype = r.headers.get("content-type","").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"): 
        return (pdf_bytes_to_text(r.content), r.url)
    if "csv" in ctype or url.lower().endswith(".csv"): 
        return (csv_bytes_to_text(r.content), r.url)
    return (html_bytes_to_text(r.content), r.url)

def extract_text_from_uploaded(f) -> str:
    raw = f.read()
    name = f.name.lower()
    if name.endswith(".pdf"): return pdf_bytes_to_text(raw)
    if name.endswith(".csv"): return csv_bytes_to_text(raw)
    return raw.decode("utf-8", errors="ignore")

def build_docs(text: str, source: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text or "")
    return [Document(page_content=c, metadata={"source": source or "Unknown Source"}) 
            for c in chunks if c.strip()]

def cosine_similarity(a, b):
    if not isinstance(a, np.ndarray): a = np.array(a)
    if not isinstance(b, np.ndarray): b = np.array(b)
    if not a.any() or not b.any(): return 0.0
    return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def format_sources(docs, used_fallback=False):
    if used_fallback or not docs:
        return ["Response From Model Knowledge (no relevant context found in the shared data)"]
    sources = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        source = meta.get("source") or meta.get("url") or "Unknown Source"
        sources.append(str(source))
    seen = set(); ordered = []
    for s in sources:
        if s not in seen:
            seen.add(s); ordered.append(s)
    return ordered

# =========================
# VECTORSTORE & HISTORY
# =========================
def get_embeddings():
    try:
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    except TypeError:
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

def load_vectorstore(user_id: str):
    return Chroma(
        collection_name=safe_collection_name(user_id),
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )

def save_history(user_id, history):
    path = os.path.join(CHAT_HISTORY_DIR, f"{safe_collection_name(user_id)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history(user_id):
    path = os.path.join(CHAT_HISTORY_DIR, f"{safe_collection_name(user_id)}.json")
    if os.path.exists(path):
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            return []
    return []

def delete_memory(user_id):
    try:
        Chroma(
            collection_name=safe_collection_name(user_id),
            embedding_function=get_embeddings(),
            persist_directory=PERSIST_DIR,
        ).delete_collection()
    except Exception:
        pass
    path = os.path.join(CHAT_HISTORY_DIR, f"{safe_collection_name(user_id)}.json")
    if os.path.exists(path):
        os.remove(path)
    if st.session_state.get("active_user") == user_id:
        st.session_state.chat_history = []
        st.session_state.active_user = None

def list_all_users():
    users = []
    for f in os.listdir(CHAT_HISTORY_DIR):
        if f.endswith(".json"):
            users.append(f.replace(".json", "").replace("user_", ""))
    return sorted(users)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 Learn with Elumalai and Murali RAG - Keep Building Yourself")

role = st.sidebar.selectbox("Login as:", ["user", "admin"])

if "role" not in st.session_state or st.session_state.get("role") != role:
    st.session_state["role"] = role
    st.session_state["admin_authenticated"] = False

# ---------- ADMIN ----------
if role == "admin":
    st.subheader("👨‍💼 Admin Login")
    admin_user = st.text_input("Admin username", value=ADMIN_USERNAME)
    admin_pass = st.text_input("Admin password", type="password")

    if st.button("Login as Admin"):
        if admin_user == ADMIN_USERNAME and admin_pass == ADMIN_PASSWORD:
            st.session_state["admin_authenticated"] = True
            st.success("Admin authenticated ✅")
        else:
            st.session_state["admin_authenticated"] = False
            st.error("Invalid admin credentials ❌")

    if st.session_state.get("admin_authenticated"):
        st.info("You are signed in as Admin.")
        users = list_all_users()
        if not users:
            st.info("No users found yet.")
        else:
            selected_user = st.selectbox("Select user", users)
            if st.button("❌ Delete user data"):
                delete_memory(selected_user)
                st.success(f"Deleted memory for {selected_user}")
                st.experimental_rerun()

# ---------- USER ----------
else:
    with st.sidebar:
        st.header("Controls")
        user_id = st.text_input("Enter your email / user id:", value="guest")
        files = st.file_uploader("Upload files", accept_multiple_files=True)
        links = st.text_area("Paste links (one per line)")

        if st.button("Ingest"):
            vs = load_vectorstore(user_id)
            added = 0
            for f in files or []:
                docs = build_docs(extract_text_from_uploaded(f), f.name)
                vs.add_documents(docs); added += len(docs)
            for url in (links or "").splitlines():
                if not url.strip(): continue
                text, final_url = fetch_url_text(url.strip())
                docs = build_docs(text, final_url)
                vs.add_documents(docs); added += len(docs)
            vs.persist(); st.success(f"Ingested {added} chunks.")

        if st.button("🗑️ Reset Memory"): 
            delete_memory(user_id); st.success("Memory cleared.")

    if "active_user" not in st.session_state or st.session_state.get("active_user") != user_id:
        st.session_state["active_user"] = user_id
        st.session_state["chat_history"] = load_history(user_id)

    st.subheader(f"Hello, {user_id} 👋")
    st.markdown("Ask questions about your uploaded files or links. I'll answer using your documents first, otherwise I'll use my model knowledge.")

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0, streaming=True)
    vectorstore = load_vectorstore(user_id)

    for msg in st.session_state.get("chat_history", []):
        with st.chat_message("user"): st.write(msg["user"])
        with st.chat_message("assistant"):
            st.markdown(msg["bot"])
            if msg.get("sources"): 
                with st.expander("Sources"): 
                    for s in msg["sources"]: st.write(f"- {s}")

    prompt = st.chat_input("Ask something...")
    if prompt:
        embeddings = get_embeddings()
        try: 
            docs = vectorstore.similarity_search(prompt, k=TOP_K)
        except Exception:
            docs = []

        try:
            q_emb = embeddings.embed_query(prompt)
            sims = [cosine_similarity(q_emb, embeddings.embed_query(d.page_content)) for d in docs]
            max_sim = max(sims) if sims else 0.0
        except Exception:
            max_sim = 0.0

        if max_sim < SIMILARITY_THRESHOLD:
            system = "Answer from your own knowledge in clean format."
            stream = llm.stream([{"role":"system","content":system},
                                 {"role":"user","content":prompt}])
            sources = format_sources(docs, used_fallback=True)
        else:
            context = "\n\n".join([d.page_content for d in docs])
            system = "You are a RAG assistant. Use the provided context to answer clearly."
            stream = llm.stream([{"role":"system","content":system},
                                 {"role":"user","content":f"Context:\n{context}\n\nQuestion:\n{prompt}"}])
            sources = format_sources(docs, used_fallback=False)

        with st.chat_message("user"): st.write(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty(); text_out = ""
            for chunk in stream:
                token = getattr(chunk, "content", "")
                text_out += token
                placeholder.markdown(text_out + "▌")
            placeholder.markdown(text_out)

            with st.expander("Sources"): 
                for s in sources: st.write(f"- {s}")

        st.session_state["chat_history"].append({"user":prompt,"bot":text_out,"sources":sources})
        save_history(user_id, st.session_state["chat_history"])
