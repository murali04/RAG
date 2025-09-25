# agentic_training_portal.py
"""
Streamlit app: Training & Assessment Portal (Agentic AI prototype, v2)

Features:
- Register/login (prototype, in-memory)
- Upskill or Knowledge Assessment flows
- User progress persistence (resume training plans)
- User optional doc ingestion (PDF/Word/Text)
- RAG pipeline with Chroma + text-embedding-3-small
- LLM: gpt-4o-mini
- Chunking + cosine similarity retrieval
- Evaluation dashboard to track progress

Requirements:
- Python 3.9+
- pip install streamlit langchain openai chromadb tiktoken python-dotenv PyPDF2 python-docx

Run:
  streamlit run agentic_training_portal.py

Note: Replace prototype auth and persistence with production-ready DB for real deployment.
"""

import streamlit as st
import os
import json
import tempfile
from typing import List
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ---------------------- Config ----------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PERSIST_DIR = "./chroma_data"

# ---------------------- Helpers ----------------------
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY)

@st.cache_resource
def get_vectorstore(embeddings):
    return Chroma(persist_directory=CHROMA_PERSIST_DIR, collection_name="training_docs", embedding_function=embeddings)


def parse_uploaded_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    text = ""
    if ext == ".pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif ext in [".docx"]:
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    return text


def ingest_texts_to_chroma(texts: List[str], embeddings, collection_name="training_docs"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    docs = [{"page_content": c, "metadata": {}} for c in chunks]
    vect = Chroma(persist_directory=CHROMA_PERSIST_DIR, collection_name=collection_name, embedding_function=embeddings)
    vect.add_texts([d["page_content"] for d in docs])
    vect.persist()
    return vect


# ---------------------- LLM-driven functions ----------------------
def generate_assessment_questions(llm, skill: str, experience: str, n_questions: int = 5) -> List[str]:
    prompt = PromptTemplate(
        input_variables=["skill", "experience", "n"],
        template=("You are an expert instructor. Create {n} assessment questions for the skill '{skill}' "
                  "tailored to a learner with experience: {experience}. Return a JSON list of questions only.")
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run({"skill": skill, "experience": experience, "n": n_questions})
    try:
        return json.loads(resp)
    except Exception:
        lines = [l.strip() for l in resp.splitlines() if l.strip()]
        return lines[:n_questions]


def grade_answers(llm, questions: List[str], answers: List[str]) -> dict:
    q_and_a_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)])
    prompt = PromptTemplate(
        input_variables=["q_and_a"],
        template=("You are an objective grader. Given a list of question-answer pairs, score each answer between 0 and 10 "
                  "and provide a short justification. Return JSON with keys 'scores' and 'feedback'.\n\n{q_and_a}")
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run({"q_and_a": q_and_a_text})
    try:
        return json.loads(resp)
    except Exception:
        return {"scores": [0]*len(questions), "feedback": ["Could not parse"]*len(questions)}


def create_personalized_plan(llm, skill: str, experience: str, grades: dict, sources: str = "") -> str:
    prompt = PromptTemplate(
        input_variables=["skill", "experience", "grades", "sources"],
        template=("You are a learning designer. Given the skill: {skill}, user experience: {experience}, and grades: {grades}, "
                  "produce a 6-week personalized learning plan with weekly goals and practice tasks. {sources}")
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"skill": skill, "experience": experience, "grades": grades, "sources": sources})

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Agentic Training Portal", layout="wide")
st.title("Agentic Training Portal — Training & Assessment Agents")

embeddings = get_embeddings()
llm = get_llm()
vectorstore = get_vectorstore(embeddings)

# User auth (prototype)
if "users" not in st.session_state:
    st.session_state["users"] = {}
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None

with st.sidebar.expander("Account"):
    mode = st.selectbox("Mode", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if mode == "Register":
        if st.button("Create account"):
            st.session_state["users"][username] = {"password": password, "progress": {}}
            st.success("Account created — please login")
    else:
        if st.button("Login"):
            ustore = st.session_state["users"]
            if username in ustore and ustore[username]["password"] == password:
                st.session_state["current_user"] = username
                st.success(f"Logged in as {username}")
            else:
                st.error("Invalid credentials")

if not st.session_state["current_user"]:
    st.stop()

user = st.session_state["current_user"]
progress = st.session_state["users"][user].setdefault("progress", {})
st.write(f"Welcome, **{user}**")

# Main choice
task = st.selectbox("Choose an action", ["Upskill (training)", "Check my knowledge (assessment)", "Dashboard"])

# Upload user docs
with st.expander("Upload your own docs (optional)"):
    uploads = st.file_uploader("Upload PDF, Word, or Text", accept_multiple_files=True)
    if uploads and st.button("Ingest docs"):
        texts = [parse_uploaded_file(f) for f in uploads]
        ingest_texts_to_chroma(texts, embeddings, collection_name=f"user_{user}_docs")
        st.success("Docs ingested")

# Collect info
skill = st.text_input("Skill (e.g., Python, SQL, ML)")
experience = st.selectbox("Experience", ["No experience","<6 months","6-24 months","2-5 years",">5 years"])

if task == "Upskill (training)":
    if st.button("Start Upskill"):
        questions = generate_assessment_questions(llm, skill, experience, 5)
        answers = [st.text_area(f"Q{i+1}: {q}", key=f"up_ans_{i}") for i,q in enumerate(questions)]
        if st.button("Submit baseline answers"):
            grading = grade_answers(llm, questions, answers)
            docs = vectorstore.similarity_search(skill, k=3)
            snippet = "\n\n".join([d.page_content for d in docs])
            plan = create_personalized_plan(llm, skill, experience, grading, sources=snippet)
            progress["plan"] = plan
            st.write("### Personalized Plan")
            st.write(plan)

elif task == "Check my knowledge (assessment)":
    n_q = st.slider("Number of questions", 3, 10, 5)
    if st.button("Start assessment"):
        questions = generate_assessment_questions(llm, skill, experience, n_q)
        answers = [st.text_area(f"Q{i+1}: {q}", key=f"assess_ans_{i}") for i,q in enumerate(questions)]
        if st.button("Submit answers"):
            grading = grade_answers(llm, questions, answers)
            avg = sum(grading.get("scores", []))/max(1, len(grading.get("scores", [])))
            progress.setdefault("assessments", []).append({"skill": skill, "scores": grading["scores"], "avg": avg})
            st.write(grading)
            st.metric("Average score", f"{avg:.2f}")

elif task == "Dashboard":
    st.subheader("Your Progress Dashboard")
    if "plan" in progress:
        st.write("### Current Plan")
        st.write(progress["plan"])
    if "assessments" in progress:
        st.write("### Past Assessments")
        for a in progress["assessments"]:
            st.json(a)

st.markdown("---")
st.caption("Prototype: API key from .env, gpt-4o-mini + text-embedding-3-small, progress persistence, PDF/Word ingestion.")
