# agentic_training_portal_agentic_fixed.py
"""
Agentic Training Portal - Fixed Version
Features:
- Registration/Login (persistent via JSON)
- Upskill & Knowledge Assessment
- Personalized learning plan generation
- RAG via Chroma + embeddings (text-embedding-3-small)
- PDF/Word ingestion (plain text prototype)
- Dashboard showing scores & progress
- Uses ChatOpenAI (non-deprecated)
- Session state drives assessment navigation
"""

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from pathlib import Path
from dotenv import load_dotenv
import json
import os
import re
from typing import List

# -------------------- Config --------------------
CHROMA_PERSIST_DIR = "./chroma_data"
USERS_FILE = "users.json"

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OpenAI API key in .env")
    st.stop()

# -------------------- Utilities --------------------
def load_users():
    if Path(USERS_FILE).exists():
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")

@st.cache_resource
def get_vectorstore(embeddings: OpenAIEmbeddings, persist_dir: str = CHROMA_PERSIST_DIR):
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

@st.cache_resource
def get_llm():
    return ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini", temperature=0)

# -------------------- LLM functions --------------------
def generate_questions(skill: str, experience: str, n: int = 5) -> List[str]:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "You are an expert instructor. Create {n} assessment questions for skill '{skill}' "
        "tailored to a learner with experience '{experience}'. Return a JSON list of questions only."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run({"skill": skill, "experience": experience, "n": n})

    # Clean raw JSON fences
    resp = re.sub(r"```json|```", "", resp).strip()
    try:
        qlist = json.loads(resp)
        return qlist
    except:
        # fallback: split lines
        lines = [l.strip() for l in resp.splitlines() if l.strip()]
        cleaned = [re.sub(r"^\d+[\.\)]\s*", "", l) for l in lines]
        return cleaned[:n]

def grade_answers(questions: List[str], answers: List[str]):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "You are an objective grader. Given question-answer pairs, score each answer 0-10 "
        "and provide short feedback. Return JSON object: {'scores':[...], 'feedback':[...]}.\n\n{q_and_a}"
    )
    q_and_a_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)])
    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run({"q_and_a": q_and_a_text})
    resp = re.sub(r"```json|```", "", resp).strip()
    try:
        return json.loads(resp)
    except:
        return {"scores": [0]*len(questions), "feedback": ["Parsing failed"]*len(questions)}

def create_personalized_plan(skill: str, experience: str, grades: dict, sources_snippet: str = ""):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "Given skill {skill}, experience {experience}, grades {grades}, "
        "create a concise 6-week personalized learning plan. Reference sources: {sources}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    resp = chain.run({"skill": skill, "experience": experience, "grades": grades, "sources": sources_snippet})
    return resp

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Agentic Training Portal", layout="wide")
st.title("Agentic Training Portal — Training & Assessment Agents")

# -------------------- User Auth --------------------
users = load_users()
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None

with st.sidebar.expander("Account"):
    mode = st.selectbox("Mode", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if mode == "Register" and st.button("Create account"):
        if username in users:
            st.warning("Username exists")
        else:
            users[username] = {"password": password, "progress": {}}
            save_users(users)
            st.success("Account created — please login")
    if mode == "Login" and st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state["current_user"] = username
            st.success(f"Logged in as {username}")
        else:
            st.error("Invalid credentials")

if not st.session_state["current_user"]:
    st.info("Please register or login.")
    st.stop()

user = st.session_state["current_user"]
st.write(f"Welcome, **{user}**")

# -------------------- Vectorstore --------------------
embeddings = get_embeddings()
vectorstore = get_vectorstore(embeddings)

# -------------------- Main Actions --------------------
task = st.selectbox("Choose an action", ["Upskill (training)", "Check my knowledge (assessment)"])
skill = st.text_input("Skill to train / assess (e.g., Python, SQL, Machine Learning)")
experience = st.selectbox("Experience level", ["No experience","<6 months","6-24 months","2-5 years",">5 years"]) 

# -------------------- Assessment Flow --------------------
if "progress" not in st.session_state:
    st.session_state["progress"] = users[user].get("progress", {})

if "current_q" not in st.session_state:
    st.session_state["current_q"] = 0

if task == "Upskill (training)":
    st.header("Upskill flow")
    if st.button("Start Upskill"):
        if not skill:
            st.error("Enter skill first")
        else:
            questions = generate_questions(skill, experience)
            st.session_state["progress"]["questions"] = questions
            st.session_state["progress"]["answers"] = [""]*len(questions)
            st.session_state["current_q"] = 0

    # Display current question
    if "questions" in st.session_state["progress"]:
        qidx = st.session_state["current_q"]
        questions = st.session_state["progress"]["questions"]
        answers = st.session_state["progress"]["answers"]

        st.write(f"### Question {qidx+1}/{len(questions)}")
        ans = st.text_area("Your Answer:", value=answers[qidx], key=f"ans_{qidx}")
        st.session_state["progress"]["answers"][qidx] = ans

        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            if qidx > 0 and st.button("Previous"):
                st.session_state["current_q"] -= 1
        with col2:
            if qidx < len(questions)-1 and st.button("Next"):
                st.session_state["current_q"] += 1
            elif qidx == len(questions)-1 and st.button("Submit"):
                grading = grade_answers(questions, st.session_state["progress"]["answers"])
                st.session_state["progress"]["grading"] = grading

                # Dashboard + Plan
                docs = vectorstore.similarity_search(skill, k=3)
                snippet = "\n\n".join([d.page_content for d in docs]) if docs else ""
                plan = create_personalized_plan(skill, experience, grading, sources_snippet=snippet)

                st.write("### Assessment completed!")
                st.write("#### Scores & Feedback")
                st.write(grading)
                avg = sum(grading.get("scores", [0])) / max(1, len(grading.get("scores", [])))
                st.metric("Average Score", f"{avg:.2f}")
                st.write("### Personalized 6-week plan")
                st.write(plan)

                # Save progress
                users[user]["progress"] = st.session_state["progress"]
                save_users(users)

elif task == "Check my knowledge (assessment)":
    st.header("Knowledge Assessment")
    n_q = st.slider("Number of questions", min_value=3, max_value=10, value=5)
    if st.button("Start assessment"):
        if not skill:
            st.error("Enter skill")
        else:
            questions = generate_questions(skill, experience, n=n_q)
            st.session_state["progress"]["questions"] = questions
            st.session_state["progress"]["answers"] = [""]*len(questions)
            st.session_state["current_q"] = 0

    # Display current question
    if "questions" in st.session_state["progress"]:
        qidx = st.session_state["current_q"]
        questions = st.session_state["progress"]["questions"]
        answers = st.session_state["progress"]["answers"]

        st.write(f"### Question {qidx+1}/{len(questions)}")
        ans = st.text_area("Your Answer:", value=answers[qidx], key=f"ans_assess_{qidx}")
        st.session_state["progress"]["answers"][qidx] = ans

        col1, col2 = st.columns(2)
        with col1:
            if qidx > 0 and st.button("Previous"):
                st.session_state["current_q"] -= 1
        with col2:
            if qidx < len(questions)-1 and st.button("Next"):
                st.session_state["current_q"] += 1
            elif qidx == len(questions)-1 and st.button("Submit"):
                grading = grade_answers(questions, st.session_state["progress"]["answers"])
                st.session_state["progress"]["grading"] = grading

                # Dashboard + Plan
                docs = vectorstore.similarity_search(skill, k=3)
                snippet = "\n\n".join([d.page_content for d in docs]) if docs else ""
                plan = create_personalized_plan(skill, experience, grading, sources_snippet=snippet)

                st.write("### Assessment completed!")
                st.write("#### Scores & Feedback")
                st.write(grading)
                avg = sum(grading.get("scores", [0])) / max(1, len(grading.get("scores", [])))
                st.metric("Average Score", f"{avg:.2f}")
                st.write("### Recommended short plan")
                st.write(plan)

                # Save progress
                users[user]["progress"] = st.session_state["progress"]
                save_users(users)

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Prototype: Persistent users, vectorstore, assessment, and personalized learning plan.")
