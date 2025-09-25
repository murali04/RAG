import streamlit as st
import json
import os
import datetime
import pandas as pd
import random
from io import BytesIO

# ---------------- File Paths ----------------
USER_FILE = "users.json"
QUESTION_FILE = "questions.txt"

# ---------------- Helpers ----------------
def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    try:
        with open(USER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # corrupted or empty file -> return empty dict (you might want to backup)
        return {}

def save_users(users):
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)

def load_questions():
    """
    Expecting each line in questions.txt like:
    Question text|Option A|Option B|Option C|Option D|AnswerLetter
    Example:
    What is 2+2?|1|2|3|4|D
    """
    questions = []
    abs_path = os.path.abspath(QUESTION_FILE)
    if not os.path.exists(QUESTION_FILE):
        st.error(f"❌ questions.txt not found at {abs_path}")
        return []

    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            parts = [p.strip() for p in raw.split("|")]
            if len(parts) >= 6:
                # allow extra '|' in question text by taking first 6 tokens only
                q, a, b, c, d, ans = parts[:6]
                questions.append({
                    "question": q,
                    "options": [f"A. {a}", f"B. {b}", f"C. {c}", f"D. {d}"],
                    "answer": ans.strip().upper()  # e.g. "A", "B", etc.
                })
            else:
                st.warning(f"⚠️ Skipping invalid line {line_no} in questions.txt: {raw}")
    return questions


# ---------------- User Functions ----------------
def register_user():
    st.subheader("📝 Register New User")
    username = st.text_input("Choose a Username", key="reg_name")
    password = st.text_input("Choose a Password", type="password", key="reg_pass")

    if st.button("Register"):
        users = load_users()
        if not username or not password:
            st.error("Please enter username and password.")
            return
        if username in users:
            st.error("❌ Username already exists")
        else:
            users[username] = {"password": password, "attendance": [], "results": []}
            save_users(users)
            st.success("✅ Registered successfully! Please login.")


def user_login():
    st.subheader("🔑 User Login")
    username = st.text_input("Username", key="user_login")
    password = st.text_input("Password", type="password", key="user_pass")

    if st.button("Login"):
        users = load_users()
        if username in users and users[username].get("password") == password:
            st.session_state["user"] = username
            st.session_state["role"] = "user"
            st.success(f"✅ Welcome {username}!")
            # rerun so UI updates to logged-in view
            try:
                st.rerun()
            except Exception:
                pass
        else:
            st.error("❌ Invalid username or password")


def mark_attendance(username):
    users = load_users()
    if username not in users:
        st.error("User not found.")
        return
    today = str(datetime.date.today())
    attendance = users[username].get("attendance", [])
    if today not in attendance:
        attendance.append(today)
        users[username]["attendance"] = attendance
        save_users(users)
        st.success("✅ Attendance marked for today!")
    else:
        st.info("ℹ️ Attendance already marked today.")

    st.subheader("📅 Your Attendance History")
    attendance = users[username].get("attendance", [])
    if attendance:
        df = pd.DataFrame({"Date": attendance})
        df.index = df.index + 1
        st.dataframe(df)
    else:
        st.info("No attendance records yet.")


def take_assessment(username):
    users = load_users()
    if username not in users:
        st.error("User not found.")
        return

    # Load all questions
    all_questions = load_questions()
    if not all_questions:
        st.error("❌ No questions found. Please add them in questions.txt")
        return

    # Only pick 10 random questions once per session
    if "assessment_questions" not in st.session_state:
        st.session_state["assessment_questions"] = random.sample(all_questions, min(10, len(all_questions)))

    questions = st.session_state["assessment_questions"]

    # Unique form key per user per attempt
    form_key = f"assessment_{username}"

    st.header("📝 Assessment")
    st.write(f"Total questions in this attempt: {len(questions)}")

    with st.form(form_key):
        for i, q in enumerate(questions, start=1):
            st.markdown(f"**Q{i}.** {q['question']}")
            st.radio("", q["options"], key=f"{form_key}_q{i}")

        submitted = st.form_submit_button("Submit Assessment")

    if submitted:
        score = 0
        results_data = []
        for i, q in enumerate(questions, start=1):
            selected = st.session_state.get(f"{form_key}_q{i}")
            selected_letter = selected.strip()[0].upper() if selected else "-"
            correct_letter = q["answer"]
            if selected_letter == correct_letter:
                score += 1
            # store question, selected, correct answer for display
            results_data.append({
                "Q.No": i,
                "Question": q["question"],
                "Your Answer": selected_letter,
                "Correct Answer": correct_letter
            })

        # Save result
        users = load_users()  # reload to avoid overwriting concurrent changes
        user_info = users.get(username, {})
        user_results = user_info.get("results", [])
        user_results.append({"score": score, "total": len(questions), "date": str(datetime.date.today())})
        user_info["results"] = user_results
        users[username] = user_info
        save_users(users)

        st.success(f"✅ Your Score: {score}/{len(questions)}")

        # Clear session_state to allow next assessment later
        del st.session_state["assessment_questions"]

        # Show correct answers alongside user's selections
        st.subheader("📌 Correct Answers")
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results)

        # Show attempt history
        st.subheader("📊 Your Previous Results")
        df_res = pd.DataFrame(user_info.get("results", []))
        if not df_res.empty:
            df_res.index = df_res.index + 1
            st.dataframe(df_res)
        else:
            st.info("No results yet.")



# ---------------- Admin Functions ----------------
def admin_login():
    st.subheader("👨‍💼 Admin Login")
    username = st.text_input("Admin Username", key="admin_user")
    password = st.text_input("Admin Password", type="password", key="admin_pass")

    if st.button("Login as Admin"):
        if username == "admin" and password == "123456789":
            st.session_state["role"] = "admin"
            st.session_state["user"] = username
            st.success("✅ Logged in as Admin")
            try:
                st.rerun()
            except Exception:
                pass
        else:
            st.error("❌ Invalid Admin credentials")


def admin_dashboard():
    st.title("📊 Admin Dashboard")
    users = load_users()

    if not users:
        st.info("No users registered yet.")
        return

    # Build attendance & results lists
    attendance_rows = []
    results_rows = []
    for user, data in users.items():
        attendance_list = data.get("attendance", [])
        for d in attendance_list:
            attendance_rows.append({"User": user, "Date": d})
        results_list = data.get("results", [])
        for r in results_list:
            results_rows.append({
                "User": user,
                "Score": r.get("score", 0),
                "Total": r.get("total", 0),
                "Date": r.get("date", "N/A")
            })

    st.subheader("🗓 Attendance Records")
    if attendance_rows:
        df_att = pd.DataFrame(attendance_rows)
        df_att.index = df_att.index + 1
        st.dataframe(df_att)
        # download attendance csv
        csv_att = df_att.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Attendance (CSV)", csv_att, "attendance.csv", "text/csv")
    else:
        st.info("No attendance records found.")

    st.subheader("📝 Assessment Results")
    if results_rows:
        df_res = pd.DataFrame(results_rows)
        df_res.index = df_res.index + 1
        st.dataframe(df_res)
        # download results csv
        csv_res = df_res.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results (CSV)", csv_res, "results.csv", "text/csv")

        # Leaderboard: highest score by user
        leaderboard = df_res.groupby("User")["Score"].max().reset_index().sort_values("Score", ascending=False)
        leaderboard.index = leaderboard.index + 1
        st.subheader("🏆 Leaderboard (Top Scores)")
        st.table(leaderboard)

        # Average scores chart
        avg_scores = df_res.groupby("User")["Score"].mean().reset_index()
        if not avg_scores.empty:
            st.subheader("📈 Average Score by User")
            st.bar_chart(data=avg_scores.set_index("User")["Score"])
    else:
        st.info("No assessment results available yet.")

    # Export full history (attendance + results) as a single CSV
    export_rows = []
    for user, data in users.items():
        for d in data.get("attendance", []):
            export_rows.append([user, "Attendance", d, "", ""])
        for r in data.get("results", []):
            export_rows.append([user, "Result", r.get("date", ""), r.get("score", ""), r.get("total", "")])
    if export_rows:
        df_export = pd.DataFrame(export_rows, columns=["Username", "Type", "Date", "Score", "Total"])
        csv_export = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export Full Data (CSV)", csv_export, "user_full_data.csv", "text/csv")


# ---------------- Main ----------------
def main():
    st.title("🎓 Training Portal")

    if "role" not in st.session_state:
        st.session_state["role"] = None
    if "user" not in st.session_state:
        st.session_state["user"] = None

    # USER VIEW
    if st.session_state["role"] == "user":
        username = st.session_state["user"]
        st.sidebar.title(f"Welcome, {username}")
        menu_choice = st.sidebar.radio("📌 Menu", ["Mark Attendance", "Take Assessment", "Logout"])

        if menu_choice == "Mark Attendance":
            mark_attendance(username)
        elif menu_choice == "Take Assessment":
            take_assessment(username)
        elif menu_choice == "Logout":
            st.session_state["role"] = None
            st.session_state["user"] = None
            st.success("✅ Logged out successfully")
            try:
                st.rerun()
            except Exception:
                pass

    # ADMIN VIEW
    elif st.session_state["role"] == "admin":
        st.sidebar.title("Admin Panel")
        menu_choice = st.sidebar.radio("📌 Menu", ["Dashboard", "Logout"])

        if menu_choice == "Dashboard":
            admin_dashboard()
        elif menu_choice == "Logout":
            st.session_state["role"] = None
            st.session_state["user"] = None
            st.success("✅ Admin logged out")
            try:
                st.rerun()
            except Exception:
                pass

    # NOT LOGGED IN
    else:
        choice = st.radio("Choose Role", ["Register", "User Login", "Admin Login"])
        if choice == "Register":
            register_user()
        elif choice == "User Login":
            user_login()
        elif choice == "Admin Login":
            admin_login()

if __name__ == "__main__":
    main()
