# app.py
import streamlit as st

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style.css")


# --- CONFIG ---
st.set_page_config(page_title="Insurance Prediction App", page_icon="🧑‍💻", layout="wide")

# --- LOGIN FUNCTION ---
def check_login(username, password):
    return username == "Selva" and password == "0412"

# --- SESSION STATE SETUP ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- LOGIN PAGE ---
if not st.session_state.logged_in:
    st.markdown(
        """
        <h2 style='text-align:center;'>🔐 Login to Access the App</h2>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

        if login_btn:
            if check_login(username, password):
                st.session_state.logged_in = True
                st.success("✅ Login successful! Use the sidebar to navigate.")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

# --- MAIN APP AFTER LOGIN ---
else:
    st.sidebar.success("✅ Logged in as Selva")
    st.sidebar.write("Use the sidebar to navigate pages 👉")
    st.markdown(
        """
        <h1 style='text-align:center; color:#4CAF50;'>Insurance Prediction Web App</h1>
        <p style='text-align:center;'>Welcome Selva 👋. Use the sidebar to explore Prediction, Leaderboard, and Data Exploration.</p>
        """,
        unsafe_allow_html=True,
    )
