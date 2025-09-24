# auth.py
import streamlit as st

def require_login():
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.warning("Please log in to access this page.")
        st.stop()
