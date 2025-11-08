# streamlit_app.py
import streamlit as st
import pandas as pd
from workflow import run_workflow  # Assume this triggers the crew

st.title("🧠 Dataset Auditor (Local LLM Powered)")

uploaded_file = st.file_uploader("Upload your CSV", type="csv")
if uploaded_file:
    with open("data/input.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    output = run_workflow("data/input.csv")
    st.markdown("### 📝 Audit Report")
    st.text(output)