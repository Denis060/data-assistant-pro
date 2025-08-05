import streamlit as st
import pandas as pd

@st.cache_data
def load_data(uploaded_file):
    """Loads data from an uploaded CSV file."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None