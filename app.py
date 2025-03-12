import streamlit as st
import os
from src.model import process_finance_query, detect_company_ticker
from src.utils import format_latex_content

# Set up page
st.set_page_config(page_title="Finance Bro AI", layout="wide")

# Title and description
st.title("Finance Bro AI Investment Banking Interview Coach")
st.write("Ask any investment banking interview question to get prepared for your interviews!")

# Input controls
query = st.text_input("Your question:")
col1, col2 = st.columns(2)
with col1:
    experience = st.selectbox("Experience level:", ["Standard", "Junior", "Experienced"])
with col2:
    bro_mode = st.checkbox("Activate Bro Mode")

# Process query when submitted
if st.button("Submit"):
    with st.spinner("Finance Bro is thinking..."):
        response = process_finance_query(
            query, 
            experience_level=experience.lower(), 
            bro_mode=bro_mode
        )
    
    # Format and display the response
    formatted_response = format_latex_content(response)
    st.markdown(formatted_response)
