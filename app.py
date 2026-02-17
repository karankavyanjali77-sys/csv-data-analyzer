import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("CSV Data Analyzer")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("### Data Preview")
    st.write(df.head())

    st.write("### Column Names")
    st.write(df.columns)

    column = st.selectbox("Select column to visualize", df.columns)

    if df[column].dtype != 'object':
        fig, ax = plt.subplots()
        df[column].hist(ax=ax)
        st.pyplot(fig)
    else:
        st.write("Selected column is not numeric.")
