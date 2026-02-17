import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 CSV Data Analyzer Pro")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.write(df.head())

    st.subheader("Dataset Info")
    st.write(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

    st.subheader("Column Names")
    st.write(list(df.columns))

    # SUMMARY STATS
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # COLUMN SELECT
    column = st.selectbox("Select column to visualize", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):

        chart_type = st.radio(
            "Choose chart type:",
            ["Histogram", "Line Chart", "Box Plot"]
        )

        fig, ax = plt.subplots()

        if chart_type == "Histogram":
            df[column].hist(ax=ax)

        elif chart_type == "Line Chart":
            ax.plot(df[column])

        elif chart_type == "Box Plot":
            ax.boxplot(df[column])

        ax.set_title(f"{chart_type} of {column}")
        st.pyplot(fig)

    else:
        st.warning("Selected column is not numeric.")

    # DOWNLOAD OPTION
    st.subheader("Download Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="processed_data.csv",
        mime="text/csv",
    )
