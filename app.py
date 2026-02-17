import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="CSV Analyzer Pro",
    page_icon="📊",
    layout="wide"
)

# ---------- DATA LOADING ----------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# ---------- SIDEBAR ----------
st.sidebar.title("CSV Analyzer Pro")
st.sidebar.write("Upload a dataset and explore insights")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ---------- MAIN ----------
st.title("📊 CSV Analyzer Pro")
st.caption("Production-style interactive dataset explorer")

if uploaded_file:

    try:
        df = load_data(uploaded_file)

        # ===== OVERVIEW =====
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isna().sum().sum()))

        # ===== PREVIEW =====
        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        # ===== SUMMARY =====
        st.subheader("Summary Statistics")
        st.write(df.describe())

        # ===== MISSING VALUES =====
        st.subheader("Missing Values by Column")
        missing = df.isna().sum()
        if missing.sum() == 0:
            st.success("No missing values found")
        else:
            st.write(missing[missing > 0])

        # ===== NUMERIC COLUMNS =====
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:

            st.subheader("Single Column Visualization")

            column = st.selectbox("Choose numeric column", numeric_cols)
            chart = st.radio("Chart type", ["Histogram", "Line", "Box"])

            fig, ax = plt.subplots()

            if chart == "Histogram":
                df[column].hist(ax=ax)
            elif chart == "Line":
                ax.plot(df[column])
            else:
                ax.boxplot(df[column])

            ax.set_title(f"{chart} plot of {column}")
            st.pyplot(fig)

            # ===== MULTI COLUMN COMPARISON =====
            st.subheader("Compare Multiple Columns")

            multi_cols = st.multiselect(
                "Select columns",
                numeric_cols
            )

            if len(multi_cols) >= 2:
                fig2, ax2 = plt.subplots()
                df[multi_cols].plot(ax=ax2)
                st.pyplot(fig2)

            # ===== CORRELATION =====
            st.subheader("Correlation Heatmap")

            corr = df[numeric_cols].corr()
            fig3, ax3 = plt.subplots()
            cax = ax3.imshow(corr)
            ax3.set_xticks(range(len(corr.columns)))
            ax3.set_xticklabels(corr.columns, rotation=90)
            ax3.set_yticks(range(len(corr.columns)))
            ax3.set_yticklabels(corr.columns)
            fig3.colorbar(cax)
            st.pyplot(fig3)

        else:
            st.warning("No numeric columns available")

        # ===== DOWNLOAD =====
        st.subheader("Download Processed Data")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            "processed_data.csv",
            "text/csv"
        )

    except Exception as e:
        st.error("Error processing file. Please upload a valid CSV.")
        st.exception(e)

else:
    st.info("Upload a CSV file from the sidebar to begin.")
