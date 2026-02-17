import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="CSV Analyzer Pro",
    page_icon="📊",
    layout="wide"
)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# ---------- SIDEBAR ----------
st.sidebar.title("📊 CSV Analyzer Pro")
st.sidebar.caption("Production-grade dataset explorer")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ---------- TITLE ----------
st.title("📊 CSV Analyzer Pro")
st.caption("Elite interactive analytics dashboard")

# ---------- MAIN ----------
if uploaded_file:

    try:
        df = load_data(uploaded_file)

        # ===== SIDEBAR INFO =====
        st.sidebar.markdown("---")
        st.sidebar.write("### Dataset Info")
        st.sidebar.write(f"Rows: {df.shape[0]}")
        st.sidebar.write(f"Columns: {df.shape[1]}")
        st.sidebar.write(f"Missing values: {int(df.isna().sum().sum())}")

        # ===== HEADER METRICS =====
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isna().sum().sum()))

        # ===== TABS =====
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "📈 Visualizations",
            "🔎 Data Explorer",
            "📥 Export"
        ])

        # =========================
        # TAB 1 — OVERVIEW
        # =========================
        with tab1:

            st.subheader("Preview")
            st.dataframe(df.head(), use_container_width=True)

            st.subheader("Summary Statistics")
            st.write(df.describe())

            st.subheader("Missing Values by Column")
            missing = df.isna().sum()
            if missing.sum() == 0:
                st.success("No missing values found")
            else:
                st.dataframe(missing[missing > 0])

        # =========================
        # TAB 2 — VISUALIZATIONS
        # =========================
        with tab2:

            numeric_cols = df.select_dtypes(include="number").columns.tolist()

            if not numeric_cols:
                st.warning("No numeric columns available")

            else:

                col1, col2 = st.columns(2)

                with col1:
                    column = st.selectbox("Select column", numeric_cols)

                with col2:
                    chart = st.selectbox(
                        "Chart type",
                        ["Histogram", "Line", "Box", "Scatter"]
                    )

                # SINGLE CHART
                if chart == "Histogram":
                    fig = px.histogram(df, x=column)

                elif chart == "Line":
                    fig = px.line(df, y=column)

                elif chart == "Box":
                    fig = px.box(df, y=column)

                else:
                    xcol = st.selectbox("X-axis", numeric_cols, key="scatterx")
                    ycol = st.selectbox("Y-axis", numeric_cols, key="scattery")
                    fig = px.scatter(df, x=xcol, y=ycol)

                st.plotly_chart(fig, use_container_width=True)

                # MULTI COLUMN COMPARISON
                st.subheader("Compare Multiple Columns")

                multi = st.multiselect(
                    "Select numeric columns",
                    numeric_cols
                )

                if len(multi) >= 2:
                    fig2 = px.line(df, y=multi)
                    st.plotly_chart(fig2, use_container_width=True)

                # CORRELATION
                st.subheader("Correlation Heatmap")

                corr = df[numeric_cols].corr()

                fig3 = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto"
                )

                st.plotly_chart(fig3, use_container_width=True)

        # =========================
        # TAB 3 — DATA EXPLORER
        # =========================
        with tab3:

            st.subheader("Filter Dataset")

            selected_cols = st.multiselect(
                "Choose columns",
                df.columns,
                default=df.columns[:min(5, len(df.columns))]
            )

            if selected_cols:
                filtered_df = df[selected_cols]
            else:
                filtered_df = df.copy()

            search = st.text_input("Search in dataset")

            if search:
                mask = filtered_df.astype(str).apply(
                    lambda x: x.str.contains(search, case=False, na=False)
                ).any(axis=1)
                filtered_df = filtered_df[mask]

            st.dataframe(filtered_df, use_container_width=True)

        # =========================
        # TAB 4 — EXPORT
        # =========================
        with tab4:

            st.subheader("Download Dataset")

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download CSV",
                csv,
                "cleaned_data.csv",
                "text/csv"
            )

            st.success("Ready for download")

    except Exception as e:
        st.error("Error processing CSV")
        st.exception(e)

else:
    st.info("Upload a CSV file from the sidebar to begin.")
