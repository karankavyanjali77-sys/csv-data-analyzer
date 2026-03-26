import streamlit as st
import pandas as pd
from analyzer import (
    compute_overview,
    compute_missing,
    detect_outliers,
    compute_data_quality_score,
    get_numeric_cols,
    get_categorical_cols,
    build_histogram,
    build_line,
    build_box,
    build_scatter,
    build_heatmap,
    build_bar_categorical,
    build_multi_line,
)
from ai_insights import generate_ai_summary

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart CSV Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fb;
        border: 1px solid #e8eaed;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #1a1a2e; }
    .metric-label { font-size: 12px; color: #6b7280; margin-top: 2px; }
    .quality-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 2px solid #6c63ff;
    }
    .ai-box {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border: 1px solid #6c63ff40;
        border-radius: 12px;
        padding: 20px;
        margin-top: 12px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Smart CSV Analyzer")
    st.caption("AI-powered data exploration & insights")
    st.divider()

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    st.divider()
    st.markdown("#### ⚙️ Settings")
    max_rows_preview = st.slider("Preview rows", 5, 50, 10)
    show_ai = st.toggle("Enable AI Insights (Groq)", value=True)

    if uploaded_file:
        st.divider()
        st.markdown("#### 📁 File Info")
        st.write(f"**Name:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

    st.divider()
    st.caption("Built by Kavyanjali Karan · AI/ML Engineer")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Smart CSV Analyzer")
st.caption("Upload any CSV — get instant profiling, visualizations, anomaly detection & AI insights")

if not uploaded_file:
    st.info("👈 Upload a CSV file from the sidebar to begin analysis.")
    st.stop()

# ─── LOAD & CACHE ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_csv(file):
    return pd.read_csv(file)

try:
    df = load_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# ─── DERIVED DATA ─────────────────────────────────────────────────────────────
overview      = compute_overview(df)
missing_info  = compute_missing(df)
outlier_info  = detect_outliers(df)
quality_score = compute_data_quality_score(df)
num_cols      = get_numeric_cols(df)
cat_cols      = get_categorical_cols(df)

# ─── TOP METRICS ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, overview["rows"],        "Rows"),
    (c2, overview["cols"],        "Columns"),
    (c3, overview["missing"],     "Missing Values"),
    (c4, overview["duplicates"],  "Duplicate Rows"),
    (c5, f"{quality_score}%",     "Data Quality Score"),
]
for col, val, label in metrics:
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{val}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.write("")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗂️ Overview",
    "📈 Visualizations",
    "⚠️ Data Quality",
    "🔎 Explorer",
    "🤖 AI Insights",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(max_rows_preview), use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Column Types</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.notna().sum().values,
            "Null %": (df.isna().mean() * 100).round(1).values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(include="all").T, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not num_cols and not cat_cols:
        st.warning("No plottable columns detected.")
    else:
        # ── Single-column chart ──
        st.markdown('<div class="section-header">Single Column Chart</div>', unsafe_allow_html=True)
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            chart_type = st.selectbox(
                "Chart type",
                ["Histogram", "Box Plot", "Line", "Scatter", "Bar (Categorical)"],
            )

        if chart_type in ["Histogram", "Box Plot", "Line"] and num_cols:
            with v_col2:
                col_sel = st.selectbox("Column", num_cols)
            if chart_type == "Histogram":
                st.plotly_chart(build_histogram(df, col_sel), use_container_width=True)
            elif chart_type == "Box Plot":
                st.plotly_chart(build_box(df, col_sel), use_container_width=True)
            elif chart_type == "Line":
                st.plotly_chart(build_line(df, col_sel), use_container_width=True)

        elif chart_type == "Scatter" and len(num_cols) >= 2:
            with v_col2:
                x_col = st.selectbox("X axis", num_cols, key="sx")
            y_col = st.selectbox("Y axis", num_cols, key="sy", index=1)
            color_col = st.selectbox("Color by (optional)", ["None"] + cat_cols, key="sc")
            color_by = None if color_col == "None" else color_col
            st.plotly_chart(build_scatter(df, x_col, y_col, color_by), use_container_width=True)

        elif chart_type == "Bar (Categorical)" and cat_cols:
            with v_col2:
                cat_col_sel = st.selectbox("Column", cat_cols, key="bc")
            top_n = st.slider("Top N categories", 5, 30, 10)
            st.plotly_chart(build_bar_categorical(df, cat_col_sel, top_n), use_container_width=True)

        else:
            st.info("Not enough suitable columns for this chart type.")

        # ── Multi-column comparison ──
        if num_cols:
            st.divider()
            st.markdown('<div class="section-header">Multi-Column Comparison</div>', unsafe_allow_html=True)
            multi_sel = st.multiselect("Select numeric columns to compare", num_cols)
            if len(multi_sel) >= 2:
                st.plotly_chart(build_multi_line(df, multi_sel), use_container_width=True)

        # ── Correlation Heatmap ──
        if len(num_cols) >= 2:
            st.divider()
            st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
            st.plotly_chart(build_heatmap(df, num_cols), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    dq_left, dq_right = st.columns(2)

    with dq_left:
        st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
        if missing_info.empty:
            st.success("✅ No missing values found in the dataset.")
        else:
            st.dataframe(missing_info, use_container_width=True, hide_index=True)

    with dq_right:
        st.markdown('<div class="section-header">Outlier Detection (IQR method)</div>', unsafe_allow_html=True)
        if outlier_info.empty:
            st.success("✅ No significant outliers detected.")
        else:
            st.dataframe(outlier_info, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<div class="section-header">Duplicate Rows</div>', unsafe_allow_html=True)
    dups = df[df.duplicated()]
    if dups.empty:
        st.success("✅ No duplicate rows found.")
    else:
        st.warning(f"⚠️ {len(dups)} duplicate rows detected.")
        with st.expander("View duplicates"):
            st.dataframe(dups, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Filter & Search</div>', unsafe_allow_html=True)

    e1, e2 = st.columns([2, 1])
    with e1:
        sel_cols = st.multiselect(
            "Select columns to display",
            df.columns.tolist(),
            default=df.columns[:min(6, len(df.columns))].tolist(),
        )
    with e2:
        search_term = st.text_input("Search across all values", placeholder="Type to filter rows...")

    working_df = df[sel_cols] if sel_cols else df.copy()

    if search_term:
        mask = working_df.astype(str).apply(
            lambda col: col.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        working_df = working_df[mask]
        st.caption(f"{len(working_df)} rows match '{search_term}'")

    st.dataframe(working_df, use_container_width=True)

    # ── Numeric range filter ──
    if num_cols:
        st.divider()
        st.markdown('<div class="section-header">Numeric Range Filter</div>', unsafe_allow_html=True)
        range_col = st.selectbox("Column to filter by range", num_cols)
        col_min = float(df[range_col].min())
        col_max = float(df[range_col].max())
        r_min, r_max = st.slider(
            f"Range for {range_col}",
            col_min, col_max,
            (col_min, col_max),
        )
        filtered = df[(df[range_col] >= r_min) & (df[range_col] <= r_max)]
        st.write(f"{len(filtered)} rows in selected range")
        st.dataframe(filtered.head(50), use_container_width=True)

    # ── Export ──
    st.divider()
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Download filtered data as CSV",
            working_df.to_csv(index=False).encode("utf-8"),
            "filtered_data.csv",
            "text/csv",
        )
    with dl2:
        st.download_button(
            "⬇️ Download full dataset as CSV",
            df.to_csv(index=False).encode("utf-8"),
            "full_data.csv",
            "text/csv",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">AI-Powered Dataset Summary</div>', unsafe_allow_html=True)
    st.caption("Powered by Groq LLM — instant natural language insights about your data")

    if not show_ai:
        st.info("Enable 'AI Insights' in the sidebar to use this feature.")
    else:
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get a free key at console.groq.com",
        )

        if st.button("🤖 Generate AI Insights", type="primary"):
            if not groq_key:
                st.warning("Please enter your Groq API key.")
            else:
                with st.spinner("Analysing your dataset with AI..."):
                    result = generate_ai_summary(df, groq_key)

                if result["success"]:
                    st.markdown(
                        f'<div class="ai-box">{result["summary"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"AI Error: {result['error']}")

        st.divider()
        st.markdown('<div class="section-header">Ask a Question About Your Data</div>', unsafe_allow_html=True)
        user_question = st.text_input("Ask anything about your dataset", placeholder="e.g. Which column has the most outliers?")

        if st.button("Ask AI", type="secondary") and user_question:
            if not groq_key:
                st.warning("Please enter your Groq API key above.")
            else:
                with st.spinner("Thinking..."):
                    result = generate_ai_summary(df, groq_key, question=user_question)
                if result["success"]:
                    st.markdown(
                        f'<div class="ai-box">{result["summary"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"AI Error: {result['error']}")
