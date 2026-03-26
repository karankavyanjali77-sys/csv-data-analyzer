# 📊 Smart CSV Analyzer

> An AI-powered, production-grade CSV data exploration tool — upload any dataset and instantly get profiling, visualisations, anomaly detection, and natural-language insights via Groq LLM.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq_LLM-LLaMA_3-00A67E?style=flat-square)](https://console.groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## 🚀 Live Demo

👉 **[Open App on Streamlit Cloud](https://your-app-link.streamlit.app)** ← deploy and paste link here

---

## ✨ Features

| Feature | Details |
|---|---|
| 🗂️ **Instant Profiling** | Shape, types, null counts, duplicates, memory usage |
| 📈 **5 Chart Types** | Histogram, Box Plot, Line, Scatter (with trendline), Bar (Categorical) |
| 🔥 **Correlation Heatmap** | Full numeric correlation matrix with colour coding |
| ⚠️ **Anomaly Detection** | IQR-based outlier detection across all numeric columns |
| 📊 **Data Quality Score** | 0–100 score based on completeness, duplicates, and column health |
| 🤖 **AI Insights (Groq)** | Natural-language summary + free-form Q&A about your data |
| 🔎 **Smart Explorer** | Column filter, full-text search, and numeric range filter |
| ⬇️ **Export** | Download filtered or full dataset as CSV |

---

## 🏗️ Project Structure

```
csv-data-analyzer/
│
├── app.py              # Streamlit UI — presentation layer only
├── analyzer.py         # Data processing & chart logic (pure functions, testable)
├── ai_insights.py      # Groq LLM integration — decoupled from UI
├── requirements.txt    # Pinned dependencies
└── README.md
```

> **Architecture note:** The codebase follows a clean separation of concerns — `app.py` handles only UI state, `analyzer.py` handles all data logic, and `ai_insights.py` handles LLM calls. Each module is independently usable and testable.

---

## ⚙️ Run Locally

```bash
git clone https://github.com/karankavyanjali77-sys/csv-data-analyzer.git
cd csv-data-analyzer
pip install -r requirements.txt
streamlit run app.py
```

For AI Insights, get a **free** Groq API key at [console.groq.com](https://console.groq.com) and enter it in the app.

---

## 🧠 Tech Stack

- **Python 3.10+** · **Streamlit** · **Pandas** · **Plotly** · **NumPy**
- **Groq LLM API** (LLaMA 3 8B) for natural-language data analysis
- IQR-based statistical outlier detection
- `@st.cache_data` for performant re-renders

---

## 📸 Screenshots

### Dashboard Overview
*(add screenshot)*

### Visualizations
*(add screenshot)*

### AI Insights
*(add screenshot)*

---

## 🔮 Roadmap

- [ ] Natural language to Pandas query (text → filter)
- [ ] Auto-generated PDF reports
- [ ] Support for Excel (.xlsx) files
- [ ] Multi-file upload and dataset join

---

## 👩‍💻 Author

**Kavyanjali Karan** · B.Tech CSE (AI/ML)
[LinkedIn](https://linkedin.com/in/kavyanjali-karan-a69b60371) · [GitHub](https://github.com/karankavyanjali77-sys) · [Email](mailto:karankavyanjali77@gmail.com)
