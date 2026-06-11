# 🤖 AI Resume Screening System

> **YBI Foundation — Machine Learning & Data Science Internship Project**

An ATS-style **Resume Screening and Candidate Ranking** platform built with Python and Streamlit. Upload multiple resumes, paste a job description, and get AI-powered analysis with semantic matching, skill extraction, composite ATS scoring, and interactive visual dashboards.

---

## 🚀 Live Demo

<div align="center">

[![🚀 Open Live App](https://img.shields.io/badge/🚀%20Open%20Live%20App-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ai-resume-screening-system-app.streamlit.app/)

</div>

> ⚡ Click the button above to launch the app — no installation required!

---

## 🏫 Internship Context

| Field | Details |
|---|---|
| **Organization** | YBI Foundation |
| **Program** | Machine Learning & Data Science Internship |
| **Domain** | Natural Language Processing · Machine Learning · Web App Deployment |
| **Tech Focus** | Python · scikit-learn · TF-IDF · Streamlit · Plotly |
| **Project Type** | End-to-End ML Application with Live Deployment |

---

## 📸 Screenshots

### 🏠 Main Dashboard — Candidate Rankings & ATS Scores
![Main Dashboard](screenshots/Screenshot%202026-06-11%20143355.png)

### 📊 Score Breakdown — Semantic Match, Skill Analysis, Gauge Chart
![Score Breakdown](screenshots/Screenshot%202026-06-11%20143414.png)

### 🔍 Comparison View — Radar Chart & Score Comparison Table
![Comparison View](screenshots/Screenshot%202026-06-11%20143430.png)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Resume Upload** | Upload multiple PDF and DOCX resumes simultaneously |
| 🤖 **Semantic AI Matching** | TF-IDF vectorization with n-gram support for intelligent text matching |
| 🎯 **Skill Extraction** | 140+ skills database with alias resolution and word-boundary matching |
| 📊 **ATS Scoring** | Composite score: 40% semantic + 30% skills + 20% experience + 10% education |
| 🏆 **Candidate Ranking** | Rank all candidates by ATS score with detailed breakdowns |
| ⚠️ **Missing Skill Analysis** | Identify matched, missing, and extra skills per candidate |
| 📈 **Visual Dashboard** | Interactive Plotly charts: gauge, donut, bar, radar, breakdown |
| 📝 **AI Recruiter Summary** | Auto-generated professional assessment with recommendations |
| 📥 **PDF Reports** | Download branded PDF analysis reports for each candidate |
| 🎨 **Professional UI** | Dark theme with glassmorphism, gradients, and micro-animations |

---

## 🧠 How It Works

### ATS Score Formula

```
ATS Score = 0.40 × Semantic Similarity
          + 0.30 × Skill Match Percentage
          + 0.20 × Experience Keywords Score
          + 0.10 × Education Keywords Score
```

| Score Range | Label |
|---|---|
| ≥ 85% | 🟢 Excellent Match |
| ≥ 70% | 🔵 Good Match |
| ≥ 50% | 🟡 Average Match |
| < 50% | 🔴 Weak Match |

### Semantic Matching Pipeline

```
Resume Text  → Preprocessing → TF-IDF Vectorization (unigrams + bigrams)
                                                        → Cosine Similarity → Scaled Score (0–100)
Job Desc.    → Preprocessing → TF-IDF Vectorization (unigrams + bigrams)
```

- **Lightweight**: No large model downloads — uses scikit-learn (already a dependency)
- **Fast**: TF-IDF vectorization is near-instant, no GPU required
- **Deployable**: Works within Streamlit Cloud's 1 GB RAM limit

### Skill Extraction

- **140+ skills** across 12 categories (Programming, Backend, Frontend, Cloud, Data/ML, DevOps, Databases, Concepts, Soft Skills, Design, Testing, Certifications)
- **Alias resolution**: `"JS"` → `"JavaScript"`, `"k8s"` → `"Kubernetes"`
- **Word-boundary matching**: Prevents false positives (e.g., `"R"` inside `"React"`)

---

## 🏗️ Project Architecture

```
AI-Resume-Screening-System/
│
├── app.py                    # Main Streamlit application (orchestration only)
│
├── utils/
│   ├── __init__.py
│   ├── parser.py             # PDF/DOCX text extraction and cleaning
│   ├── skills.py             # Skill extraction with regex + alias matching
│   ├── matcher.py            # Semantic similarity via TF-IDF (scikit-learn)
│   ├── ats.py                # ATS scoring engine
│   ├── ranking.py            # Candidate ranking and summary generation
│   ├── charts.py             # Interactive Plotly visualizations
│   └── report.py             # PDF report generation (fpdf2)
│
├── data/
│   └── skills_database.csv   # 140+ skills with categories and aliases
│
├── assets/
│   └── style.css             # Premium dark theme design system
│
├── screenshots/              # App screenshots for documentation
│
├── sample_resumes/           # Sample resumes for testing
│
├── .streamlit/
│   └── config.toml           # Streamlit theme and server configuration
│
├── requirements.txt          # Python dependencies
├── packages.txt              # System dependencies for Streamlit Cloud
└── README.md
```

---

## 📦 Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit, Plotly, Custom CSS |
| **AI / ML** | scikit-learn (TF-IDF + Cosine Similarity) |
| **NLP** | Regex-based skill extraction, n-gram tokenization |
| **PDF Parsing** | pdfplumber, PyPDF2 |
| **DOCX Parsing** | python-docx |
| **Data Processing** | Pandas, NumPy |
| **PDF Reports** | fpdf2 |
| **Deployment** | Streamlit Community Cloud |

---

## 🚀 Quick Start (Run Locally)

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Manishrao2004/AI-Resume-Screening-System.git
cd AI-Resume-Screening-System

# 2. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at **`http://localhost:8501`** 🎉

---

## ☁️ Deploy to Streamlit Cloud

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: AI Resume Screening System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AI-Resume-Screening-System.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Click **"New app"**
3. Connect your GitHub repository
4. Set:
   - **Repository:** `YOUR_USERNAME/AI-Resume-Screening-System`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy"** 🚀

### Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Verify all packages are in `requirements.txt` |
| App crashes / reboots | Likely OOM — check Streamlit Cloud logs |
| Slow first load | Normal — pip installs on first deploy |
| Upload size limit | Max 50 MB per file (configured in `config.toml`) |

## 🙏 Acknowledgments

- [YBI Foundation](https://www.ybifoundation.org/) for the ML & DS Internship opportunity
- [scikit-learn](https://scikit-learn.org/) for TF-IDF vectorization and cosine similarity
- [Streamlit](https://streamlit.io/) for the web framework
- [Plotly](https://plotly.com/) for interactive visualizations
- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF text extraction
- [fpdf2](https://pyfpdf.github.io/fpdf2/) for PDF report generation

---

<div align="center">

Made with ❤️ for the **YBI Foundation ML & DS Internship**

[![🚀 Open Live App](https://img.shields.io/badge/🚀%20Open%20Live%20App-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ai-resume-screening-system-app.streamlit.app/)

</div>
