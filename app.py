"""
AI Resume Screening System

An ATS-style Resume Screening and Candidate Ranking platform.
Upload resumes, paste a job description, and get AI-powered
analysis with semantic matching, skill extraction, ATS scoring,
and visual dashboards.

Author: AI Resume Screening System
License: MIT
"""

import os
import streamlit as st
import pandas as pd

from utils.parser import extract_text
from utils.skills import load_skills_database, extract_skills, get_skill_analysis
from utils.matcher import load_model, compute_similarity, compute_batch_similarity
from utils.ats import compute_ats_score, get_experience_score, get_education_score, get_ats_label
from utils.ranking import rank_candidates, generate_summary, generate_skill_recommendations
from utils.charts import (
    create_ats_gauge,
    create_skill_pie,
    create_ranking_bar,
    create_comparison_radar,
    create_score_breakdown_bar,
)
from utils.report import generate_pdf_report


# ═══════════════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════
# Load CSS
# ═══════════════════════════════════════════════════════════════════

def load_css():
    """Load external CSS file."""
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ═══════════════════════════════════════════════════════════════════
# Session State Initialization
# ═══════════════════════════════════════════════════════════════════

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = ''


# ═══════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    # Branding
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.3rem;">📄</div>
        <h2 style="margin: 0; font-size: 1.3rem;">AI Resume Screener</h2>
        <p style="color: #8B8FA3; font-size: 0.8rem; margin-top: 0.3rem;">
            Powered by AI Matching
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # File Upload
    st.markdown("### 📁 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Drag and drop or browse to upload multiple resume files.",
    )

    if uploaded_files:
        st.success(f"📎 {len(uploaded_files)} file(s) uploaded")

    st.markdown("---")

    # Job Description Input
    st.markdown("### 📋 Job Description")
    job_description = st.text_area(
        "Paste the job description here",
        height=200,
        placeholder=(
            "Example:\n\n"
            "Looking for a Python Backend Engineer.\n\n"
            "Required Skills:\n"
            "- Python\n"
            "- SQL\n"
            "- AWS\n"
            "- Docker\n"
            "- REST APIs\n"
            "- Git\n\n"
            "Requirements:\n"
            "- 3+ years of experience\n"
            "- Bachelor's degree in CS"
        ),
        help="Paste the complete job description including required skills, experience, and qualifications.",
    )

    st.markdown("---")

    # Analyze Button
    analyze_clicked = st.button(
        "🚀 Analyze Resumes",
        disabled=not (uploaded_files and job_description),
        use_container_width=True,
    )

    # Clear Results button
    if st.session_state.analyzed:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.analyzed = False
            st.session_state.jd_text = ''
            # Clear any cached PDF report data
            keys_to_clear = [k for k in st.session_state if k.startswith('pdf_data_')]
            for k in keys_to_clear:
                del st.session_state[k]
            st.rerun()

    if not uploaded_files:
        st.info("⬆️ Upload at least one resume to begin.")
    elif not job_description:
        st.info("📝 Paste a job description to continue.")

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0;">
        <p style="color: #5A5E72; font-size: 0.7rem; margin: 0;">
            Built with Streamlit & scikit-learn<br>
            TF-IDF Semantic Matching
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════

def run_analysis(files, jd_text: str) -> list[dict]:
    """
    Run the complete analysis pipeline on uploaded resumes.

    Steps:
    1. Parse each resume (PDF/DOCX → text)
    2. Load skills database
    3. Extract skills from JD and each resume
    4. Compute semantic similarity (transformer embeddings)
    5. Compute experience and education scores
    6. Compute composite ATS score
    7. Rank candidates

    Args:
        files: List of uploaded files.
        jd_text: Job description text.

    Returns:
        List of candidate result dictionaries.
    """
    results = []
    errors = []
    parsed_resumes = []

    # Step 1: Load resources
    progress = st.progress(0, text="Initializing AI engine...")
    model = load_model()
    skills_db = load_skills_database()

    # Step 2: Extract JD skills
    progress.progress(10, text="Analyzing job description...")
    jd_skills = extract_skills(jd_text, skills_db)

    # Step 3: Parse all resumes first
    total = len(files)
    for i, file in enumerate(files):
        pct = int(15 + (i / total) * 40)
        progress.progress(pct, text=f"Parsing resume {i+1}/{total}: {file.name}...")

        parsed = extract_text(file)

        if not parsed['success']:
            errors.append({
                'filename': parsed['filename'],
                'error': parsed['error'],
            })
            if not parsed['text']:
                continue

        parsed_resumes.append(parsed)

    # Step 4: Batch semantic similarity (much faster than individual calls)
    progress.progress(60, text="Computing semantic similarity...")
    resume_texts = [p['text'] for p in parsed_resumes]
    semantic_scores = compute_batch_similarity(resume_texts, jd_text, model)

    # Step 5: Score each resume
    for i, (parsed, semantic_score) in enumerate(zip(parsed_resumes, semantic_scores)):
        pct = int(65 + (i / max(1, len(parsed_resumes))) * 30)
        progress.progress(pct, text=f"Scoring candidate {i+1}/{len(parsed_resumes)}...")

        resume_text = parsed['text']

        # Extract skills
        resume_skills = extract_skills(resume_text, skills_db)
        skill_analysis = get_skill_analysis(resume_skills, jd_skills)

        # Experience & education scores
        exp_score = get_experience_score(resume_text)
        edu_score = get_education_score(resume_text)

        # ATS score
        ats_score = compute_ats_score(
            semantic_score=semantic_score,
            skill_match_pct=skill_analysis['match_percentage'],
            experience_score=exp_score,
            education_score=edu_score,
        )
        label, color = get_ats_label(ats_score)

        results.append({
            'filename': parsed['filename'],
            'text': resume_text,
            'ats_score': ats_score,
            'semantic_score': semantic_score,
            'skill_match_pct': skill_analysis['match_percentage'],
            'experience_score': exp_score,
            'education_score': edu_score,
            'label': label,
            'label_color': color,
            'matched_skills': skill_analysis['matched'],
            'missing_skills': skill_analysis['missing'],
            'extra_skills': skill_analysis['extra'],
            'resume_skills': resume_skills,
            'jd_skills': jd_skills,
        })

    progress.progress(100, text="Analysis complete! ✅")

    # Show any parsing errors
    if errors:
        for err in errors:
            st.warning(f"⚠️ Issue with **{err['filename']}**: {err['error']}")

    return results


# ═══════════════════════════════════════════════════════════════════
# Run Analysis on Button Click
# ═══════════════════════════════════════════════════════════════════

if analyze_clicked and uploaded_files and job_description:
    # Validate inputs
    if len(job_description.strip()) < 20:
        st.error("❌ Job description is too short. Please provide a more detailed description.")
    else:
        # Check for duplicates
        filenames = [f.name for f in uploaded_files]
        if len(filenames) != len(set(filenames)):
            st.warning("⚠️ Duplicate filenames detected. Only the last instance of each will be processed.")
            # Deduplicate (keep last)
            seen = {}
            for f in uploaded_files:
                seen[f.name] = f
            uploaded_files = list(seen.values())

        with st.spinner(""):
            new_results = run_analysis(uploaded_files, job_description)
            if new_results:
                # Initialize or reset if JD changed
                if st.session_state.analysis_results is None or st.session_state.get('jd_text', '') != job_description:
                    st.session_state.analysis_results = []
                
                # Accumulate results, keeping the newest version if filename matches
                new_filenames = {r['filename'] for r in new_results}
                accumulated = [r for r in st.session_state.analysis_results if r['filename'] not in new_filenames]
                accumulated.extend(new_results)
                
                st.session_state.analysis_results = accumulated
                st.session_state.analyzed = True
                st.session_state.jd_text = job_description
            elif not st.session_state.analysis_results:
                st.error("❌ No resumes could be processed. Please check your files and try again.")


# ═══════════════════════════════════════════════════════════════════
# Main Content
# ═══════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>AI Resume Screening System</h1>
    <p>Upload resumes • Paste job description • Get AI-powered candidate analysis</p>
</div>
""", unsafe_allow_html=True)


# ─── No Results State ────────────────────────────────────────────
if not st.session_state.analyzed or not st.session_state.analysis_results:
    # Welcome screen
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 3rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🎯</div>
        <h2 style="color: #FAFAFA; margin-bottom: 0.5rem;">Welcome to AI Resume Screener</h2>
        <p style="color: #8B8FA3; max-width: 600px; margin: 0 auto; line-height: 1.6;">
            Upload candidate resumes and paste a job description to get started.
            Our AI will analyze each resume using intelligent text matching,
            extract and compare skills, and rank candidates with a comprehensive ATS score.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    cols = st.columns(4)
    features = [
        ("🤖", "Semantic AI Matching", "TF-IDF powered intelligent understanding of resume relevance"),
        ("📊", "ATS Scoring", "Composite score: semantic + skills + experience + education"),
        ("🎯", "Skill Gap Analysis", "Identify matched, missing, and extra skills instantly"),
        ("📥", "PDF Reports", "Download detailed analysis reports for each candidate"),
    ]

    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; min-height: 180px;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <h4 style="color: #FAFAFA; font-size: 0.95rem; margin-bottom: 0.3rem;">{title}</h4>
                <p style="color: #8B8FA3; font-size: 0.78rem; line-height: 1.4;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.stop()


# ═══════════════════════════════════════════════════════════════════
# Dashboard (Results Available)
# ═══════════════════════════════════════════════════════════════════

results = st.session_state.analysis_results

# Rank candidates
ranked_df = rank_candidates(results)

# ─── Overview Metrics ────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

top_candidate = results[0] if results else None
if ranked_df is not None and not ranked_df.empty:
    top_idx = ranked_df['ats_score'].idxmax()
    for r in results:
        if r['filename'] == ranked_df.loc[top_idx, 'filename']:
            top_candidate = r
            break

avg_score = sum(r['ats_score'] for r in results) / len(results) if results else 0
max_score = max(r['ats_score'] for r in results) if results else 0

metric_cols = st.columns(4)

with metric_cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">📄</div>
        <div class="metric-value">{len(results)}</div>
        <div class="metric-label">Resumes Analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">⭐</div>
        <div class="metric-value">{max_score}%</div>
        <div class="metric-label">Top ATS Score</div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[2]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">📊</div>
        <div class="metric-value">{avg_score:.1f}%</div>
        <div class="metric-label">Average Score</div>
    </div>
    """, unsafe_allow_html=True)

with metric_cols[3]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🏆</div>
        <div class="metric-value" style="font-size: 1.2rem;">
            {top_candidate['filename'][:18] if top_candidate else 'N/A'}
        </div>
        <div class="metric-label">Top Candidate</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════

tab_dashboard, tab_details, tab_comparison, tab_reports = st.tabs([
    "📊 Dashboard",
    "🔍 Candidate Details",
    "⚖️ Comparison",
    "📥 Reports",
])


# ─── Tab 1: Dashboard ───────────────────────────────────────────
with tab_dashboard:
    dash_col1, dash_col2 = st.columns([3, 2])

    with dash_col1:
        st.markdown("#### 🏆 Candidate Rankings")
        fig_ranking = create_ranking_bar(ranked_df)
        st.plotly_chart(fig_ranking, key="ranking_bar", width='stretch')

    with dash_col2:
        if top_candidate:
            st.markdown("#### 🎯 Top Candidate")
            label, color = get_ats_label(top_candidate['ats_score'])
            fig_gauge = create_ats_gauge(top_candidate['ats_score'], label)
            st.plotly_chart(fig_gauge, key="top_gauge", width='stretch')

            # Skill coverage for top candidate
            matched_count = len(top_candidate['matched_skills'])
            missing_count = len(top_candidate['missing_skills'])
            fig_pie = create_skill_pie(matched_count, missing_count)
            st.plotly_chart(fig_pie, key="top_pie", width='stretch')

    # Ranking table and Export
    col_table, col_export = st.columns([4, 1])
    with col_table:
        st.markdown("#### 📋 Detailed Rankings")
    with col_export:
        csv = ranked_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export to CSV",
            data=csv,
            file_name="candidate_rankings.csv",
            mime="text/csv",
            key="export_csv_btn",
            use_container_width=True,
        )

    display_df = ranked_df.copy()
    display_df.columns = [c.replace('_', ' ').title() for c in display_df.columns]
    st.dataframe(
        display_df,
        hide_index=True,
        width='stretch',
    )


# ─── Tab 2: Candidate Details ───────────────────────────────────
with tab_details:
    # Candidate selector
    candidate_names = [r['filename'] for r in results]
    selected_name = st.selectbox(
        "Select a candidate to view detailed analysis",
        candidate_names,
        index=0,
        key="candidate_selector",
    )

    # Find selected candidate
    selected = next((r for r in results if r['filename'] == selected_name), None)

    if selected:
        # Header with score badge
        label, color = get_ats_label(selected['ats_score'])
        score_class = "excellent" if selected['ats_score'] >= 85 else \
                      "good" if selected['ats_score'] >= 70 else \
                      "average" if selected['ats_score'] >= 50 else "weak"

        st.markdown(f"""
        <div class="glass-card" style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <div>
                <h3 style="color: #FAFAFA; margin: 0;">📄 {selected['filename']}</h3>
                <p style="color: #8B8FA3; margin: 0.2rem 0 0 0; font-size: 0.85rem;">
                    Semantic Match: {selected['semantic_score']}% • Skill Match: {selected['skill_match_pct']}%
                </p>
            </div>
            <div class="score-badge score-{score_class}">
                ATS Score: {selected['ats_score']}% — {label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Charts row
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            fig_gauge = create_ats_gauge(selected['ats_score'], label)
            st.plotly_chart(fig_gauge, key="detail_gauge", width='stretch')

        with detail_col2:
            fig_breakdown = create_score_breakdown_bar(selected)
            st.plotly_chart(fig_breakdown, key="detail_breakdown", width='stretch')

        # Skill Analysis
        st.markdown("#### 🎯 Skill Analysis")
        skill_col1, skill_col2, skill_col3 = st.columns(3)

        with skill_col1:
            matched_list = sorted(selected['matched_skills'])
            st.markdown(f"**✅ Matched Skills ({len(matched_list)})**")
            if matched_list:
                tags_html = " ".join(
                    f'<span class="skill-tag skill-matched">{s}</span>'
                    for s in matched_list
                )
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.markdown("_No matched skills found._")

        with skill_col2:
            missing_list = sorted(selected['missing_skills'])
            st.markdown(f"**❌ Missing Skills ({len(missing_list)})**")
            if missing_list:
                tags_html = " ".join(
                    f'<span class="skill-tag skill-missing">{s}</span>'
                    for s in missing_list
                )
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.markdown("_No missing skills — full match! 🎉_")

        with skill_col3:
            extra_list = sorted(selected['extra_skills'])
            st.markdown(f"**🌟 Extra Skills ({len(extra_list)})**")
            if extra_list:
                tags_html = " ".join(
                    f'<span class="skill-tag skill-extra">{s}</span>'
                    for s in extra_list
                )
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.markdown("_No additional skills beyond requirements._")

        # Skill coverage pie chart
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        pie_col1, pie_col2 = st.columns([1, 2])

        with pie_col1:
            fig_pie = create_skill_pie(len(matched_list), len(missing_list))
            st.plotly_chart(fig_pie, key="detail_pie", width='stretch')

        with pie_col2:
            # Recommendations
            if missing_list:
                st.markdown("#### 💡 Recommendations")
                recs = generate_skill_recommendations(missing_list)
                for rec in recs:
                    st.markdown(rec)

        # Recruiter Summary
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        with st.expander("📝 AI Recruiter Summary", expanded=True):
            summary = generate_summary(selected)
            st.markdown(summary)


# ─── Tab 3: Comparison ──────────────────────────────────────────
with tab_comparison:
    if len(results) >= 2:
        st.markdown("#### ⚖️ Multi-Candidate Comparison")
        fig_radar = create_comparison_radar(results)
        st.plotly_chart(fig_radar, key="comparison_radar", width='stretch')

        # Side-by-side comparison table
        st.markdown("#### 📊 Score Comparison Table")

        comp_data = []
        for r in sorted(results, key=lambda x: x['ats_score'], reverse=True):
            comp_data.append({
                "Candidate": r['filename'],
                "ATS Score": f"{r['ats_score']}%",
                "Semantic": f"{r['semantic_score']}%",
                "Skills": f"{r['skill_match_pct']}%",
                "Experience": f"{r['experience_score']}%",
                "Education": f"{r['education_score']}%",
                "Verdict": r['label'],
                "Matched": len(r['matched_skills']),
                "Missing": len(r['missing_skills']),
            })

        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, hide_index=True, width='stretch')

        # Skill overlap analysis
        st.markdown("#### 🔗 Common Skills Across Candidates")
        if len(results) >= 2:
            all_resume_skills = [set(r['resume_skills']) for r in results]
            common_skills = set.intersection(*all_resume_skills) if all_resume_skills else set()

            if common_skills:
                tags_html = " ".join(
                    f'<span class="skill-tag skill-matched">{s}</span>'
                    for s in sorted(common_skills)
                )
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.info("No skills are common across all candidates.")

    else:
        st.info("📊 Upload at least 2 resumes to enable comparison view.")


# ─── Tab 4: Reports ──────────────────────────────────────────────
with tab_reports:
    st.markdown("#### 📥 Download Candidate Reports")
    st.markdown("Generate and download detailed PDF analysis reports for each candidate.")

    num_cols = min(len(results), 3) if len(results) > 0 else 1
    report_cols = st.columns(num_cols)

    for i, candidate in enumerate(sorted(results, key=lambda x: x['ats_score'], reverse=True)):
        col_idx = i % num_cols
        with report_cols[col_idx]:
            label, color = get_ats_label(candidate['ats_score'])

            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <h4 style="color: #FAFAFA; font-size: 0.9rem; margin-bottom: 0.3rem;">
                    #{i+1} {candidate['filename'][:20]}
                </h4>
                <div style="font-size: 1.5rem; font-weight: 800; color: {color};">
                    {candidate['ats_score']}%
                </div>
                <p style="color: {color}; font-size: 0.75rem; margin: 0.2rem 0 0.8rem 0;">
                    {label}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Generate PDF report only when requested
            safe_name = candidate['filename'].rsplit('.', 1)[0]
            pdf_key = f"pdf_data_{i}_{candidate['filename']}"
            gen_btn_key = f"gen_btn_{i}_{candidate['filename']}"

            if pdf_key not in st.session_state:
                if st.button("⚙️ Generate Report", key=gen_btn_key, use_container_width=True):
                    with st.spinner("Generating PDF..."):
                        # Generate and store in session state
                        pdf_bytes = generate_pdf_report(candidate, rank=i+1)
                        st.session_state[pdf_key] = pdf_bytes
                    st.rerun()
            else:
                st.download_button(
                    label=f"📥 Download Report",
                    data=st.session_state[pdf_key],
                    file_name=f"ATS_Report_{safe_name}.pdf",
                    mime="application/pdf",
                    key=f"download_{i}_{candidate['filename']}",
                    use_container_width=True,
                )

    # Download all reports as a note
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.info("💡 **Tip:** Each report includes ATS score, score breakdown, skill analysis, and hiring recommendations.")
