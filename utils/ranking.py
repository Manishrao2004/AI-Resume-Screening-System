"""
Candidate Ranking Module

Ranks multiple candidates by ATS score and generates
AI-style recruiter summaries for each candidate.
"""

import pandas as pd
from typing import Any


def rank_candidates(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Rank candidates by ATS score in descending order.

    Args:
        candidates: List of candidate dicts, each containing at minimum:
            - filename (str)
            - ats_score (float)
            - semantic_score (float)
            - skill_match_pct (float)
            - experience_score (float)
            - education_score (float)

    Returns:
        DataFrame with columns: Rank, Candidate, ATS Score, Semantic Score,
        Skill Match %, Experience Score, Education Score, Label.
    """
    if not candidates:
        return pd.DataFrame()

    df = pd.DataFrame(candidates)

    # Sort by ATS score descending
    df = df.sort_values('ats_score', ascending=False).reset_index(drop=True)

    # Add rank column (1-indexed)
    df['rank'] = range(1, len(df) + 1)

    # Reorder columns for display
    display_columns = ['rank', 'filename', 'ats_score', 'semantic_score',
                       'skill_match_pct', 'experience_score', 'education_score', 'label']

    available_cols = [c for c in display_columns if c in df.columns]
    df = df[available_cols]

    return df


def generate_summary(candidate: dict[str, Any]) -> str:
    """
    Generate an AI-style recruiter summary for a candidate.

    Produces a professional summary covering:
    - Overall assessment
    - Strong areas (matched skills)
    - Weak areas (missing skills)
    - Score breakdown
    - Recommendation

    Args:
        candidate: Dictionary with candidate analysis data:
            - filename (str)
            - ats_score (float)
            - semantic_score (float)
            - matched_skills (set or list)
            - missing_skills (set or list)
            - extra_skills (set or list)
            - experience_score (float)
            - education_score (float)
            - label (str)

    Returns:
        Formatted summary string.
    """
    filename = candidate.get('filename', 'Unknown')
    ats_score = candidate.get('ats_score', 0)
    semantic_score = candidate.get('semantic_score', 0)
    matched = list(candidate.get('matched_skills', []))
    missing = list(candidate.get('missing_skills', []))
    extra = list(candidate.get('extra_skills', []))
    exp_score = candidate.get('experience_score', 0)
    edu_score = candidate.get('education_score', 0)
    label = candidate.get('label', 'N/A')

    # Build summary sections
    sections = []

    # Header
    sections.append(f"## 📋 Candidate Analysis: {filename}")
    sections.append(f"**Overall ATS Score: {ats_score}/100 — {label}**")
    sections.append("")

    # Overview paragraph
    if ats_score >= 85:
        overview = (
            f"This candidate demonstrates an **excellent** alignment with the job requirements. "
            f"The semantic analysis shows a {semantic_score}% match with the job description, "
            f"indicating strong relevance in both technical skills and domain experience."
        )
    elif ats_score >= 70:
        overview = (
            f"This candidate shows a **good** fit for the role. "
            f"With a {semantic_score}% semantic match, the candidate's background "
            f"aligns well with the core requirements, though there are some skill gaps to address."
        )
    elif ats_score >= 50:
        overview = (
            f"This candidate presents an **average** match for the position. "
            f"The {semantic_score}% semantic similarity suggests partial alignment. "
            f"Several key skills may need to be developed or verified during the interview."
        )
    else:
        overview = (
            f"This candidate shows a **below-average** match for the role. "
            f"The {semantic_score}% semantic similarity indicates limited alignment "
            f"with the job requirements. Significant skill gaps have been identified."
        )

    sections.append(overview)
    sections.append("")

    # Score breakdown
    sections.append("### 📊 Score Breakdown")
    sections.append(f"| Component | Score | Weight |")
    sections.append(f"|:---|:---:|:---:|")
    sections.append(f"| Semantic Similarity | {semantic_score}% | 40% |")
    sections.append(f"| Skill Match | {candidate.get('skill_match_pct', 0)}% | 30% |")
    sections.append(f"| Experience Indicators | {exp_score}% | 20% |")
    sections.append(f"| Education Indicators | {edu_score}% | 10% |")
    sections.append("")

    # Strong areas
    if matched:
        sections.append("### ✅ Strong Areas")
        skill_list = ", ".join(sorted(matched)[:15])
        sections.append(f"**Matched Skills:** {skill_list}")
        sections.append(f"")
        sections.append(f"The candidate possesses {len(matched)} out of the required skills, "
                       f"demonstrating competency in key areas needed for the role.")
        sections.append("")

    # Extra skills (bonus)
    if extra:
        sections.append("### 🌟 Additional Skills")
        extra_list = ", ".join(sorted(extra)[:10])
        sections.append(f"**Beyond Requirements:** {extra_list}")
        sections.append("")
        sections.append(f"The candidate brings {len(extra)} additional skill(s) "
                       f"beyond the stated requirements, which may add value to the team.")
        sections.append("")

    # Weak areas
    if missing:
        sections.append("### ⚠️ Areas for Improvement")
        missing_list = ", ".join(sorted(missing))
        sections.append(f"**Missing Skills:** {missing_list}")
        sections.append("")

        # Generate recommendations
        if len(missing) <= 3:
            sections.append(
                f"The candidate is missing {len(missing)} required skill(s). "
                f"These gaps are relatively minor and could potentially be bridged "
                f"through on-the-job training or short-term upskilling."
            )
        elif len(missing) <= 6:
            sections.append(
                f"The candidate is missing {len(missing)} required skills. "
                f"While the core competencies may be present, these gaps should "
                f"be discussed during the interview to assess willingness to learn."
            )
        else:
            sections.append(
                f"The candidate is missing {len(missing)} required skills, "
                f"indicating a significant gap between the candidate's profile "
                f"and the job requirements."
            )
        sections.append("")

    # Recommendation
    sections.append("### 🎯 Recommendation")
    if ats_score >= 85:
        sections.append(
            "**Strong Recommend for Interview.** This candidate is highly aligned "
            "with the job requirements and should be prioritized for the next round."
        )
    elif ats_score >= 70:
        sections.append(
            "**Recommend for Interview.** This candidate shows solid potential. "
            "Consider discussing the identified skill gaps during the interview "
            "to assess fit and growth potential."
        )
    elif ats_score >= 50:
        sections.append(
            "**Conditional Recommendation.** This candidate may be suitable if "
            "the skill gaps can be addressed. Consider for the role if the "
            "candidate pool is limited or if training resources are available."
        )
    else:
        sections.append(
            "**Not Recommended at This Time.** The candidate's profile does not "
            "sufficiently match the current role requirements. May be reconsidered "
            "for alternative positions or future openings."
        )

    return "\n".join(sections)


def generate_skill_recommendations(missing_skills: list[str] | set[str]) -> list[str]:
    """
    Generate actionable recommendations based on missing skills.

    Args:
        missing_skills: Skills the candidate is lacking.

    Returns:
        List of recommendation strings.
    """
    missing = list(missing_skills)
    if not missing:
        return ["No skill gaps identified — candidate meets all listed requirements."]

    recommendations = []

    for skill in sorted(missing)[:8]:
        recommendations.append(
            f"• Consider developing proficiency in **{skill}** through "
            f"online courses, projects, or certifications."
        )

    if len(missing) > 8:
        recommendations.append(
            f"• ... and {len(missing) - 8} additional skill(s) to develop."
        )

    return recommendations
