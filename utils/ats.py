"""
ATS Scoring Module

Computes a composite ATS (Applicant Tracking System) score based on:
- 40% Semantic Similarity
- 30% Skill Match Percentage
- 20% Experience Keywords Score
- 10% Education Keywords Score
"""

import re
from typing import Tuple


# Experience-related keywords and patterns
EXPERIENCE_PATTERNS = [
    # Years of experience (extracting digits)
    r'\b(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)\b',
    r'\b(?:experience|exp)\s*(?:of\s+)?(\d+)\+?\s*(?:years?|yrs?)\b',
    r'\b(\d+)\+?\s*months?\s*(?:of\s+)?(?:experience|exp)\b',
]

EXPERIENCE_KEYWORDS = [
    # Action verbs (strong indicators of professional experience)
    'developed', 'designed', 'implemented', 'managed', 'led', 'built',
    'created', 'architected', 'deployed', 'optimized', 'maintained',
    'automated', 'integrated', 'delivered', 'launched', 'spearheaded',
    'engineered', 'established', 'streamlined', 'supervised',
    'collaborated', 'mentored', 'coordinated', 'executed', 'resolved',
    'analyzed', 'improved', 'reduced', 'increased', 'scaled',
    'migrated', 'refactored', 'contributed', 'presented', 'published',
    # Job titles
    'senior', 'junior', 'lead', 'principal', 'staff', 'intern',
    'manager', 'director', 'engineer', 'developer', 'architect',
    'analyst', 'consultant', 'specialist', 'administrator',
    # Professional terms
    'team', 'project', 'stakeholder', 'client', 'production',
    'cross-functional', 'enterprise', 'revenue', 'kpi', 'roi',
    'sprint', 'milestone', 'deadline', 'budget',
]

# Education-related keywords
EDUCATION_KEYWORDS = [
    # Degree levels
    'bachelor', 'bachelors', "bachelor's", 'b.s.', 'b.sc', 'b.tech', 'btech',
    'b.e.', 'b.a.', 'bs', 'ba',
    'master', 'masters', "master's", 'm.s.', 'm.sc', 'm.tech', 'mtech',
    'm.e.', 'm.a.', 'ms', 'ma', 'mba', 'm.b.a.',
    'ph.d', 'phd', 'doctorate', 'doctoral',
    'associate', 'diploma', 'certificate', 'certification',
    # Academic terms
    'university', 'college', 'institute', 'school',
    'degree', 'gpa', 'cgpa', 'honors', 'cum laude',
    'computer science', 'information technology', 'software engineering',
    'data science', 'electrical engineering', 'mathematics',
    'statistics', 'physics', 'economics', 'business administration',
    # Certifications
    'certified', 'certification', 'aws certified', 'azure certified',
    'google certified', 'pmp', 'scrum master', 'cissp', 'comptia',
    'professional certificate', 'nanodegree', 'bootcamp',
    'coursera', 'udemy', 'edx', 'udacity',
]


def get_experience_score(text: str) -> float:
    """
    Compute an experience relevance score from text.

    Scores based on presence of experience patterns, action verbs,
    job titles, and professional terminology.

    Args:
        text: Cleaned resume text.

    Returns:
        Score from 0 to 100.
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    score = 0.0
    max_score = 100.0

    # Check for years-of-experience patterns
    total_years = 0.0
    for pattern in EXPERIENCE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            try:
                val = float(match)
                if 'month' in pattern:
                    total_years += val / 12.0
                else:
                    total_years += val
            except ValueError:
                pass
                
    # Add to score based on total years (up to 5 years gives max bonus of 40 points)
    if total_years > 0:
        score += min(40.0, total_years * 8.0)

    # Check for experience keywords
    keyword_hits = 0
    for keyword in EXPERIENCE_KEYWORDS:
        # Count occurrences (capped per keyword to avoid repetition gaming)
        count = min(text_lower.count(keyword), 3)
        keyword_hits += count

    # Scale keyword hits
    if keyword_hits > 0:
        # Max 60 points from keywords (40 from years + 60 from keywords = 100)
        score += min(60, keyword_hits * 3)

    return min(max_score, round(score, 1))


def get_education_score(text: str) -> float:
    """
    Compute an education relevance score from text.

    Scores based on presence of degree levels, academic institutions,
    certifications, and educational terminology.

    Args:
        text: Cleaned resume text.

    Returns:
        Score from 0 to 100.
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    score = 0.0
    max_score = 100.0

    keyword_hits = 0

    # Check for highest degree level only (mutually exclusive tiers)
    phd_keywords = ('ph.d', 'phd', 'doctorate', 'doctoral')
    masters_keywords = ('master', 'masters', "master's", 'm.s.', 'm.sc', 'mba', 'm.b.a.', 'm.tech', 'mtech')
    bachelors_keywords = ('bachelor', 'bachelors', "bachelor's", 'b.s.', 'b.sc', 'b.tech', 'btech', 'b.e.', 'b.a.')

    if any(k in text_lower for k in phd_keywords):
        score += 30
    elif any(k in text_lower for k in masters_keywords):
        score += 22
    elif any(k in text_lower for k in bachelors_keywords):
        score += 15

    # Check for certifications (additive, capped)
    cert_keywords = ('certified', 'certification', 'aws certified', 'azure certified',
                     'google certified', 'pmp', 'scrum master', 'cissp', 'comptia')
    cert_count = sum(1 for k in cert_keywords if k in text_lower)
    score += min(15, cert_count * 5)

    # Check for general academic/education keywords
    general_keywords = (
        'university', 'college', 'institute', 'school',
        'degree', 'gpa', 'cgpa', 'honors', 'cum laude',
        'computer science', 'information technology', 'software engineering',
        'data science', 'electrical engineering', 'mathematics',
        'statistics', 'coursera', 'udemy', 'edx', 'udacity',
        'nanodegree', 'bootcamp', 'professional certificate',
    )
    for kw in general_keywords:
        if kw in text_lower:
            keyword_hits += 1

    # General keyword presence (capped at 30 points)
    score += min(30, keyword_hits * 4)

    return min(max_score, round(score, 1))


def compute_ats_score(
    semantic_score: float,
    skill_match_pct: float,
    experience_score: float,
    education_score: float,
) -> float:
    """
    Compute the composite ATS score.

    Formula:
        ATS Score = 0.4 * semantic + 0.3 * skill_match + 0.2 * experience + 0.1 * education

    All inputs should be on a 0-100 scale.

    Args:
        semantic_score: Semantic similarity score (0-100).
        skill_match_pct: Percentage of JD skills matched (0-100).
        experience_score: Experience keyword score (0-100).
        education_score: Education keyword score (0-100).

    Returns:
        Composite ATS score (0-100).
    """
    score = (
        0.40 * semantic_score +
        0.30 * skill_match_pct +
        0.20 * experience_score +
        0.10 * education_score
    )
    return round(min(100.0, max(0.0, score)), 1)


def get_ats_label(score: float) -> Tuple[str, str]:
    """
    Get a human-readable label and emoji for an ATS score.

    Args:
        score: ATS score (0-100).

    Returns:
        Tuple of (label, color_hex).
    """
    if score >= 85:
        return "Excellent Match", "#00E676"
    elif score >= 70:
        return "Good Match", "#448AFF"
    elif score >= 50:
        return "Average Match", "#FFC107"
    else:
        return "Weak Match", "#FF5252"
