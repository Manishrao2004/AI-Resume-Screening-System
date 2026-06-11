"""
Skill Extraction Module

Loads a skills database and extracts skills from text using
regex word-boundary matching with alias resolution.
"""

import re
import os
from typing import Set

import pandas as pd
import streamlit as st


@st.cache_data
def load_skills_database() -> pd.DataFrame:
    """
    Load the skills database CSV and return as a DataFrame.

    The CSV has columns: skill, category, aliases
    Aliases are pipe-delimited strings of alternative names.

    Returns:
        DataFrame with skill data.
    """
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "skills_database.csv")
    df = pd.read_csv(csv_path)
    df['skill'] = df['skill'].str.strip()
    df['category'] = df['category'].str.strip()
    df['aliases'] = df['aliases'].fillna('').str.strip()
    return df


def _build_skill_patterns(skills_db: pd.DataFrame) -> list[dict]:
    """
    Build compiled regex patterns for each skill and its aliases.

    Uses word boundary matching to avoid false positives
    (e.g., matching 'R' inside 'React').

    Args:
        skills_db: DataFrame from load_skills_database().

    Returns:
        List of dicts with keys: skill, category, pattern (compiled regex).
    """
    patterns = []

    for _, row in skills_db.iterrows():
        skill_name = row['skill']
        category = row['category']
        aliases_str = row['aliases']

        # Collect all names to match (skill name + aliases)
        names = [skill_name]
        if aliases_str:
            names.extend([a.strip() for a in aliases_str.split('|') if a.strip()])

        # Build alternation pattern, sorted by length (longest first to avoid partial matches)
        names = sorted(set(names), key=len, reverse=True)
        escaped_names = [re.escape(name) for name in names]
        alternation = '|'.join(escaped_names)

        # Word boundary pattern (case-insensitive)
        # Use (?<!\w) and (?!\w) instead of \b for special chars like C++, C#
        pattern = re.compile(
            r'(?<![a-zA-Z0-9_])(?:' + alternation + r')(?![a-zA-Z0-9_])',
            re.IGNORECASE
        )

        patterns.append({
            'skill': skill_name,
            'category': category,
            'pattern': pattern,
        })

    return patterns


# Module-level cache for compiled patterns
_cached_patterns: list[dict] | None = None


def _get_patterns(skills_db: pd.DataFrame) -> list[dict]:
    """Get or build cached skill patterns."""
    global _cached_patterns
    if _cached_patterns is None:
        _cached_patterns = _build_skill_patterns(skills_db)
    return _cached_patterns


def extract_skills(text: str, skills_db: pd.DataFrame) -> list[str]:
    """
    Extract skills from text using regex word-boundary matching.

    Args:
        text: Cleaned text to search for skills.
        skills_db: DataFrame from load_skills_database().

    Returns:
        List of canonical skill names found in the text, ordered by first appearance.
    """
    if not text:
        return []

    patterns = _get_patterns(skills_db)
    found_skills = []

    for entry in patterns:
        match = entry['pattern'].search(text)
        if match:
            found_skills.append({
                'skill': entry['skill'],
                'pos': match.start()
            })

    # Sort by position to get order of appearance
    found_skills.sort(key=lambda x: x['pos'])
    
    # Return unique skills in order
    result = []
    seen = set()
    for item in found_skills:
        if item['skill'] not in seen:
            seen.add(item['skill'])
            result.append(item['skill'])
            
    return result


def extract_skills_with_categories(text: str, skills_db: pd.DataFrame) -> dict[str, list[str]]:
    """
    Extract skills from text, grouped by category.

    Args:
        text: Cleaned text to search for skills.
        skills_db: DataFrame from load_skills_database().

    Returns:
        Dict mapping category names to lists of skill names.
    """
    if not text:
        return {}

    patterns = _get_patterns(skills_db)
    categorized: dict[str, list[str]] = {}

    for entry in patterns:
        if entry['pattern'].search(text):
            category = entry['category']
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(entry['skill'])

    return categorized


def get_skill_analysis(resume_skills: list[str], jd_skills: list[str]) -> dict:
    """
    Compare resume skills against job description skills with weighting.
    Skills mentioned earlier in the JD are given higher weight (assumed to be core requirements).

    Args:
        resume_skills: List of skills found in the resume.
        jd_skills: List of skills found in the job description.

    Returns:
        Dictionary with:
        - matched: Skills present in both resume and JD
        - missing: Skills in JD but not in resume
        - extra: Skills in resume but not in JD
        - match_percentage: Percentage of JD skills matched (0-100)
    """
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)
    
    matched = resume_set & jd_set
    missing = jd_set - resume_set
    extra = resume_set - jd_set

    match_percentage = 0.0
    if jd_skills:
        # Calculate weighted score (skills early in JD get higher weight)
        total_weight = 0.0
        earned_weight = 0.0
        
        for i, skill in enumerate(jd_skills):
            # Weight decays from 2.0 (first skill) to 1.0 (last skill)
            weight = 2.0 - (i / max(1, len(jd_skills) - 1))
            total_weight += weight
            if skill in matched:
                earned_weight += weight
                
        match_percentage = (earned_weight / total_weight) * 100

    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "match_percentage": round(match_percentage, 1),
    }
