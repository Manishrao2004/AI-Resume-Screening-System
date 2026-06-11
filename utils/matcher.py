"""
Semantic Matching Module

Uses TF-IDF vectorization with n-gram support from scikit-learn
to compute text similarity between resume text and job descriptions.

This approach is lightweight (no PyTorch dependency), fast, and
works reliably on Streamlit Community Cloud without large model
downloads or high memory usage.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Text Preprocessing ─────────────────────────────────────────

_STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'shall',
    'can', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we',
    'our', 'you', 'your', 'he', 'she', 'it', 'they', 'them', 'its',
    'not', 'no', 'nor', 'so', 'if', 'then', 'than', 'too', 'very',
    'just', 'about', 'above', 'after', 'again', 'all', 'also', 'am',
    'as', 'because', 'before', 'between', 'both', 'each', 'few',
    'here', 'how', 'into', 'more', 'most', 'other', 'out', 'over',
    'own', 'same', 'some', 'such', 'there', 'through', 'under',
    'until', 'up', 'what', 'when', 'where', 'which', 'while', 'who',
    'why', 'any', 'only',
}


def _preprocess(text: str) -> str:
    """
    Clean and normalize text for TF-IDF vectorization.

    - Lowercases everything
    - Keeps alphanumeric, '+', '#', '.', '-' (important for tech terms
      like C++, C#, Node.js, scikit-learn)
    - Collapses whitespace
    - Removes generic stopwords while keeping domain-relevant short words

    Args:
        text: Raw input text.

    Returns:
        Cleaned text ready for vectorization.
    """
    if not text:
        return ""

    text = text.lower()

    # Keep characters relevant to technical terms
    text = re.sub(r'[^a-z0-9+#.\-\s]', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    return ' '.join(tokens)


# ── TF-IDF Vectorizer ──────────────────────────────────────────

def _build_vectorizer() -> TfidfVectorizer:
    """
    Build a TF-IDF vectorizer tuned for resume/JD matching.

    Uses (1,2)-grams to capture multi-word skills like
    "machine learning", "data engineering", etc. Sublinear TF
    dampens the effect of repeated terms in long resumes.

    Returns:
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),          # Unigrams + bigrams
        max_features=10000,          # Cap vocabulary for speed
        sublinear_tf=True,           # Apply log normalization
        norm='l2',                   # L2-normalize TF-IDF vectors
        min_df=1,                    # Keep all terms (small corpus)
        max_df=1.0,                  # No upper bound filtering
        dtype=np.float32,            # Memory efficient
    )


# ── Public API (same interface as before) ──────────────────────

def load_model():
    """
    Return the TF-IDF vectorizer factory.

    This replaces the previous SentenceTransformer model loader.
    Returns None because TF-IDF doesn't need a pre-loaded model —
    it's fit on-the-fly per analysis run. Keeping this function
    ensures backward compatibility with app.py.

    Returns:
        None (kept for API compatibility).
    """
    return None


def compute_similarity(resume_text: str, jd_text: str, model=None) -> float:
    """
    Compute text similarity between a resume and a job description
    using TF-IDF vectorization and cosine similarity.

    Args:
        resume_text: Cleaned resume text.
        jd_text: Job description text.
        model: Ignored (kept for API compatibility).

    Returns:
        Similarity score as a float between 0 and 100.
    """
    if not resume_text or not jd_text:
        return 0.0

    try:
        # Truncate very long texts
        max_chars = 15000
        clean_resume = _preprocess(resume_text[:max_chars])
        clean_jd = _preprocess(jd_text[:max_chars])

        if not clean_resume or not clean_jd:
            return 0.0

        # Fit TF-IDF on both texts
        vectorizer = _build_vectorizer()
        tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])

        # Cosine similarity between the two TF-IDF vectors
        sim_score = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2],
        )[0][0]

        # Scale to 0-100
        # Raw TF-IDF cosine similarity for resumes typically falls in 0.05-0.5 range.
        # We apply a scaling curve so results feel meaningful (not all clustered low).
        # Linear scale: map [0, 0.6] → [0, 100], capped at 100.
        scaled = min(100.0, max(0.0, float(sim_score) / 0.6 * 100))

        return round(scaled, 1)

    except Exception:
        return 0.0


def compute_batch_similarity(
    resume_texts: list[str],
    jd_text: str,
    model=None,
) -> list[float]:
    """
    Compute text similarity for multiple resumes against a single JD.

    More efficient than calling compute_similarity() in a loop because
    it fits the vectorizer once on all texts together.

    Args:
        resume_texts: List of cleaned resume texts.
        jd_text: Job description text.
        model: Ignored (kept for API compatibility).

    Returns:
        List of similarity scores (0-100) in the same order as input.
    """
    if not jd_text or not resume_texts:
        return [0.0] * len(resume_texts)

    try:
        max_chars = 15000
        clean_resumes = [_preprocess(t[:max_chars]) if t else "" for t in resume_texts]
        clean_jd = _preprocess(jd_text[:max_chars])

        if not clean_jd:
            return [0.0] * len(resume_texts)

        # Fit on all texts at once (resumes + JD)
        all_texts = clean_resumes + [clean_jd]
        vectorizer = _build_vectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # JD vector is the last one
        jd_vector = tfidf_matrix[-1:]
        resume_vectors = tfidf_matrix[:-1]

        # Compute similarities
        similarities = cosine_similarity(resume_vectors, jd_vector).flatten()

        scores = [
            round(min(100.0, max(0.0, float(s) / 0.6 * 100)), 1)
            for s in similarities
        ]

        return scores

    except Exception:
        return [0.0] * len(resume_texts)
