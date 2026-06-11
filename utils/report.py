"""
PDF Report Generator Module

Generates downloadable PDF reports for candidate analysis
using fpdf2 (pure Python, no system dependencies).
"""

import io
from typing import Any

from fpdf import FPDF


class ReportPDF(FPDF):
    """Custom PDF class with header and footer branding."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        """Add header to each page."""
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(108, 99, 255)  # Primary purple
        self.cell(0, 10, 'AI Resume Screening System', align='L')
        self.set_font('Helvetica', '', 8)
        self.set_text_color(139, 143, 163)
        self.cell(0, 10, 'Candidate Analysis Report', align='R', new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(108, 99, 255)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        """Add footer with page number."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(139, 143, 163)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, title: str):
        """Add a styled section title."""
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(108, 99, 255)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(60, 60, 80)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def key_value(self, key: str, value: str):
        """Add a key-value pair line."""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(70, 70, 90)
        self.cell(55, 7, f'{key}:', align='L')
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 60)
        self.cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")

    def body_text(self, text: str):
        """Add body text with word wrap."""
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 60)
        # Sanitize text - replace problematic unicode characters
        safe_text = _sanitize_text(text)
        self.multi_cell(0, 6, safe_text)
        self.ln(2)

    def skill_tag(self, skill: str, color: tuple = (108, 99, 255)):
        """Add a styled skill tag inline."""
        self.set_font('Helvetica', '', 9)
        w = self.get_string_width(skill) + 6
        # Check if we need to wrap to next line
        if self.get_x() + w > 195:
            self.ln(7)
        self.set_fill_color(*color)
        self.set_text_color(255, 255, 255)
        self.cell(w, 6, skill, fill=True, new_x="RIGHT", new_y="TOP")
        self.cell(2, 6, '')  # Spacing


def _sanitize_text(text: str) -> str:
    """Replace non-latin1 characters for fpdf2 compatibility."""
    replacements = {
        '\u2019': "'", '\u2018': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-',
        '\u2022': '*', '\u25cf': '*',
        '\u2026': '...', '\u00a0': ' ',
        '\u2192': '->', '\u2190': '<-',
        '\u2713': '[Y]', '\u2717': '[N]',
        '\u2605': '*', '\u2606': '*',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Remove any remaining non-latin1 characters
    return text.encode('latin-1', errors='replace').decode('latin-1')


def generate_pdf_report(candidate: dict[str, Any], rank: int = 0) -> bytes:
    """
    Generate a PDF report for a single candidate.

    Args:
        candidate: Dictionary with candidate analysis data:
            - filename (str)
            - ats_score (float)
            - semantic_score (float)
            - skill_match_pct (float)
            - experience_score (float)
            - education_score (float)
            - matched_skills (set/list)
            - missing_skills (set/list)
            - extra_skills (set/list)
            - label (str)
        rank: Candidate's rank (1-indexed).

    Returns:
        PDF file as bytes, ready for st.download_button.
    """
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    filename = candidate.get('filename', 'Unknown')
    ats_score = candidate.get('ats_score', 0)
    semantic_score = candidate.get('semantic_score', 0)
    skill_match_pct = candidate.get('skill_match_pct', 0)
    exp_score = candidate.get('experience_score', 0)
    edu_score = candidate.get('education_score', 0)
    matched = sorted(candidate.get('matched_skills', []))
    missing = sorted(candidate.get('missing_skills', []))
    extra = sorted(candidate.get('extra_skills', []))
    label = candidate.get('label', 'N/A')

    # ─── Candidate Info ──────────────────────────────────
    pdf.section_title('Candidate Information')
    pdf.key_value('File Name', filename)
    if rank > 0:
        pdf.key_value('Rank', f'#{rank}')
    pdf.key_value('Overall Assessment', label)
    pdf.ln(3)

    # ─── ATS Score ───────────────────────────────────────
    pdf.section_title('ATS Score Summary')
    pdf.set_font('Helvetica', 'B', 28)
    if ats_score >= 85:
        pdf.set_text_color(0, 230, 118)
    elif ats_score >= 70:
        pdf.set_text_color(68, 138, 255)
    elif ats_score >= 50:
        pdf.set_text_color(255, 193, 7)
    else:
        pdf.set_text_color(255, 82, 82)
    pdf.cell(0, 15, f'{ats_score}/100', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Score breakdown table
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(108, 99, 255)
    pdf.cell(70, 8, 'Component', fill=True, align='C')
    pdf.cell(40, 8, 'Score', fill=True, align='C')
    pdf.cell(30, 8, 'Weight', fill=True, align='C')
    pdf.cell(40, 8, 'Weighted', fill=True, align='C', new_x="LMARGIN", new_y="NEXT")

    rows = [
        ('Semantic Similarity', semantic_score, 40),
        ('Skill Match', skill_match_pct, 30),
        ('Experience Indicators', exp_score, 20),
        ('Education Indicators', edu_score, 10),
    ]

    for i, (comp, score, weight) in enumerate(rows):
        bg = (240, 240, 250) if i % 2 == 0 else (250, 250, 255)
        pdf.set_fill_color(*bg)
        pdf.set_text_color(40, 40, 60)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(70, 7, comp, fill=True, align='L')
        pdf.cell(40, 7, f'{score}%', fill=True, align='C')
        pdf.cell(30, 7, f'{weight}%', fill=True, align='C')
        pdf.cell(40, 7, f'{score * weight / 100:.1f}', fill=True, align='C',
                 new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)

    # ─── Matched Skills ──────────────────────────────────
    if matched:
        pdf.section_title(f'Matched Skills ({len(matched)})')
        for skill in matched:
            pdf.skill_tag(skill, color=(0, 212, 170))
        pdf.ln(10)

    # ─── Missing Skills ──────────────────────────────────
    if missing:
        pdf.section_title(f'Missing Skills ({len(missing)})')
        for skill in missing:
            pdf.skill_tag(skill, color=(255, 82, 82))
        pdf.ln(10)

    # ─── Extra Skills ────────────────────────────────────
    if extra:
        pdf.section_title(f'Additional Skills ({len(extra)})')
        for skill in extra:
            pdf.skill_tag(skill, color=(108, 99, 255))
        pdf.ln(10)

    # ─── Recommendations ─────────────────────────────────
    pdf.section_title('Recommendations')
    if ats_score >= 85:
        pdf.body_text(
            'STRONG RECOMMEND FOR INTERVIEW. This candidate is highly aligned '
            'with the job requirements and should be prioritized for the next round.'
        )
    elif ats_score >= 70:
        pdf.body_text(
            'RECOMMEND FOR INTERVIEW. This candidate shows solid potential. '
            'Consider discussing the identified skill gaps during the interview '
            'to assess fit and growth potential.'
        )
    elif ats_score >= 50:
        pdf.body_text(
            'CONDITIONAL RECOMMENDATION. This candidate may be suitable if '
            'the skill gaps can be addressed. Consider for the role if the '
            'candidate pool is limited or if training resources are available.'
        )
    else:
        pdf.body_text(
            'NOT RECOMMENDED AT THIS TIME. The candidate\'s profile does not '
            'sufficiently match the current role requirements. May be reconsidered '
            'for alternative positions or future openings.'
        )

    if missing:
        pdf.ln(2)
        pdf.body_text('Areas for improvement:')
        for skill in missing[:8]:
            pdf.body_text(f'  - Develop proficiency in {skill}')

    # Output — convert bytearray to bytes for Streamlit compatibility
    return bytes(pdf.output())
