"""
Charts Module

Interactive Plotly visualizations for the ATS dashboard.
All charts use a consistent dark theme with premium aesthetics.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Any

# ─── Theme Constants ─────────────────────────────────────────────────────────

COLORS = {
    'bg': '#0E1117',
    'card_bg': '#1A1D29',
    'text': '#FAFAFA',
    'text_muted': '#8B8FA3',
    'primary': '#6C63FF',
    'secondary': '#00D4AA',
    'accent': '#FF6B6B',
    'excellent': '#00E676',
    'good': '#448AFF',
    'average': '#FFC107',
    'weak': '#FF5252',
    'gradient': ['#6C63FF', '#00D4AA', '#448AFF', '#FF6B6B', '#FFC107',
                 '#E040FB', '#00BCD4', '#FF9800', '#8BC34A', '#FF5722'],
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color=COLORS['text']),
    margin=dict(l=20, r=20, t=40, b=20),
)


def create_ats_gauge(score: float, label: str) -> go.Figure:
    """
    Create a gauge chart displaying the ATS score.

    Args:
        score: ATS score (0-100).
        label: Score label (e.g., 'Excellent Match').

    Returns:
        Plotly Figure object.
    """
    # Determine color based on score
    if score >= 85:
        bar_color = COLORS['excellent']
    elif score >= 70:
        bar_color = COLORS['good']
    elif score >= 50:
        bar_color = COLORS['average']
    else:
        bar_color = COLORS['weak']

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number=dict(
            suffix="%",
            font=dict(size=42, color=COLORS['text'], family='Inter, sans-serif'),
        ),
        title=dict(
            text=f"<b>{label}</b>",
            font=dict(size=16, color=COLORS['text_muted']),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor=COLORS['text_muted'],
                tickfont=dict(size=11, color=COLORS['text_muted']),
                dtick=25,
            ),
            bar=dict(color=bar_color, thickness=0.75),
            bgcolor=COLORS['card_bg'],
            borderwidth=0,
            steps=[
                dict(range=[0, 50], color='rgba(255, 82, 82, 0.1)'),
                dict(range=[50, 70], color='rgba(255, 193, 7, 0.1)'),
                dict(range=[70, 85], color='rgba(68, 138, 255, 0.1)'),
                dict(range=[85, 100], color='rgba(0, 230, 118, 0.1)'),
            ],
            threshold=dict(
                line=dict(color=COLORS['text'], width=2),
                thickness=0.8,
                value=score,
            ),
        ),
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=280,
    )

    return fig


def create_skill_pie(matched: int, missing: int) -> go.Figure:
    """
    Create a donut chart showing skill coverage.

    Args:
        matched: Number of matched skills.
        missing: Number of missing skills.

    Returns:
        Plotly Figure object.
    """
    if matched == 0 and missing == 0:
        # Empty state
        fig = go.Figure()
        fig.add_annotation(
            text="No skills data",
            font=dict(size=16, color=COLORS['text_muted']),
            showarrow=False,
        )
        fig.update_layout(**LAYOUT_DEFAULTS, height=300)
        return fig

    labels = ['Matched Skills', 'Missing Skills']
    values = [matched, missing]
    colors = [COLORS['secondary'], COLORS['accent']]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(
            colors=colors,
            line=dict(color=COLORS['bg'], width=3),
        ),
        textinfo='label+percent',
        textfont=dict(size=12, color=COLORS['text']),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
    )])

    fig.add_annotation(
        text=f"<b>{matched}</b><br><span style='font-size:11px;color:{COLORS['text_muted']}'>of {matched + missing}</span>",
        font=dict(size=22, color=COLORS['text'], family='Inter, sans-serif'),
        showarrow=False,
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=300,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=11, color=COLORS['text_muted']),
        ),
    )

    return fig


def create_ranking_bar(df: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal bar chart of candidate rankings.

    Args:
        df: DataFrame from rank_candidates() with 'filename' and 'ats_score' columns.

    Returns:
        Plotly Figure object.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No candidates to display", showarrow=False)
        fig.update_layout(**LAYOUT_DEFAULTS, height=400)
        return fig

    # Sort by score for display (lowest at top so highest appears at bottom of horizontal bar)
    sorted_df = df.sort_values('ats_score', ascending=True).reset_index(drop=True)

    # Color each bar based on score
    colors = []
    for score in sorted_df['ats_score']:
        if score >= 85:
            colors.append(COLORS['excellent'])
        elif score >= 70:
            colors.append(COLORS['good'])
        elif score >= 50:
            colors.append(COLORS['average'])
        else:
            colors.append(COLORS['weak'])

    # Truncate filenames for display
    display_names = [
        name[:25] + '...' if len(name) > 28 else name
        for name in sorted_df['filename']
    ]

    fig = go.Figure(go.Bar(
        x=sorted_df['ats_score'],
        y=display_names,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0),
            cornerradius=6,
        ),
        text=[f"{s}%" for s in sorted_df['ats_score']],
        textposition='outside',
        textfont=dict(size=12, color=COLORS['text'], family='Inter, sans-serif'),
        hovertemplate='<b>%{y}</b><br>ATS Score: %{x}%<extra></extra>',
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=max(300, len(sorted_df) * 50 + 100),
        xaxis=dict(
            range=[0, 110],
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            zeroline=False,
            title=dict(text='ATS Score (%)', font=dict(size=12, color=COLORS['text_muted'])),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12),
        ),
        title=dict(
            text='<b>Candidate Rankings</b>',
            font=dict(size=16, color=COLORS['text']),
            x=0.5,
        ),
    )

    return fig


def create_comparison_radar(candidates: list[dict[str, Any]]) -> go.Figure:
    """
    Create a radar chart comparing candidates across scoring dimensions.

    Args:
        candidates: List of candidate dicts with scoring data.
            Limited to top 5 for readability.

    Returns:
        Plotly Figure object.
    """
    if not candidates:
        fig = go.Figure()
        fig.add_annotation(text="No candidates to compare", showarrow=False)
        fig.update_layout(**LAYOUT_DEFAULTS, height=400)
        return fig

    # Limit to top 5 candidates
    top = sorted(candidates, key=lambda c: c.get('ats_score', 0), reverse=True)[:5]

    categories = ['Semantic\nMatch', 'Skill\nMatch', 'Experience', 'Education', 'ATS\nScore']

    fig = go.Figure()

    for i, candidate in enumerate(top):
        values = [
            candidate.get('semantic_score', 0),
            candidate.get('skill_match_pct', 0),
            candidate.get('experience_score', 0),
            candidate.get('education_score', 0),
            candidate.get('ats_score', 0),
        ]
        # Close the polygon
        values.append(values[0])

        name = candidate.get('filename', f'Candidate {i+1}')
        if len(name) > 20:
            name = name[:17] + '...'

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=f'rgba({_hex_to_rgb(COLORS["gradient"][i])}, 0.15)',
            line=dict(color=COLORS['gradient'][i], width=2),
            name=name,
            hovertemplate='%{theta}: %{r}%<extra>' + name + '</extra>',
        ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=450,
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                tickfont=dict(size=10, color=COLORS['text_muted']),
                gridcolor='rgba(255,255,255,0.08)',
                linecolor='rgba(255,255,255,0.08)',
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=COLORS['text']),
                gridcolor='rgba(255,255,255,0.08)',
                linecolor='rgba(255,255,255,0.08)',
            ),
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=11, color=COLORS['text_muted']),
            bgcolor='rgba(0,0,0,0)',
        ),
        title=dict(
            text='<b>Candidate Comparison</b>',
            font=dict(size=16, color=COLORS['text']),
            x=0.5,
        ),
    )

    return fig


def create_score_breakdown_bar(candidate: dict[str, Any]) -> go.Figure:
    """
    Create a grouped bar chart showing the score breakdown for a single candidate.

    Args:
        candidate: Candidate dict with scoring data.

    Returns:
        Plotly Figure object.
    """
    categories = ['Semantic Match', 'Skill Match', 'Experience', 'Education']
    scores = [
        candidate.get('semantic_score', 0),
        candidate.get('skill_match_pct', 0),
        candidate.get('experience_score', 0),
        candidate.get('education_score', 0),
    ]
    weights = [40, 30, 20, 10]
    weighted = [s * w / 100 for s, w in zip(scores, weights)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Raw Score',
        x=categories,
        y=scores,
        marker=dict(color=COLORS['primary'], cornerradius=6),
        text=[f"{s}%" for s in scores],
        textposition='outside',
        textfont=dict(size=11, color=COLORS['text']),
    ))

    fig.add_trace(go.Bar(
        name='Weighted Contribution',
        x=categories,
        y=weighted,
        marker=dict(color=COLORS['secondary'], cornerradius=6),
        text=[f"{w:.1f}" for w in weighted],
        textposition='outside',
        textfont=dict(size=11, color=COLORS['text']),
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=350,
        barmode='group',
        xaxis=dict(
            tickfont=dict(size=11),
            showgrid=False,
        ),
        yaxis=dict(
            range=[0, 115],
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            zeroline=False,
            title=dict(text='Score', font=dict(size=12, color=COLORS['text_muted'])),
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=11, color=COLORS['text_muted']),
        ),
        title=dict(
            text='<b>Score Breakdown</b>',
            font=dict(size=16, color=COLORS['text']),
            x=0.5,
        ),
    )

    return fig


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to comma-separated RGB string."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r},{g},{b}"
