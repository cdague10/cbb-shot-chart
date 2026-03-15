import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, Arc
import io
import os

# Set page config
st.set_page_config(page_title="CBB Shot Chart Explorer", layout="wide", page_icon="🏀")

DEFAULT_CBB_DATA_URL = "https://github.com/cdague10/cbb-shot-chart/releases/download/cbb-pbp-data/cbb_pbp_shots.parquet"
SHOT_TYPES = {'DunkShot', 'JumpShot', 'LayUpShot', 'TipShot'}
REQUIRED_SHOT_COLUMNS = [
    'game_id', 'clock', 'text', 'home_team', 'away_team', 'team',
    'play_type', 'score', 'miss', 'coord_x', 'coord_y'
]
PREPROCESSED_SHOT_COLUMNS = [
    'game_id', 'home_team', 'away_team', 'team', 'player',
    'coord_x', 'coord_y', 'is_made', 'is_three_point', 'points'
]
SHOT_PLUS_OPTIONAL_COLUMNS = [
    'make_probability_model', 'expected_points_model', 'shot_plus', 'shot_grade',
    'shot_value_added', 'result_plus', 'result_grade'
]

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background:
            radial-gradient(circle at 12% 4%, rgba(255, 107, 53, 0.18), transparent 28%),
            radial-gradient(circle at 88% 92%, rgba(73, 132, 255, 0.16), transparent 30%),
            linear-gradient(165deg, #070b18 0%, #0c1630 55%, #0a1224 100%);
    }
    .main {
        padding: 0rem 1rem;
    }
    .block-container {
        padding-top: 0.75rem !important;
        padding-bottom: 0.5rem !important;
    }
    section[data-testid="stSidebar"] {
        min-width: 300px !important;
        max-width: 300px !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        min-width: 300px !important;
        max-width: 300px !important;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label {
        font-size: 1rem;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 1.2rem;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
    }
    div[data-testid="metric-container"] {
        padding: 0.15rem 0.25rem !important;
        margin-bottom: 0 !important;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.72rem !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.05rem !important;
    }
    .stats-panel-header {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: 0.01em;
        color: #f4f8ff;
        margin: 0.05rem 0 0.35rem 0;
    }
    .stats-section-title {
        font-size: 0.96rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: #e8f0ff;
        margin: 0.3rem 0 0.14rem 0;
    }
    .stats-card {
        border: 1px solid rgba(143, 184, 255, 0.32);
        border-radius: 10px;
        padding: 0.3rem 0.55rem;
        margin-bottom: 0.42rem;
        background: linear-gradient(180deg, rgba(18, 26, 48, 0.78) 0%, rgba(10, 16, 34, 0.88) 100%);
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.22);
    }
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.55rem;
        font-size: 0.9rem;
        line-height: 1.28;
        color: #f4f7ff;
        padding: 0.1rem 0;
    }
    .stat-row span:last-child {
        font-weight: 700;
        color: #ffffff;
    }
    h1 {
        color: #FF8F4A;
        text-shadow: 0 3px 20px rgba(255, 107, 53, 0.28);
        margin-top: 0.1rem !important;
        margin-bottom: 0.35rem !important;
    }
    .creator-credit {
        font-size: 0.82rem;
        color: #c8d7f0;
        opacity: 0.92;
        margin-top: 0.6rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data with caching
def _read_shot_source(source):
    """Read shot data from CSV or Parquet path/URL."""
    source_lower = str(source).lower()
    if source_lower.endswith('.parquet'):
        try:
            df = pd.read_parquet(source, columns=REQUIRED_SHOT_COLUMNS)
        except Exception:
            df = pd.read_parquet(source)
    else:
        chunks = []
        csv_iter = pd.read_csv(
            source,
            on_bad_lines='skip',
            usecols=lambda c: c in REQUIRED_SHOT_COLUMNS,
            chunksize=150000,
            low_memory=False
        )
        for chunk in csv_iter:
            if 'play_type' not in chunk.columns:
                continue
            shot_chunk = chunk[chunk['play_type'].isin(SHOT_TYPES)]
            if not shot_chunk.empty:
                chunks.append(shot_chunk)

        if chunks:
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.DataFrame(columns=REQUIRED_SHOT_COLUMNS)

    for column in REQUIRED_SHOT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    return df


def _coerce_bool_series(series):
    """Handle bool columns robustly across CSV/Parquet/string inputs."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    normalized = series.fillna('').astype(str).str.strip().str.lower()
    return normalized.isin({'1', 'true', 't', 'yes', 'y'})


@st.cache_data
def load_data():
    data_url = os.environ.get('CBB_DATA_URL', '').strip()
    data_file_override = os.environ.get('CBB_DATA_FILE', '').strip()
    default_data_url = os.environ.get('CBB_DEFAULT_DATA_URL', DEFAULT_CBB_DATA_URL).strip()

    source_candidates = []
    if data_url:
        source_candidates.append(('url', data_url))

    # Respect explicit local override before defaults.
    if data_file_override:
        source_candidates.append(('file', data_file_override))

    # Prefer locally-scored Shot+ parquet when present.
    source_candidates.append(('file', 'cbb_pbp_shot_plus.parquet'))
    source_candidates.append(('file', 'cbb_pbp.csv'))

    if default_data_url and not data_url:
        source_candidates.append(('url', default_data_url))

    source_candidates.append(('file', 'filtered_shots.csv'))

    df = None
    data_source_used = None
    load_errors = []

    for source_type, source in source_candidates:
        if source_type == 'file' and not os.path.exists(source):
            continue
        try:
            df = _read_shot_source(source)
            data_source_used = source
            break
        except Exception as exc:
            load_errors.append(f"{source}: {exc}")

    if df is None:
        error_details = "\n".join(load_errors[:3])
        raise FileNotFoundError(
            "No usable data source found. Set CBB_DATA_URL to a hosted CSV/Parquet URL, "
            "or place cbb_pbp_shot_plus.parquet / cbb_pbp.csv / filtered_shots.csv beside this app."
            + (f"\n\nRecent load errors:\n{error_details}" if error_details else "")
        )

    # Fast path for preprocessed full-shot datasets used in deployment.
    if set(PREPROCESSED_SHOT_COLUMNS).issubset(df.columns):
        optional_columns = [c for c in SHOT_PLUS_OPTIONAL_COLUMNS if c in df.columns]
        shots = df[PREPROCESSED_SHOT_COLUMNS + optional_columns].copy()
        shots['coord_x'] = pd.to_numeric(shots['coord_x'], errors='coerce')
        shots['coord_y'] = pd.to_numeric(shots['coord_y'], errors='coerce')
        shots = shots[(shots['coord_x'].notna()) & (shots['coord_y'].notna())]
        shots['is_made'] = _coerce_bool_series(shots['is_made'])
        shots['is_three_point'] = _coerce_bool_series(shots['is_three_point'])
        shots['points'] = pd.to_numeric(shots['points'], errors='coerce').fillna(0)
        shots['player'] = shots['player'].fillna('').astype(str).str.strip()

        for col in ('make_probability_model', 'expected_points_model', 'shot_plus', 'shot_value_added', 'result_plus'):
            if col in shots.columns:
                shots[col] = pd.to_numeric(shots[col], errors='coerce')

        for col in ('shot_grade', 'result_grade'):
            if col in shots.columns:
                shots[col] = shots[col].fillna('').astype(str).str.strip()

        shots.attrs['data_source'] = data_source_used
        shots.attrs['requested_data_url'] = data_url
        shots.attrs['load_errors'] = load_errors
        return shots

    if 'play_type' not in df.columns:
        raise ValueError("Data source is missing required column 'play_type'.")

    # Filter for shots only
    shots = df[df['play_type'].isin(SHOT_TYPES)].copy()
    shots['coord_x'] = pd.to_numeric(shots['coord_x'], errors='coerce')
    shots['coord_y'] = pd.to_numeric(shots['coord_y'], errors='coerce')
    shots = shots[(shots['coord_x'].notna()) & (shots['coord_y'].notna())]

    # Remove duplicate plays
    shots = shots.drop_duplicates(subset=['text', 'game_id', 'clock'], keep='first')

    shot_text = shots['text'].fillna('').str.lower()
    score_nonempty = shots['score'].fillna('').astype(str).str.strip().ne('')
    miss_nonempty  = shots['miss'].fillna('').astype(str).str.strip().ne('')
    text_says_made   = shot_text.str.contains(r'\bmade\b',   regex=True, na=False)
    text_says_missed = shot_text.str.contains(r'\bmissed\b', regex=True, na=False)
    text_says_makes  = shot_text.str.contains(r'\bmakes?\b', regex=True, na=False)
    text_says_misses = shot_text.str.contains(r'\bmisses?\b', regex=True, na=False)

    # A shot is made if score column is filled, OR text says made/makes with no miss signal
    shots['is_made'] = (
        score_nonempty |
        ((text_says_made | text_says_makes) & ~(miss_nonempty | text_says_missed | text_says_misses))
    )
    shots['is_three_point'] = shot_text.str.contains('three point', na=False)

    # Assign points
    shots['points'] = 0
    shots.loc[shots['is_made'] & shots['is_three_point'], 'points'] = 3
    shots.loc[shots['is_made'] & ~shots['is_three_point'], 'points'] = 2

    # Extract player names from score/miss columns
    shots['player'] = shots['score'].fillna('') + shots['miss'].fillna('')
    shots['player'] = shots['player'].str.strip()

    # Fallback: extract player directly from text for rows still blank after backfill
    # Covers verbless ESPN format: "Player Two Point Jump Shot"
    player_blank = shots['player'].eq('')
    if player_blank.any():
        import re

        def _player_from_text(t):
            t = str(t)
            m = re.match(r"^(.+?)\s+(?:made?|makes?|missed?|misses?)\b", t, re.IGNORECASE)
            if m:
                return m.group(1).strip()
            m = re.match(
                r"^(.+?)\s+(?:Two|Three|Free\s+Throw|Jump\s+Shot|Dunk|Lay.?[Uu]p|Tip\s+Shot|Hook\s+Shot)",
                t, re.IGNORECASE
            )
            return m.group(1).strip() if m else ''

        text_players = shots.loc[player_blank, 'text'].apply(_player_from_text)
        shots.loc[player_blank & text_players.ne(''), 'player'] = text_players[text_players.ne('')]

        # Fix is_made for verbless makes (player found, no miss signal)
        verbless_no_miss = (
            player_blank &
            text_players.ne('') &
            ~shot_text.str.contains(r'\bmiss', regex=True, na=False)
        )
        shots.loc[verbless_no_miss, 'is_made'] = True

    shots.attrs['data_source'] = data_source_used
    shots.attrs['requested_data_url'] = data_url
    shots.attrs['load_errors'] = load_errors
    return shots

def draw_court(ax, color='white', overlay_zorder=10):
    """Draw basketball court."""
    hoop_x, hoop_y = 25, 0
    
    # Hoop
    hoop = Circle((hoop_x, hoop_y), 1, color=color, fill=False, linewidth=2, zorder=overlay_zorder)
    backboard = Rectangle((hoop_x - 3, hoop_y - 1.05), 6, 0.15, color=color, zorder=overlay_zorder)
    ax.add_patch(hoop)
    ax.add_patch(backboard)

    # Layup guide arc
    layup_arc = Arc(
        (hoop_x, hoop_y),
        10,
        10,
        angle=0,
        theta1=0,
        theta2=180,
        linewidth=1.8,
        edgecolor=color,
        linestyle='--',
        zorder=overlay_zorder
    )
    ax.add_patch(layup_arc)
    
    # Paint
    paint = Rectangle((19, -4), 12, 19, fill=False, linewidth=2, edgecolor=color, zorder=overlay_zorder)
    ax.add_patch(paint)
    
    # Free throw circle
    ft_circle = Circle((hoop_x, 15), 6, fill=False, linewidth=2, edgecolor=color, zorder=overlay_zorder)
    ax.add_patch(ft_circle)
    
    # 3PT line
    three_pt_radius = 22.61
    y_cutoff = 8.1
    theta_cutoff = np.degrees(np.arcsin((y_cutoff - hoop_y) / three_pt_radius))
    three_pt_arc = Arc(
        (hoop_x, hoop_y),
        2 * three_pt_radius,
        2 * three_pt_radius,
        angle=0,
        theta1=theta_cutoff,
        theta2=180 - theta_cutoff,
        linewidth=2,
        edgecolor=color,
        zorder=overlay_zorder
    )
    ax.add_patch(three_pt_arc)
    
    # Corner 3PT lines
    ax.plot([3.93, 3.93], [-4, 8.15], color=color, linewidth=2, zorder=overlay_zorder)
    ax.plot([46.07, 46.07], [-4, 8.15], color=color, linewidth=2, zorder=overlay_zorder)

def create_shot_chart(filtered_shots, title, chart_type='scatter'):
    """Create shot chart visualization."""
    if len(filtered_shots) == 0:
        st.warning("No shots found for the selected filters.")
        return None
    
    if chart_type == 'scatter':
        # Scatter plot individual shots
        fig, ax = plt.subplots(figsize=(5.2, 5.8))
        fig.subplots_adjust(top=0.90, bottom=0.01, left=0.01, right=0.99)
        ax.set_facecolor('#f0f0f0')
        fig.patch.set_facecolor('white')
        
        # Made shots
        made = filtered_shots[filtered_shots['is_made']]
        ax.scatter(
            made['coord_x'],
            made['coord_y'],
            c='green',
            s=100,
            alpha=0.55,
            edgecolors='darkgreen',
            linewidths=1,
            label=f'Made ({len(made)})',
            zorder=3
        )
        
        # Missed shots
        missed = filtered_shots[~filtered_shots['is_made']]
        ax.scatter(
            missed['coord_x'],
            missed['coord_y'],
            c='red',
            s=80,
            marker='x',
            linewidths=1,
            alpha=0.55,
            label=f'Missed ({len(missed)})',
            zorder=3
        )

        # Keep court markings visible above shot markers
        draw_court(ax, color='black', overlay_zorder=10)
        
        ax.set_xlim(0, 50)
        ax.set_ylim(-3, 33)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=14, fontweight='bold', pad=4)
        ax.legend(fontsize=8, loc='upper left')
        
    else:  # heatmap
        fig, ax = plt.subplots(figsize=(5.2, 5.8))
        fig.subplots_adjust(top=0.90, bottom=0.10, left=0.01, right=0.99)
        ax.set_facecolor('#1a1a1a')
        fig.patch.set_facecolor('#1a1a1a')
        
        hexbin = ax.hexbin(
            filtered_shots['coord_x'],
            filtered_shots['coord_y'],
            C=filtered_shots['points'],
            gridsize=25,
            cmap='turbo',
            mincnt=3,
            alpha=0.9,
            vmin=0,
            vmax=2.5,
            edgecolors='none',
            reduce_C_function=np.mean
        )
        
        draw_court(ax, color='white', overlay_zorder=10)
        
        cbar = plt.colorbar(hexbin, ax=ax, orientation='horizontal', pad=0.025)
        cbar.set_label('Expected Points', rotation=0, labelpad=0, fontsize=14, color='white')
        cbar.ax.tick_params(colors='white')
        
        ax.set_xlim(0, 50)
        ax.set_ylim(-3, 30)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=4)
    
    return fig

def render_stats_section(title, rows):
    """Render compact stats rows that fit in the right panel without scrolling."""
    st.markdown(f'<div class="stats-section-title">{title}</div>', unsafe_allow_html=True)
    rows_html = "".join(
        f'<div class="stat-row"><span>{label}</span><span>{value}</span></div>'
        for label, value in rows
    )
    st.markdown(f'<div class="stats-card">{rows_html}</div>', unsafe_allow_html=True)


def calculate_expected_point_stats(filtered_shots):
    """Calculate expected points by zone regardless of chart type."""
    hoop_x, hoop_y = 25, 0
    zone_df = filtered_shots.copy()
    zone_df['dist_from_hoop'] = np.sqrt(
        (zone_df['coord_x'] - hoop_x) ** 2 +
        (zone_df['coord_y'] - hoop_y) ** 2
    )

    overall_avg = zone_df['points'].mean()
    two_pt_shots = zone_df[~zone_df['is_three_point']]
    three_pt_shots = zone_df[zone_df['is_three_point']]

    layup_shots = two_pt_shots[two_pt_shots['dist_from_hoop'] <= 5]
    paint_shots = two_pt_shots[
        (two_pt_shots['coord_x'] >= 19) & (two_pt_shots['coord_x'] <= 31) &
        (two_pt_shots['coord_y'] >= -4) & (two_pt_shots['coord_y'] <= 15) &
        (two_pt_shots['dist_from_hoop'] > 5)
    ]
    midrange_shots = two_pt_shots[
        (two_pt_shots['dist_from_hoop'] > 5) &
        ~((two_pt_shots['coord_x'] >= 19) & (two_pt_shots['coord_x'] <= 31) &
          (two_pt_shots['coord_y'] >= -4) & (two_pt_shots['coord_y'] <= 15))
    ]

    return {
        'overall': overall_avg if pd.notna(overall_avg) else 0,
        'layup': layup_shots['points'].mean() if len(layup_shots) > 0 else 0,
        'paint': paint_shots['points'].mean() if len(paint_shots) > 0 else 0,
        'midrange': midrange_shots['points'].mean() if len(midrange_shots) > 0 else 0,
        'three_pt': three_pt_shots['points'].mean() if len(three_pt_shots) > 0 else 0,
        'layup_count': len(layup_shots),
        'paint_count': len(paint_shots),
        'midrange_count': len(midrange_shots),
        'three_pt_count': len(three_pt_shots)
    }

# Main app
st.title("College Basketball Shot Charts")
#st.markdown("Filter by team, player, or game to visualize shot charts and performance")

# Load data
with st.spinner("Loading shot data..."):
    try:
        shots = load_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

# Sidebar filters
st.sidebar.header("Shooting Filters")
st.sidebar.markdown("<style>div.row-widget.stRadio > div{font-size: 30px;}</style>", unsafe_allow_html=True)
st.sidebar.markdown(f"✅ Loaded {len(shots):,} shots from {shots['game_id'].nunique():,} games")
loaded_source = str(shots.attrs.get('data_source', 'Unknown'))
requested_data_url = str(shots.attrs.get('requested_data_url', '')).strip()
load_errors = shots.attrs.get('load_errors', [])


if requested_data_url and loaded_source != requested_data_url:
    st.sidebar.warning("CBB_DATA_URL failed, app is using fallback data source.")
    if load_errors:
        st.sidebar.caption(f"URL load error: {load_errors[0]}")

#if 'shot_plus' in shots.columns:
#    st.sidebar.success("Shot+ columns detected.")
#else:
#    st.sidebar.info("No Shot+ columns in loaded data.")

filter_modes = ["Team", "Player", "Game", "All Games"]
if st.session_state.get("filter_mode") not in filter_modes:
    st.session_state["filter_mode"] = "Player"

filter_mode = st.sidebar.radio(
    "Filter by:",
    filter_modes,
    key="filter_mode"
)

filtered_shots = shots.copy()

if filter_mode == "Team":
    teams = sorted(shots['team'].dropna().unique())
    selected_team = st.sidebar.selectbox("Select Team:", teams, key="team_select")
    filtered_shots = shots[shots['team'] == selected_team]

    # Optional player filter within selected team
    team_players = sorted([p for p in filtered_shots['player'].dropna().unique() if p])
    team_player_options = ["All Players"] + team_players
    if st.session_state.get("team_player_filter") not in team_player_options:
        st.session_state["team_player_filter"] = "All Players"
    team_player_filter = st.sidebar.selectbox("Player:", team_player_options, key="team_player_filter")

    if team_player_filter != "All Players":
        filtered_shots = filtered_shots[filtered_shots['player'] == team_player_filter]
        title = f"Shot Chart - {team_player_filter} ({selected_team})"
    else:
        title = f"Shot Chart - {selected_team}"
    
elif filter_mode == "Player":
    players = sorted(shots['player'].dropna().unique())
    players = [p for p in players if p]  # Remove empty strings
    preferred_player = "Cameron Boozer"
    default_player = preferred_player if preferred_player in players else players[0]
    if st.session_state.get("player_select") not in players:
        st.session_state["player_select"] = default_player

    selected_player = st.sidebar.selectbox("Select Player:", players, key="player_select")
    filtered_shots = shots[shots['player'] == selected_player]
    title = f"Shot Chart - {selected_player}"
    
elif filter_mode == "Game":
    # Create game labels (home vs away)
    game_labels = shots.groupby('game_id').agg({
        'home_team': 'first',
        'away_team': 'first'
    })
    game_labels['label'] = game_labels['home_team'] + ' vs ' + game_labels['away_team']
    game_dict = game_labels['label'].to_dict()
    
    game_options = [f"{game_dict[gid]} (ID: {gid})" for gid in sorted(shots['game_id'].unique())]
    selected_game_str = st.sidebar.selectbox("Select Game:", game_options, key="game_select")
    selected_game_id = int(selected_game_str.split("ID: ")[1].rstrip(")"))
    
    filtered_shots = shots[shots['game_id'] == selected_game_id]
    
    # Option to filter by team within game
    teams_in_game = filtered_shots['team'].dropna().unique()
    team_filter = st.sidebar.radio("Show:", ["Both Teams"] + list(teams_in_game), key="team_filter")
    
    if team_filter != "Both Teams":
        filtered_shots = filtered_shots[filtered_shots['team'] == team_filter]

    # Player filter within game (updates based on team selection)
    game_players = sorted([p for p in filtered_shots['player'].dropna().unique() if p])
    game_player_options = ["All Players"] + game_players
    game_player_filter = st.sidebar.selectbox("Player:", game_player_options, key="game_player_filter")

    if game_player_filter != "All Players":
        filtered_shots = filtered_shots[filtered_shots['player'] == game_player_filter]
        if team_filter != "Both Teams":
            title = f"{game_player_filter} — {team_filter} ({game_dict[selected_game_id]})"
        else:
            title = f"{game_player_filter} — {game_dict[selected_game_id]}"
    elif team_filter != "Both Teams":
        title = f"{team_filter} — {game_dict[selected_game_id]}"
    else:
        title = f"Shot Chart — {game_dict[selected_game_id]}"
else:
    title = "Shot Chart - All Games"

# Chart type selector
if filter_mode == "All Games":
    chart_mode = 'heatmap'
    st.sidebar.caption("All Games view defaults to heatmap.")
else:
    chart_type = st.sidebar.radio(
        "Chart Type:",
        ["Scatter (Individual Shots)", "Heatmap (Expected Points)"],
        key="chart_type"
    )
    chart_mode = 'scatter' if "Scatter" in chart_type else 'heatmap'

st.sidebar.markdown('<div class="creator-credit">Created by Connor Dague | @cdague10</div>', unsafe_allow_html=True)

# Create and display chart with stats on the right
if len(filtered_shots) > 0:
    col_chart, col_stats = st.columns([4, 1])
    
    with col_chart:
        with st.spinner("Generating shot chart..."):
            fig = create_shot_chart(filtered_shots, title, chart_mode)
            if fig:
                st.pyplot(fig)
                plt.close(fig)
    
    with col_stats:
        st.markdown('<div class="stats-panel-header">Statistics</div>', unsafe_allow_html=True)

        total_shots = len(filtered_shots)
        made_shots = int(filtered_shots['is_made'].sum())
        fg_pct = (made_shots / total_shots * 100) if total_shots > 0 else 0

        two_pt = filtered_shots[~filtered_shots['is_three_point']]
        three_pt = filtered_shots[filtered_shots['is_three_point']]

        two_pt_attempts = len(two_pt)
        three_pt_attempts = len(three_pt)
        two_pt_makes = int(two_pt['is_made'].sum())
        three_pt_makes = int(three_pt['is_made'].sum())

        two_pt_pct = (
            f"{(two_pt_makes / two_pt_attempts * 100):.1f}% ({two_pt_makes}/{two_pt_attempts})"
            if two_pt_attempts > 0 else "N/A"
        )
        three_pt_pct = (
            f"{(three_pt_makes / three_pt_attempts * 100):.1f}% ({three_pt_makes}/{three_pt_attempts})"
            if three_pt_attempts > 0 else "N/A"
        )

        render_stats_section("Overall", [
            ("FG%", f"{fg_pct:.1f}%"),
            ("Makes", f"{made_shots}"),
            ("Total Shots", f"{total_shots}")
        ])

        render_stats_section("Shot Distribution", [
            ("2PT%", two_pt_pct),
            ("3PT%", three_pt_pct)
        ])

        total_points = filtered_shots['points'].sum()
        render_stats_section("Scoring", [
            ("Total Points", f"{int(total_points)}"),
            ("Pts/Shot", f"{filtered_shots['points'].mean():.3f}")
        ])

        if 'shot_plus' in filtered_shots.columns:
            shot_plus_series = pd.to_numeric(filtered_shots['shot_plus'], errors='coerce')
            shot_plus_avg = shot_plus_series.mean()

            shot_plus_rows = [
                ("Shot+", f"{shot_plus_avg:.1f}" if pd.notna(shot_plus_avg) else "N/A")
            ]

            if 'expected_points_model' in filtered_shots.columns:
                model_xpts = pd.to_numeric(filtered_shots['expected_points_model'], errors='coerce').mean()
                shot_plus_rows.append(("Model xPts/Shot", f"{model_xpts:.3f}" if pd.notna(model_xpts) else "N/A"))

            if 'shot_grade' in filtered_shots.columns:
                top_grades = filtered_shots['shot_grade'].isin(['A', 'A+']).sum()
                shot_plus_rows.append(("A/A+ Shot Rate", f"{(top_grades/total_shots*100):.1f}%" if total_shots > 0 else "N/A"))

            if 'result_plus' in filtered_shots.columns:
                result_plus_avg = pd.to_numeric(filtered_shots['result_plus'], errors='coerce').mean()
                shot_plus_rows.append(("Result+", f"{result_plus_avg:.1f}" if pd.notna(result_plus_avg) else "N/A"))

            render_stats_section("Shot+", shot_plus_rows)

        stats = calculate_expected_point_stats(filtered_shots)
        render_stats_section("Expected Points", [
            ("Overall", f"{stats['overall']:.3f}"),
            (f"Layup ({stats['layup_count']})", f"{stats['layup']:.3f}"),
            (f"Paint ({stats['paint_count']})", f"{stats['paint']:.3f}"),
            (f"Midrange ({stats['midrange_count']})", f"{stats['midrange']:.3f}"),
            (f"3-Point ({stats['three_pt_count']})", f"{stats['three_pt']:.3f}")
        ])
else:
    st.warning("No shots found for the selected filters.")


