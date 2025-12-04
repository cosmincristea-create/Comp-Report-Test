import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os
import json
from mplsoccer import Pitch, PyPizza, VerticalPitch
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats

# Set page config
st.set_page_config(layout="wide", page_title="Romanian Football Analytics", page_icon="⚽")

# Custom CSS for fonts
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@500&display=swap');

h1, h2, h3 {
    font-family: 'Roboto Slab', serif !important;
}
body {
    font-family: 'Google Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("Romanian Football Analytics Hub (Liga I & II)")

# --- Constants & Configuration ---
LIGA_II_ID = '1adpmWo0Bx4AskINiHQVZqh_yEnIjtBLh'
SUPERLIGA_ID = '1LmyY56NSMw95kRIruFKsY7p4w1vzglda'
ROLE_DEFINITIONS_FILE = 'role_definitions.json'

# --- Helper Functions ---

@st.cache_data
def download_file(file_id, output_filename):
    """Downloads a file from Google Drive if it doesn't exist."""
    if os.path.exists(output_filename):
        return output_filename

    url = f'https://drive.google.com/uc?id={file_id}'
    try:
        gdown.download(url, output_filename, quiet=False)
        return output_filename
    except Exception as e:
        st.error(f"Failed to download {output_filename}: {e}")
        return None

def detect_file_type(df):
    """Detects if the dataframe is Event Data or Stats Data."""
    cols = df.columns
    if 'eventId' in cols and 'x' in cols and 'y' in cols:
        return 'event'
    elif 'total_goals' in cols or 'average_goals' in cols:
        return 'stats'
    return 'unknown'

def normalize_event_columns(df):
    """Standardizes column names for Event Data."""
    # Map possible variations to standard names
    col_map = {
        'teamId': ['team.id', 'team_id', 'team', 'teams_wyId'],
        'playerId': ['player.id', 'player_id', 'player', 'player_wyId'],
        'matchId': ['match.id', 'match_id', 'match'],
        'subEventName': ['event_secondary_type', 'sub_event', 'subEvent', 'sub_event_name'],
        'eventName': ['event_type', 'event_name', 'event', 'event_type_name', 'type_name'],
        'x': ['location.x', 'pos_x', 'start_x', 'x_start', 'positions_0_x'],
        'y': ['location.y', 'pos_y', 'start_y', 'y_start', 'positions_0_y'],
        'end_x': ['pass.endLocation.x', 'end_location.x', 'end_x', 'positions_1_x'],
        'end_y': ['pass.endLocation.y', 'end_location.y', 'end_y', 'positions_1_y']
    }

    renamed = {}
    for standard, variations in col_map.items():
        if standard in df.columns:
            continue
        for var in variations:
            if var in df.columns:
                renamed[var] = standard
                break

    if renamed:
        df = df.rename(columns=renamed)

    # Fallback if eventName is still missing but subEventName exists
    if 'eventName' not in df.columns and 'subEventName' in df.columns:
        df['eventName'] = df['subEventName']

    # Check if we need to parse 'positions' column (Wyscout JSON format)
    # Check both 'positions' and potential aliases like 'pos'
    pos_col = None
    if 'positions' in df.columns:
        pos_col = 'positions'
    elif 'pos' in df.columns:
        pos_col = 'pos'

    if 'x' not in df.columns and pos_col:
        import ast

        def parse_pos(row):
            try:
                pos_data = row
                if isinstance(pos_data, str):
                    # Handle typical CSV issues where JSON might be malformed or doubled quotes
                    pos_data = pos_data.replace('""', '"')
                    pos_data = ast.literal_eval(pos_data)

                if isinstance(pos_data, list) and len(pos_data) > 0:
                    start = pos_data[0]
                    x = start.get('x')
                    y = start.get('y')
                    end_x = x
                    end_y = y
                    if len(pos_data) > 1:
                        end = pos_data[1]
                        end_x = end.get('x')
                        end_y = end.get('y')
                    return pd.Series([x, y, end_x, end_y])
            except:
                pass
            return pd.Series([None, None, None, None])

        # Apply parsing
        cols = df[pos_col].apply(parse_pos)
        cols.columns = ['x', 'y', 'end_x', 'end_y']
        df = pd.concat([df, cols], axis=1)

    return df

@st.cache_data
def load_and_process_wyscout(file_path):
    """
    Loads Wyscout event CSV and normalizes columns.
    Returns a DataFrame with standard columns.
    """
    if not os.path.exists(file_path):
        return pd.DataFrame()

    # Load and Normalize
    df = pd.read_csv(file_path)
    df = normalize_event_columns(df)

    return df

def load_role_definitions():
    with open(ROLE_DEFINITIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def map_position_to_family(position_code, role_defs):
    """Maps a specific position code to a Position Family."""
    code = str(position_code).lower().strip()

    # Iterate through families in the JSON
    for family, roles in role_defs.items():
        for role, details in roles.items():
            if code in details['position_codes']:
                return family

    # Fallback logic if code not found in JSON explicit lists
    if code in ['gk']: return 'Goalkeeper'
    if code in ['cb', 'rcb', 'lcb', 'rcb3', 'lcb3']: return 'Centre Back'
    if code in ['rb', 'lb', 'rwb', 'lwb', 'rb5', 'lb5']: return 'Fullback'
    if code in ['dmf', 'ldmf', 'rdmf', 'cmf', 'lcmf', 'rcmf', 'amf', 'lamf', 'ramf']: return 'Centre Midfielder'
    if code in ['rwf', 'lwf', 'rw', 'lw', 'cf', 'ss']: return 'Attacker'

    return 'Unknown'

def preprocess_stats(df, role_defs):
    """
    Cleans and processes player stats:
    1. Deduplicates by playerId.
    2. Maps to Position Family.
    3. Normalizes per 90.
    4. Calculates Percentiles & Z-Scores within Family.
    """
    if 'positions_percent' in df.columns:
        df = df.sort_values('positions_percent', ascending=False)

    # Deduplicate
    df = df.drop_duplicates(subset=['playerId'], keep='first').copy()

    # Map Position Family
    pos_col = 'positions_position_code' if 'positions_position_code' in df.columns else None

    if pos_col:
        df['PositionFamily'] = df[pos_col].apply(lambda x: map_position_to_family(x, role_defs))
    else:
        df['PositionFamily'] = 'Unknown'

    # Filter out unknowns or minimal minutes
    df = df[df['total_minutesOnField'] > 90]

    # Normalize per 90
    total_cols = [c for c in df.columns if c.startswith('total_') and c != 'total_minutesOnField']

    norm_cols = {}
    for col in total_cols:
        new_col = col.replace('total_', 'per90_')
        norm_cols[new_col] = (df[col] / df['total_minutesOnField']) * 90

    if norm_cols:
        norm_df = pd.DataFrame(norm_cols)
        df = pd.concat([df, norm_df], axis=1)

    # Calculate Percentiles and Z-Scores WITHIN Position Family
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    new_cols = {}

    for family in df['PositionFamily'].unique():
        if family == 'Unknown': continue

        family_mask = df['PositionFamily'] == family
        subset = df.loc[family_mask, numeric_cols]

        for col in numeric_cols:
            pct_col_name = f'pct_{col}'
            z_col_name = f'z_{col}'

            if pct_col_name not in new_cols:
                new_cols[pct_col_name] = pd.Series(index=df.index, dtype='float64')
            if z_col_name not in new_cols:
                new_cols[z_col_name] = pd.Series(index=df.index, dtype='float64')

            if subset[col].abs().sum() == 0:
                new_cols[pct_col_name].loc[family_mask] = 0
                new_cols[z_col_name].loc[family_mask] = 0
            else:
                new_cols[pct_col_name].loc[family_mask] = subset[col].rank(pct=True) * 100
                if subset[col].std() > 0:
                     new_cols[z_col_name].loc[family_mask] = (subset[col] - subset[col].mean()) / subset[col].std()
                else:
                     new_cols[z_col_name].loc[family_mask] = 0

    if new_cols:
        new_cols_df = pd.DataFrame(new_cols)
        df = pd.concat([df, new_cols_df], axis=1)

    if 'player_birthDate' in df.columns:
        df['player_birthDate'] = pd.to_datetime(df['player_birthDate'], errors='coerce')
        now = pd.Timestamp.now()
        df['Age'] = (now - df['player_birthDate']).dt.days // 365
        df['Age'] = df['Age'].fillna(0).astype(int)
    else:
        df['Age'] = 0

    return df

@st.cache_data
def load_data():
    role_defs = load_role_definitions()

    # 1. Download Events
    liga_ii_event_file = download_file(LIGA_II_ID, 'LigaII_2025_2026_events.csv')
    superliga_event_file = download_file(SUPERLIGA_ID, 'Superliga_719_2025_2026_events.csv')

    data_store = {
        'events': {},
        'stats': {}
    }

    # Prioritize Local Files for Events if they exist
    local_liga = 'LigaII_2025_2026_events.csv'
    local_super = 'Superliga_719_2025_2026_events.csv'

    if os.path.exists(local_liga):
        liga_ii_event_file = local_liga

    if os.path.exists(local_super):
        superliga_event_file = local_super

    if liga_ii_event_file:
        df = load_and_process_wyscout(liga_ii_event_file)
        if not df.empty:
            df['Competition'] = 'Liga II'
            data_store['events']['Liga II'] = df

    if superliga_event_file:
        df = load_and_process_wyscout(superliga_event_file)
        if not df.empty:
            df['Competition'] = 'Superliga'
            data_store['events']['Superliga'] = df

    def load_and_tag(path, competition):
        if not path or not os.path.exists(path): return None
        df = pd.read_csv(path)
        df['Competition'] = competition
        return df

    repo_files = [
        ('all_players_advanced_stats_Liga2_2025_2026.csv', 'Liga II'),
        ('all_players_advanced_stats_Superliga_2025_2026.csv', 'Superliga'),
        ('teams_advanced_stats_Liga2_2025_2026.csv', 'Liga II'),
        ('teams_advanced_stats_Superliga_2025_2026.csv', 'Superliga')
    ]

    player_stats_list = []
    team_stats_list = []

    for fpath, comp in repo_files:
        if os.path.exists(fpath):
            df = load_and_tag(fpath, comp)
            if 'player' in fpath:
                player_stats_list.append(df)
            else:
                team_stats_list.append(df)

    if player_stats_list:
        full_player_df = pd.concat(player_stats_list, ignore_index=True)
        data_store['stats']['players'] = preprocess_stats(full_player_df, role_defs)

    if team_stats_list:
        data_store['stats']['teams'] = pd.concat(team_stats_list, ignore_index=True)

    return data_store, role_defs

# --- Main App Logic ---

with st.spinner("Loading Data..."):
    data, role_definitions = load_data()

st.success("Data Loaded Successfully!")

st.sidebar.header("Global Settings")

clean_comps = ["Superliga", "Liga II"]
selected_comp = st.sidebar.selectbox("Select Competition", clean_comps, index=0)

uploaded_files = st.sidebar.file_uploader("Upload CSV Data", accept_multiple_files=True, type=['csv'])
if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        ftype = detect_file_type(df)
        if ftype == 'event':
            df['Competition'] = 'Uploaded'
            df = normalize_event_columns(df)
            data['events'][uploaded_file.name] = df
        elif ftype == 'stats':
            if 'playerId' in df.columns:
                processed = preprocess_stats(df, role_definitions)
                processed['Competition'] = 'Uploaded'
                if 'players' in data['stats']:
                    data['stats']['players'] = pd.concat([data['stats']['players'], processed], ignore_index=True)
                else:
                    data['stats']['players'] = processed
            else:
                st.warning(f"Uploaded stats file {uploaded_file.name} does not have 'playerId'.")

tab_team, tab_player, tab_zscore, tab_advanced = st.tabs(["Team Analysis", "Player Analysis", "Role Rankings (Z-Scores)", "Advanced Models"])

# --- Tab A: Team Analysis ---
with tab_team:
    st.header("Team Analysis Hub")

    team_df = data['stats']['teams'] if 'teams' in data['stats'] else pd.DataFrame()

    if not team_df.empty:
        team_df['Competition'] = team_df['Competition'].astype(str)
        available_teams = team_df[team_df['Competition'] == str(selected_comp)]['team_name'].sort_values().unique()
    else:
        available_teams = []

    col_filters, col_viz = st.columns([1, 2])

    with col_filters:
        team_a = st.selectbox("Select Team A", available_teams)
        team_b = st.selectbox("Select Team B (Comparison)", ["None"] + list(available_teams))

        # Match Selector (Using Raw Event Data)
        match_options = {}
        if selected_comp in data['events']:
            events_df = data['events'][selected_comp]
            if not team_df.empty:
                team_info = team_df[(team_df['team_name'] == team_a) & (team_df['Competition'] == selected_comp)]
                if not team_info.empty:
                    team_a_id = team_info.iloc[0]['teamId']

                    if 'teamId' in events_df.columns:
                        team_matches = events_df[events_df['teamId'] == team_a_id]['matchId'].unique()

                        for m_id in team_matches:
                            m_events = events_df[events_df['matchId'] == m_id]
                            teams_in_match = m_events['teamId'].unique()
                            opp_id = next((t for t in teams_in_match if t != team_a_id), None)

                            label = f"Match {m_id}"
                            if opp_id:
                                opp_row = team_df[team_df['teamId'] == opp_id]
                                if not opp_row.empty:
                                    opp_name = opp_row.iloc[0]['team_name']
                                    label = f"vs {opp_name}"

                            match_options[label] = m_id

        selected_match_label = st.selectbox("Select Match", ["All"] + list(match_options.keys()))
        selected_match = match_options[selected_match_label] if selected_match_label != "All" else "All"

        # --- Top Performers Section ---
        st.subheader("Top Performers (xT, xG, Defensive)")
        if selected_comp in data['events']:
             ev_df = data['events'][selected_comp]

             # Filter by Match or Team
             if selected_match != "All":
                 perf_events = ev_df[ev_df['matchId'] == selected_match]
                 # If we want only Team A players? Usually top performers across the game or team?
                 # Let's show for Team A specifically as it's the main selector.
                 team_info = team_df[(team_df['team_name'] == team_a) & (team_df['Competition'] == selected_comp)]
                 if not team_info.empty:
                    tid = team_info.iloc[0]['teamId']
                    perf_events = perf_events[perf_events['teamId'] == tid]
             else:
                 # All matches for Team A
                 team_info = team_df[(team_df['team_name'] == team_a) & (team_df['Competition'] == selected_comp)]
                 if not team_info.empty:
                    tid = team_info.iloc[0]['teamId']
                    perf_events = ev_df[ev_df['teamId'] == tid]
                 else:
                    perf_events = pd.DataFrame()

             if not perf_events.empty:
                 # 1. xT Generated (Karun Singh Grid)
                 # 16x12 grid
                 xt_grid = np.array([
                    [0.00638, 0.0077, 0.00845, 0.00978, 0.01126, 0.01248, 0.01474, 0.01745, 0.02122, 0.02756, 0.03485, 0.03792, 0.03505, 0.02806, 0.02421, 0.01575],
                    [0.0075, 0.00869, 0.00997, 0.01112, 0.01267, 0.01429, 0.01686, 0.01935, 0.02412, 0.0304, 0.04066, 0.04543, 0.04615, 0.0416, 0.03687, 0.02404],
                    [0.00888, 0.00978, 0.01143, 0.01242, 0.01494, 0.01666, 0.02017, 0.02271, 0.02945, 0.03816, 0.05151, 0.0601, 0.0683, 0.06525, 0.05206, 0.03351],
                    [0.0097, 0.01087, 0.01247, 0.0143, 0.01625, 0.01918, 0.02256, 0.02677, 0.03456, 0.04639, 0.06319, 0.0772, 0.09117, 0.10685, 0.09279, 0.05581],
                    [0.01048, 0.01146, 0.01347, 0.01493, 0.01804, 0.02081, 0.02552, 0.03108, 0.0425, 0.05597, 0.0768, 0.10237, 0.12932, 0.16543, 0.15582, 0.09063],
                    [0.01143, 0.01223, 0.01389, 0.01614, 0.0187, 0.02244, 0.02728, 0.03498, 0.04705, 0.06059, 0.08985, 0.11306, 0.16508, 0.2312, 0.26089, 0.16709],
                    [0.01143, 0.01223, 0.01389, 0.01614, 0.0187, 0.02244, 0.02728, 0.03498, 0.04705, 0.06059, 0.08985, 0.11306, 0.16508, 0.2312, 0.26089, 0.16709],
                    [0.01048, 0.01146, 0.01347, 0.01493, 0.01804, 0.02081, 0.02552, 0.03108, 0.0425, 0.05597, 0.0768, 0.10237, 0.12932, 0.16543, 0.15582, 0.09063],
                    [0.0097, 0.01087, 0.01247, 0.0143, 0.01625, 0.01918, 0.02256, 0.02677, 0.03456, 0.04639, 0.06319, 0.0772, 0.09117, 0.10685, 0.09279, 0.05581],
                    [0.00888, 0.00978, 0.01143, 0.01242, 0.01494, 0.01666, 0.02017, 0.02271, 0.02945, 0.03816, 0.05151, 0.0601, 0.0683, 0.06525, 0.05206, 0.03351],
                    [0.0075, 0.00869, 0.00997, 0.01112, 0.01267, 0.01429, 0.01686, 0.01935, 0.02412, 0.0304, 0.04066, 0.04543, 0.04615, 0.0416, 0.03687, 0.02404],
                    [0.00638, 0.0077, 0.00845, 0.00978, 0.01126, 0.01248, 0.01474, 0.01745, 0.02122, 0.02756, 0.03485, 0.03792, 0.03505, 0.02806, 0.02421, 0.01575]
                 ])

                 # Calculate xT for successful passes/dribbles
                 # Wyscout coords 0-100
                 # Grid 16x12
                 def calc_xt_raw(row):
                    try:
                        # Ensure success?
                        # Wyscout typically has tags for success, or we check 'subEventName' or similar?
                        # Actually raw wyscout 'tags' column contains 1801 (Success) etc.
                        # But simpler check: if it has end_x, it's likely a pass.
                        # We will assume all rows passed here are valid actions.

                        # X: 0-100 -> 0-16
                        start_x_bin = int(np.clip(row['x'] / 100 * 16, 0, 15))
                        end_x_bin = int(np.clip(row['end_x'] / 100 * 16, 0, 15))

                        # Y: 0-100 -> 0-12
                        start_y_bin = int(np.clip(row['y'] / 100 * 12, 0, 11))
                        end_y_bin = int(np.clip(row['end_y'] / 100 * 12, 0, 11))

                        return xt_grid[end_y_bin][end_x_bin] - xt_grid[start_y_bin][start_x_bin]
                    except:
                        return 0.0

                 # Passes/Carries
                 # Filter by eventName or subEventName
                 # Check if eventName exists (it should due to normalization, but be safe)
                 if 'eventName' in perf_events.columns:
                     valid_moves = perf_events[perf_events['eventName'].isin(['Pass', 'Others on the ball', 'Free Kick', 'Shot'])]
                 else:
                     valid_moves = pd.DataFrame()

                 if not valid_moves.empty:
                    valid_moves['xT'] = valid_moves.apply(calc_xt_raw, axis=1)
                    xt_stats = valid_moves.groupby('playerId')['xT'].sum().reset_index().rename(columns={'xT': 'xT Generated'})
                 else:
                    xt_stats = pd.DataFrame(columns=['playerId', 'xT Generated'])

                 # 2. xG Generated
                 # Check for 'xg' column in perf_events
                 # Normalize 'xg' column name?
                 xg_col = next((c for c in perf_events.columns if 'xg' in c.lower()), None)
                 if xg_col:
                     xg_stats = perf_events.groupby('playerId')[xg_col].sum().reset_index().rename(columns={xg_col: 'xG Generated'})
                 else:
                     xg_stats = pd.DataFrame(columns=['playerId', 'xG Generated'])

                 # 3. Defensive Actions (Duels + Interceptions + Recoveries)
                 # Filter: eventName 'Duel', 'Interception', 'Others on the ball' (recoveries?)
                 # Wyscout 'Duel' is explicit. 'Interception' is often a tag or subEvent.
                 # Simplified: Count events named 'Duel', 'Interception', 'Interruption'
                 if 'eventName' in perf_events.columns:
                     def_mask = perf_events['eventName'].isin(['Duel', 'Interception', 'Save attempt', 'Goalkeeper leaving line', 'Interruption'])
                     def_stats = perf_events[def_mask].groupby('playerId').size().reset_index(name='Defensive Actions')
                 else:
                     def_stats = pd.DataFrame(columns=['playerId', 'Defensive Actions'])

                 # Merge
                 merged_perf = xt_stats.merge(xg_stats, on='playerId', how='outer').merge(def_stats, on='playerId', how='outer').fillna(0)

                 # Add Player Names
                 if 'players' in data['stats']:
                     pdf = data['stats']['players']
                     # Minimal info
                     pinfo = pdf[['playerId', 'player_shortName']].drop_duplicates()
                     merged_perf = merged_perf.merge(pinfo, on='playerId', how='left')

                 # Display
                 st.dataframe(merged_perf[['player_shortName', 'xT Generated', 'xG Generated', 'Defensive Actions']].sort_values('xT Generated', ascending=False).head(10))

        # Comparison Table
        if team_b != "None" and not team_df.empty:
            st.subheader("Head-to-Head Stats")
            metrics = {
                'Goals': 'average_goals',
                'xG': 'average_xgShot',
                'Possession %': 'average_possessionPercent',
                'Shots': 'average_shots',
                'PPDA': 'total_ppda',
                'Fouls': 'average_fouls',
                'Goals Conceded': 'average_concededGoals'
            }
            inverse_metrics = ['total_ppda', 'average_fouls', 'average_concededGoals']

            row_a = team_df[(team_df['team_name'] == team_a) & (team_df['Competition'] == selected_comp)].iloc[0]
            row_b = team_df[(team_df['team_name'] == team_b) & (team_df['Competition'] == selected_comp)].iloc[0]

            comp_data = []
            for label, col in metrics.items():
                val_a = row_a.get(col, 0)
                val_b = row_b.get(col, 0)

                color_a = "white"
                color_b = "white"

                if val_a != val_b:
                    is_inverse = col in inverse_metrics
                    if is_inverse:
                        better_a = val_a < val_b
                    else:
                        better_a = val_a > val_b

                    color_a = "lightgreen" if better_a else "lightcoral"
                    color_b = "lightcoral" if better_a else "lightgreen"

                comp_data.append({
                    "Metric": label,
                    f"{team_a}": val_a,
                    f"{team_b}": val_b,
                    "color_a": color_a,
                    "color_b": color_b
                })

            html = "<table><thead><tr><th>Metric</th><th>{}</th><th>{}</th></tr></thead><tbody>".format(team_a, team_b)
            for row in comp_data:
                html += f"<tr><td>{row['Metric']}</td>"
                html += f"<td style='background-color:{row['color_a']}; color:black'>{row[f'{team_a}']:.2f}</td>"
                html += f"<td style='background-color:{row['color_b']}; color:black'>{row[f'{team_b}']:.2f}</td></tr>"
            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)

    with col_viz:
        st.subheader("Match Visualizations")

        if selected_comp in data['events'] and selected_match != "All":
            events_df = data['events'][selected_comp]

            # Filter Match and Team A
            # Get ID
            team_info = team_df[(team_df['team_name'] == team_a) & (team_df['Competition'] == selected_comp)]
            if not team_info.empty:
                team_a_id = team_info.iloc[0]['teamId']

                match_events = events_df[(events_df['matchId'] == selected_match) & (events_df['teamId'] == team_a_id)]

                # 1. Pass Map
                st.markdown("### Pass Map")
                # Filter Passes
                passes = match_events[match_events['eventName'] == 'Pass'].copy()

                # Pitch: Wyscout (0-100)
                pitch = Pitch(pitch_type='wyscout', pitch_color='#222222', line_color='#c7d5cc')
                fig, ax = pitch.draw(figsize=(10, 7))

                # Use a simple success logic? Wyscout CSV doesn't strictly have 'result_name' like SPADL.
                # Assuming all are passes for now, or check for failure tags (1802).
                # We will plot all as one color if we can't easily distinguish, or just Green.
                pitch.arrows(passes.x, passes.y, passes.end_x, passes.end_y, ax=ax, color='green', width=2, headwidth=3, alpha=0.6)

                st.pyplot(fig)

                # 2. Shot Map
                st.markdown("### Shot Map")
                shots = match_events[match_events['eventName'] == 'Shot'].copy()

                fig2, ax2 = pitch.draw(figsize=(10, 7))

                # Check xG
                xg_col = next((c for c in shots.columns if 'xg' in c.lower()), None)
                sizes = shots[xg_col] * 500 if xg_col else 100

                pitch.scatter(shots.x, shots.y, ax=ax2, color='blue', edgecolors='white', s=sizes, alpha=0.7)
                st.pyplot(fig2)

        elif selected_match == "All":
             st.info("Select a specific match to view Pass Maps and Shot Maps.")
        else:
             st.warning("No Event Data available for this competition.")

# --- Tab B: Player Analysis ---
with tab_player:
    st.header("Player Analysis Hub")

    if 'stats' in data and 'players' in data['stats']:
        player_df = data['stats']['players']
        player_df['Competition'] = player_df['Competition'].astype(str)
        comp_player_df = player_df[player_df['Competition'] == str(selected_comp)]

        teams_list = sorted(comp_player_df['team_name'].dropna().unique())
        selected_team_p = st.selectbox("Select Team", teams_list, key="p_team")

        team_players = comp_player_df[comp_player_df['team_name'] == selected_team_p]
        families = sorted(team_players['PositionFamily'].unique())
        selected_family = st.selectbox("Select Position Family", families)

        family_players = team_players[team_players['PositionFamily'] == selected_family]
        player_names = sorted(family_players['player_shortName'].unique())
        selected_player = st.selectbox("Select Player", player_names)

        if selected_player:
            player_stats = family_players[family_players['player_shortName'] == selected_player].iloc[0]

            col_pizza, col_heat = st.columns(2)

            with col_pizza:
                st.subheader("Percentile Pizza Plot")
                if selected_family in role_definitions:
                    archetypes = role_definitions[selected_family]
                    selected_archetype_name = st.selectbox("Select Archetype Template", list(archetypes.keys()))
                    archetype = archetypes[selected_archetype_name]
                    metric_mapping = archetype['metrics']
                    params = list(metric_mapping.keys())

                    values = []
                    valid_params = []

                    for p in params:
                        pct_col = f"pct_{p}"
                        if pct_col in player_stats:
                            values.append(int(player_stats[pct_col]))
                            valid_params.append(p.replace("average_", "").replace("percent_", "").replace("_", " ").title())

                    if values:
                        slice_colors = []
                        for v in values:
                            if v >= 90: slice_colors.append("#2ca02c")
                            elif v >= 60: slice_colors.append("#1f77b4")
                            else: slice_colors.append("#d62728")

                        baker = PyPizza(
                            params=valid_params,
                            background_color="#222222",
                            straight_line_color="#000000",
                            straight_line_lw=1,
                            last_circle_lw=0,
                            other_circle_lw=0,
                            inner_circle_size=20
                        )

                        fig_pizza, ax_pizza = baker.make_pizza(
                            values,
                            figsize=(8, 8),
                            color_blank_space="same",
                            slice_colors=slice_colors,
                            value_colors=["white"] * len(values),
                            value_bck_colors=slice_colors,
                            blank_alpha=0.4,
                            kwargs_slices=dict(edgecolor="#000000", zorder=2, linewidth=1),
                            kwargs_params=dict(color="white", fontsize=11, va="center"),
                            kwargs_values=dict(color="white", fontsize=11, zorder=3,
                                               bbox=dict(edgecolor="#000000", facecolor="cornflowerblue", boxstyle="round,pad=0.2", lw=1))
                        )
                        st.pyplot(fig_pizza)
                    else:
                        st.warning("No valid metrics found.")
                else:
                    st.info("No templates found.")

            with col_heat:
                st.subheader("Event Heatmap")
                if selected_comp in data['events']:
                    events_df = data['events'][selected_comp]
                    pid = player_stats['playerId']
                    p_col = 'playerId'

                    player_events = events_df[events_df[p_col] == pid]

                    if not player_events.empty:
                         pitch = Pitch(pitch_type='wyscout', pitch_color='#222222', line_color='#efefef')
                         fig_h, ax_h = pitch.draw(figsize=(10, 7))

                         valid_ev = player_events.dropna(subset=['x', 'y'])
                         if not valid_ev.empty:
                             pitch.kdeplot(valid_ev.x, valid_ev.y, ax=ax_h, cmap='hot', fill=True, levels=100, alpha=0.6)
                             st.pyplot(fig_h)
                         else:
                             st.write("No location data.")
                    else:
                        st.write("No events found.")
                else:
                    st.write("Event data not loaded.")

            st.subheader("Head-to-Head Comparison")
            opp_player_name = st.selectbox("Select Opponent Player", ["None"] + list(player_names), index=0, key='opp_p')

            if opp_player_name != "None":
                 opp_stats = family_players[family_players['player_shortName'] == opp_player_name].iloc[0]

                 st.write("### Head-to-Head Comparison")
                 if selected_family in role_definitions:
                    if 'valid_params' in locals() and valid_params:
                        values_p1 = values
                        values_p2 = []
                        for p in params:
                            pct_col = f"pct_{p}"
                            if pct_col in opp_stats:
                                values_p2.append(int(opp_stats[pct_col]))
                            else:
                                values_p2.append(0)

                        def plot_mini_pizza(vals, p_names, title):
                            slice_colors_local = []
                            for v in vals:
                                if v >= 90: slice_colors_local.append("#2ca02c")
                                elif v >= 60: slice_colors_local.append("#1f77b4")
                                else: slice_colors_local.append("#d62728")

                            baker_local = PyPizza(
                                params=p_names,
                                background_color="#222222",
                                straight_line_color="#000000",
                                straight_line_lw=1,
                                last_circle_lw=0,
                                other_circle_lw=0,
                                inner_circle_size=20
                            )
                            return baker_local.make_pizza(
                                vals,
                                figsize=(8, 8),
                                color_blank_space="same",
                                slice_colors=slice_colors_local,
                                value_colors=["white"] * len(vals),
                                value_bck_colors=slice_colors_local,
                                blank_alpha=0.4,
                                kwargs_slices=dict(edgecolor="#000000", zorder=2, linewidth=1),
                                kwargs_params=dict(color="white", fontsize=11, va="center"),
                                kwargs_values=dict(color="white", fontsize=11, zorder=3,
                                                   bbox=dict(edgecolor="#000000", facecolor="cornflowerblue", boxstyle="round,pad=0.2", lw=1))
                            )

                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption(f"{selected_player}")
                            fig_p1, _ = plot_mini_pizza(values_p1, valid_params, selected_player)
                            st.pyplot(fig_p1)
                        with c2:
                            st.caption(f"{opp_player_name}")
                            fig_p2, _ = plot_mini_pizza(values_p2, valid_params, opp_player_name)
                            st.pyplot(fig_p2)

                 st.write("### Verdict Table")
                 comp_rows = []
                 for p in params:
                     val_a = player_stats.get(p, 0)
                     val_b = opp_stats.get(p, 0)
                     is_inverse = 'fouls' in p.lower() or 'losses' in p.lower() or 'conceded' in p.lower()

                     color_a, color_b = "white", "white"
                     if val_a != val_b:
                         if is_inverse:
                             better_a = val_a < val_b
                         else:
                             better_a = val_a > val_b
                         color_a = "lightgreen" if better_a else "lightcoral"
                         color_b = "lightcoral" if better_a else "lightgreen"

                     comp_rows.append({
                         "Metric": p,
                         selected_player: val_a,
                         opp_player_name: val_b,
                         "ca": color_a,
                         "cb": color_b
                     })

                 html_p = f"<table><thead><tr><th>Metric</th><th>{selected_player}</th><th>{opp_player_name}</th></tr></thead><tbody>"
                 for r in comp_rows:
                     html_p += f"<tr><td>{r['Metric']}</td>"
                     html_p += f"<td style='background-color:{r['ca']}; color:black'>{r[selected_player]:.2f}</td>"
                     html_p += f"<td style='background-color:{r['cb']}; color:black'>{r[opp_player_name]:.2f}</td></tr>"
                 html_p += "</tbody></table>"
                 st.markdown(html_p, unsafe_allow_html=True)

# --- Tab C: Role Rankings (Z-Scores) ---
with tab_zscore:
    st.header("Role Rankings (Weighted Z-Scores)")

    if 'stats' in data and 'players' in data['stats']:
        player_df = data['stats']['players']

        # Select Position Family
        z_families = sorted(player_df['PositionFamily'].unique())
        selected_z_family = st.selectbox("Select Position Family", z_families, key='z_fam')

        if selected_z_family in role_definitions:
            roles = role_definitions[selected_z_family]
            selected_role = st.selectbox("Select Role", list(roles.keys()), key='z_role')

            # Calculate Scores
            # Weights are in roles[selected_role]['metrics']
            weights = roles[selected_role]['metrics']

            # Filter players in this family (across all competitions or selected?)
            # Usually ranking is per competition.
            df_filtered = player_df[(player_df['PositionFamily'] == selected_z_family) & (player_df['Competition'] == str(selected_comp))].copy()

            if not df_filtered.empty:
                # Calculate Score
                # Score = Sum(z_metric * weight)

                score_col_name = f"{selected_role} Score"
                df_filtered[score_col_name] = 0.0

                valid_calc = False
                for metric, weight in weights.items():
                    z_col = f"z_{metric}"
                    if z_col in df_filtered.columns:
                        df_filtered[score_col_name] += df_filtered[z_col] * weight
                        valid_calc = True

                if valid_calc:
                    # Rank
                    df_filtered = df_filtered.sort_values(score_col_name, ascending=False)

                    # Display Table
                    st.write(f"Top Players for **{selected_role}** Role")

                    # Columns to show
                    cols_show = ['player_shortName', 'team_name', 'Age', score_col_name]
                    # Add raw metrics for context? Maybe top 3.
                    top_metrics = list(weights.keys())[:3]
                    cols_show.extend(top_metrics)

                    st.dataframe(df_filtered[cols_show].head(20).style.background_gradient(subset=[score_col_name], cmap='Greens'))
                else:
                    st.warning("Metrics for this role not found in data.")
            else:
                st.info("No players found for this family in selected competition.")
        else:
            st.info("No definitions for this family.")

# --- Tab D: Advanced Models (Clustering) ---
with tab_advanced:
    st.header("Advanced Models")

    st.subheader("Player Clustering (K-Means)")

    if 'stats' in data and 'players' in data['stats']:
        player_df = data['stats']['players']

        families_c = sorted(player_df['PositionFamily'].unique())
        sel_fam_c = st.selectbox("Select Position Family for Clustering", families_c)

        cluster_data = player_df[player_df['PositionFamily'] == sel_fam_c].copy()

        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if not c.startswith('pct_') and not c.startswith('z_') and 'Id' not in c and 'percent' not in c]

        selected_features = st.multiselect("Select Features for Clustering", feature_cols, default=feature_cols[:5])

        if len(selected_features) > 1:
            X = cluster_data[selected_features].fillna(0)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=2)
            pca_res = pca.fit_transform(X_scaled)

            k = st.slider("Select Number of Clusters (K)", 2, 10, 4)
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)

            cluster_data['PCA1'] = pca_res[:, 0]
            cluster_data['PCA2'] = pca_res[:, 1]
            cluster_data['Cluster'] = clusters

            chart = alt.Chart(cluster_data).mark_circle(size=60).encode(
                x='PCA1',
                y='PCA2',
                color='Cluster:N',
                tooltip=['player_shortName', 'team_name', 'PositionFamily', 'Cluster', alt.Tooltip('Age', type='quantitative')] + selected_features
            ).interactive().properties(
                title='Player Clusters (PCA)'
            )

            st.altair_chart(chart, use_container_width=True)

            st.markdown("**Cluster Summary (Average Values):**")
            summary = cluster_data.groupby('Cluster')[selected_features].mean()
            st.dataframe(summary.style.highlight_max(axis=0, color='lightgreen'))

        else:
            st.warning("Select at least 2 features.")
