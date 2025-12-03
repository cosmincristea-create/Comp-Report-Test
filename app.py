import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os
import json
from mplsoccer import Pitch, PyPizza, VerticalPitch
import matplotlib.pyplot as plt
import seaborn as sns
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
        # This can be slow, but necessary if data is in this format
        cols = df[pos_col].apply(parse_pos)
        cols.columns = ['x', 'y', 'end_x', 'end_y']
        df = pd.concat([df, cols], axis=1)

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
    1. Deduplicates by playerId (keeping highest positions_percent).
    2. Maps to Position Family.
    3. Normalizes per 90.
    4. Calculates Percentiles & Z-Scores within Family.
    """
    if 'positions_percent' in df.columns:
        # Sort by percent descending so we keep the highest one
        df = df.sort_values('positions_percent', ascending=False)

    # Deduplicate
    df = df.drop_duplicates(subset=['playerId'], keep='first').copy()

    # Map Position Family
    # Use 'positions_position_code' if available, else try 'player_role_code2' or similar
    pos_col = 'positions_position_code' if 'positions_position_code' in df.columns else None

    if pos_col:
        df['PositionFamily'] = df[pos_col].apply(lambda x: map_position_to_family(x, role_defs))
    else:
        # Fallback if specific code column is missing
        df['PositionFamily'] = 'Unknown'

    # Filter out unknowns or minimal minutes
    df = df[df['total_minutesOnField'] > 90] # Minimum 90 mins played

    # Normalize per 90
    # Identify total columns (start with 'total_')
    total_cols = [c for c in df.columns if c.startswith('total_') and c != 'total_minutesOnField']

    for col in total_cols:
        new_col = col.replace('total_', 'per90_')
        df[new_col] = (df[col] / df['total_minutesOnField']) * 90

    # Calculate Percentiles and Z-Scores WITHIN Position Family
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Initialize columns
    for col in numeric_cols:
        df[f'pct_{col}'] = np.nan
        df[f'z_{col}'] = np.nan

    for family in df['PositionFamily'].unique():
        if family == 'Unknown': continue

        family_mask = df['PositionFamily'] == family
        subset = df.loc[family_mask, numeric_cols]

        # Percentiles (RankPct)
        for col in numeric_cols:
            # Handle cases where all values are 0 (e.g. GK stats for Strikers)
            if subset[col].abs().sum() == 0:
                df.loc[family_mask, f'pct_{col}'] = 0
                df.loc[family_mask, f'z_{col}'] = 0
            else:
                df.loc[family_mask, f'pct_{col}'] = subset[col].rank(pct=True) * 100
                # Z-Score
                if subset[col].std() > 0:
                     df.loc[family_mask, f'z_{col}'] = (subset[col] - subset[col].mean()) / subset[col].std()
                else:
                     df.loc[family_mask, f'z_{col}'] = 0

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

    # Helper to load and tag
    def load_and_tag(path, competition):
        if not path or not os.path.exists(path): return None
        df = pd.read_csv(path)
        df['Competition'] = competition
        return df

    # Load Events
    if liga_ii_event_file:
        df = load_and_tag(liga_ii_event_file, 'Liga II')
        if df is not None:
            df = normalize_event_columns(df)
            data_store['events']['Liga II'] = df

    if superliga_event_file:
        df = load_and_tag(superliga_event_file, 'Superliga')
        if df is not None:
            df = normalize_event_columns(df)
            data_store['events']['Superliga'] = df

    # Load Stats (Repo Files)
    # Finding files dynamically or hardcoded
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

    # Combine and Process Player Stats
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

# Global Filters (Sidebar)
st.sidebar.header("Global Settings")

# Competition Selector
comps = []
if 'events' in data:
    comps.extend(list(data['events'].keys()))
if 'stats' in data and 'teams' in data['stats']:
    comps.extend(data['stats']['teams']['competitionId'].unique().tolist())
comps = list(set(comps))
# Clean up comp list
clean_comps = [c for c in comps if str(c) != 'nan']
if not clean_comps: clean_comps = ["Superliga", "Liga II"]

selected_comp = st.sidebar.selectbox("Select Competition", clean_comps, index=0)

# Allow user to upload extra files
uploaded_files = st.sidebar.file_uploader("Upload CSV Data", accept_multiple_files=True, type=['csv'])
if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        ftype = detect_file_type(df)
        if ftype == 'event':
            # Tag as 'Uploaded'
            df['Competition'] = 'Uploaded'
            df = normalize_event_columns(df)
            # Add to events list (might need a better structure for multi-file)
            data['events'][uploaded_file.name] = df
        elif ftype == 'stats':
            # We assume it's player stats for now, can be improved
            if 'playerId' in df.columns:
                processed = preprocess_stats(df, role_definitions)
                processed['Competition'] = 'Uploaded'
                if 'players' in data['stats']:
                    data['stats']['players'] = pd.concat([data['stats']['players'], processed], ignore_index=True)
                else:
                    data['stats']['players'] = processed
            else:
                st.warning(f"Uploaded stats file {uploaded_file.name} does not have 'playerId'.")

# Tabs
tab_team, tab_player, tab_advanced = st.tabs(["Team Analysis", "Player Analysis", "Advanced Models"])

# --- Tab B: Player Analysis ---
with tab_player:
    st.header("Player Analysis Hub")

    # Get Player Stats Data
    if 'stats' in data and 'players' in data['stats']:
        player_df = data['stats']['players']
        # Filter by Competition
        # Ensure compatible types for comparison
        player_df['Competition'] = player_df['Competition'].astype(str)
        comp_player_df = player_df[player_df['Competition'] == str(selected_comp)]

        # Team Selector
        teams_list = sorted(comp_player_df['team_name'].dropna().unique())
        selected_team_p = st.selectbox("Select Team", teams_list, key="p_team")

        # Position Family Selector
        # Filter players by team first
        team_players = comp_player_df[comp_player_df['team_name'] == selected_team_p]
        families = sorted(team_players['PositionFamily'].unique())
        selected_family = st.selectbox("Select Position Family", families)

        # Player Selector
        family_players = team_players[team_players['PositionFamily'] == selected_family]
        player_names = sorted(family_players['player_shortName'].unique())
        selected_player = st.selectbox("Select Player", player_names)

        if selected_player:
            player_stats = family_players[family_players['player_shortName'] == selected_player].iloc[0]

            # --- Visualizations ---
            col_pizza, col_heat = st.columns(2)

            with col_pizza:
                st.subheader("Percentile Pizza Plot")

                # Template Selector (Based on Family)
                # Load Role Defs
                if selected_family in role_definitions:
                    archetypes = role_definitions[selected_family]
                    selected_archetype_name = st.selectbox("Select Archetype Template", list(archetypes.keys()))

                    archetype = archetypes[selected_archetype_name]
                    metric_mapping = archetype['metrics'] # Dict: metric_name -> weight/unused? User said "Select... metrics".
                    # The JSON has values like 10, 8. These seem to be "Importance" or "Expected Value"?
                    # The prompt says: "Select using the json and then any other reasonable metric... 10-14 slices".
                    # I will simply take the KEYS of the 'metrics' dictionary as the list of params to plot.

                    params = list(metric_mapping.keys())

                    # Get values (Percentiles)
                    # We calculated pct_metric in preprocess_stats
                    # Check if they exist
                    values = []
                    valid_params = []

                    for p in params:
                        # Construct per90 name if it exists?
                        # In JSON, keys are like 'average_defensiveDuels'.
                        # In DF, we have 'per90_total_defensiveDuels' OR 'average_defensiveDuels'?
                        # Let's check DF columns.
                        # The repo files have columns like 'average_defensiveDuels'.
                        # My preprocessing created 'per90_total_...'.
                        # If the JSON keys match the repo columns (which start with 'average_' or 'percent_'), use them directly.
                        # BUT, I calculated 'pct_' for ALL numeric columns.
                        # So if JSON has 'average_defensiveDuels', I look for 'pct_average_defensiveDuels'.

                        pct_col = f"pct_{p}"
                        if pct_col in player_stats:
                            values.append(int(player_stats[pct_col])) # Integer for Pizza
                            valid_params.append(p.replace("average_", "").replace("percent_", "").replace("_", " ").title())
                        else:
                            # Try mapping 'average_' to 'per90_total_'?
                            # Example: JSON 'average_passes' -> DF 'average_passes'.
                            # If DF has 'average_passes', pct_average_passes exists.
                            # If DF only had total_passes, I made per90_total_passes.
                            # The repo CSVs ALREADY have average_ columns. So I should stick to those if JSON uses them.
                            # However, I should check if I calculated PCT for them.
                            # Yes, 'numeric_cols' included all numeric cols.
                            pass

                    if values:
                        # Slice colors based on percentile
                        slice_colors = []
                        for v in values:
                            if v >= 90: slice_colors.append("#2ca02c") # Green
                            elif v >= 60: slice_colors.append("#1f77b4") # Blue
                            else: slice_colors.append("#d62728") # Red

                        # Instantiate PyPizza
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
                        st.warning("No valid metrics found for this template in the dataset.")
                else:
                    st.info("No templates found for this Position Family.")

            with col_heat:
                st.subheader("Event Heatmap")
                # Need Event Data for this player
                if selected_comp in data['events']:
                    events_df = data['events'][selected_comp]
                    # Filter by Player ID
                    # Player Stats has 'playerId' (WyId).
                    pid = player_stats['playerId']
                    player_events = events_df[events_df['playerId'] == pid]

                    if not player_events.empty:
                         pitch = Pitch(pitch_type='wyscout', line_zorder=2, pitch_color='#222222', line_color='#efefef')
                         fig_h, ax_h = pitch.draw(figsize=(10, 7))

                         # KDE Plot
                         # Use seaborn or mplsoccer kdeplot
                         # Filter only valid x, y
                         valid_ev = player_events.dropna(subset=['x', 'y'])
                         if not valid_ev.empty:
                             pitch.kdeplot(valid_ev.x, valid_ev.y, ax=ax_h, cmap='hot', fill=True, levels=100, alpha=0.6)
                             st.pyplot(fig_h)
                         else:
                             st.write("No location data for events.")
                    else:
                        st.write("No events found for this player.")
                else:
                    st.write("Event data not loaded.")

            # --- Head to Head ---
            st.subheader("Head-to-Head Comparison")
            # Select Opponent
            opp_player_name = st.selectbox("Select Opponent Player", ["None"] + list(player_names), index=0, key='opp_p')

            if opp_player_name != "None":
                 opp_stats = family_players[family_players['player_shortName'] == opp_player_name].iloc[0]

                 st.write("### Head-to-Head Comparison")

                 # 1. Side-by-Side Pizza Plots
                 # We need to replicate the Pizza Logic for both players
                 if selected_family in role_definitions:
                    # Reuse archetype selection from above or force same?
                    # Usually usage implies same archetype for comparison.
                    # 'archetype' and 'valid_params' are defined in the previous block.
                    # We should check if they are available.

                    if 'valid_params' in locals() and valid_params:
                        # Get Values for P1 (already got 'values' list)
                        values_p1 = values

                        # Get Values for P2
                        values_p2 = []
                        for p in params:
                            pct_col = f"pct_{p}"
                            if pct_col in opp_stats:
                                values_p2.append(int(opp_stats[pct_col]))
                            else:
                                values_p2.append(0)

                        # Plotting Helper
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

                 # Compare the metrics from the Template
                 comp_rows = []
                 for p in params:
                     val_a = player_stats.get(p, 0)
                     val_b = opp_stats.get(p, 0)

                     # Check if inverse
                     # Heuristic: metrics with "Fouls", "Losses", "Error" might be inverse?
                     # Standard logic: Higher is Green.
                     # Exceptions: 'average_fouls', 'average_ownHalfLosses' (from JSON sample)
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

                 # Render HTML Table
                 html_p = f"<table><thead><tr><th>Metric</th><th>{selected_player}</th><th>{opp_player_name}</th></tr></thead><tbody>"
                 for r in comp_rows:
                     html_p += f"<tr><td>{r['Metric']}</td>"
                     html_p += f"<td style='background-color:{r['ca']}; color:black'>{r[selected_player]:.2f}</td>"
                     html_p += f"<td style='background-color:{r['cb']}; color:black'>{r[opp_player_name]:.2f}</td></tr>"
                 html_p += "</tbody></table>"
                 st.markdown(html_p, unsafe_allow_html=True)
        else:
            st.info("Select a player.")
    else:
        st.warning("Player stats not available.")

# --- Tab C: Advanced Models ---
with tab_advanced:
    st.header("Advanced Models")

    subtab_cluster, subtab_xt = st.tabs(["Clustering (K-Means)", "Expected Threat (xT)"])

    with subtab_cluster:
        st.subheader("Player Clustering")

        if 'stats' in data and 'players' in data['stats']:
            player_df = data['stats']['players']

            # Select Position Family to Cluster
            families_c = sorted(player_df['PositionFamily'].unique())
            sel_fam_c = st.selectbox("Select Position Family for Clustering", families_c)

            # Filter Data
            cluster_data = player_df[player_df['PositionFamily'] == sel_fam_c].copy()

            # Select Features
            # Default to some numeric columns
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns.tolist()
            # Filter out IDs and Pct/Z cols
            feature_cols = [c for c in numeric_cols if not c.startswith('pct_') and not c.startswith('z_') and 'Id' not in c and 'percent' not in c]

            # Let user select
            selected_features = st.multiselect("Select Features for Clustering", feature_cols, default=feature_cols[:5])

            if len(selected_features) > 1:
                # Prepare Data
                X = cluster_data[selected_features].fillna(0)

                # Standardize
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # PCA
                pca = PCA(n_components=2)
                pca_res = pca.fit_transform(X_scaled)

                # K-Means
                k = st.slider("Select Number of Clusters (K)", 2, 10, 4)
                kmeans = KMeans(n_clusters=k, random_state=42)
                clusters = kmeans.fit_predict(X_scaled)

                # Plot
                cluster_data['PCA1'] = pca_res[:, 0]
                cluster_data['PCA2'] = pca_res[:, 1]
                cluster_data['Cluster'] = clusters

                # Use Seaborn/Matplotlib
                fig_c, ax_c = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=cluster_data, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', ax=ax_c, s=100)

                # Annotate some players? Or just hover (Streamlit native scatter might be better for hover)
                # Let's use st.scatter_chart or Altair?
                # Or just simple maplotlib with some labels for top players

                # Let's use st.scatter_chart (simple) or custom Altair for hover
                # Or Plotly if installed? No plotly in plan.
                # Just Matplotlib is fine.

                st.pyplot(fig_c)

                st.write("Cluster Members:")
                st.dataframe(cluster_data[['player_shortName', 'team_name', 'Cluster'] + selected_features])

            else:
                st.warning("Select at least 2 features.")
        else:
            st.warning("No player data.")

    with subtab_xt:
        st.subheader("Expected Threat (xT) - Simplified Grid")

        # Simplified xT Matrix (12x16 grid)
        # Increasing value towards goal (right side)
        # Create a gradient matrix
        xT_rows, xT_cols = 12, 16
        xT_grid = np.zeros((xT_rows, xT_cols))

        for r in range(xT_rows):
            for c in range(xT_cols):
                # Simple logic: closer to center y, closer to end x -> higher value
                # Normalize x (0-1), y (0-0.5-0)
                x_val = c / xT_cols
                y_val = 1 - abs((r - xT_rows/2) / (xT_rows/2))

                # xT usually low in back, high in half-spaces/zone 14
                val = (x_val ** 2) * 0.1 + (x_val * y_val * 0.05)
                xT_grid[r, c] = val

        # Visualize Grid
        st.write("Visualizing High Threat Zones")
        pitch = Pitch(pitch_type='wyscout', line_zorder=2, pitch_color='#222222', line_color='#efefef')
        fig_xt, ax_xt = pitch.draw(figsize=(10, 7))

        # Heatmap of the Grid
        # Pitch dimensions: 105x68 usually. Wyscout is 100x100
        # mplsoccer handles binning if we used bin_statistic, but here we have a raw grid.
        # We can plot it using imshow if we map it to extent.

        ax_xt.imshow(xT_grid, extent=(0, 100, 0, 100), cmap='magma', alpha=0.6, origin='lower')

        st.pyplot(fig_xt)

        st.info("This is a simplified gradient-based xT model. Real xT requires training on large datasets.")

# --- Tab A: Team Analysis ---
with tab_team:
    st.header("Team Analysis Hub")

    # Filter Teams based on global selected_comp
    team_df = data['stats']['teams'] if 'teams' in data['stats'] else pd.DataFrame()

    if not team_df.empty:
        # Ensure compatible comparison
        team_df['Competition'] = team_df['Competition'].astype(str)
        available_teams = team_df[team_df['Competition'] == str(selected_comp)]['team_name'].sort_values().unique()
    else:
        available_teams = []

    col_filters, col_viz = st.columns([1, 2])

    with col_filters:
        team_a = st.selectbox("Select Team A", available_teams)
        team_b = st.selectbox("Select Team B (Comparison)", ["None"] + list(available_teams))

        # Match Selector (Requires Event Data)
        match_options = []
        if selected_comp in data['events']:
            events_df = data['events'][selected_comp]
            # Filter for Team A events
            # Need to map Team Name to Team ID or just use team name if available
            # Event data usually uses teamId. We need a mapping.
            # We can create a mapping from team_df

            # Create mapping: team_name -> teamId
            if not team_df.empty:
                # Get the row for Team A
                team_info = team_df[(team_df['team_name'] == team_a) & (team_df['Competition'] == selected_comp)]
                if not team_info.empty:
                    team_a_id = team_info.iloc[0]['teamId']

                    # Filter events
                    team_events = events_df[events_df['teamId'] == team_a_id]
                    match_ids = team_events['matchId'].unique()
                    match_options = sorted(match_ids)

        selected_match = st.selectbox("Select Match (ID)", ["All"] + list(match_options))

        # Comparison Table
        if team_b != "None" and not team_df.empty:
            st.subheader("Head-to-Head Stats")

            # Metrics to compare
            metrics = {
                'Goals': 'average_goals',
                'xG': 'average_xgShot',
                'Possession %': 'average_possessionPercent',
                'Shots': 'average_shots',
                'PPDA': 'total_ppda', # Lower is better usually (intensity)
                'Fouls': 'average_fouls', # Lower is better
                'Goals Conceded': 'average_concededGoals' # Lower is better
            }

            # Lower is Green metrics
            inverse_metrics = ['total_ppda', 'average_fouls', 'average_concededGoals']

            row_a = team_df[(team_df['team_name'] == team_a) & (team_df['Competition'] == selected_comp)].iloc[0]
            row_b = team_df[(team_df['team_name'] == team_b) & (team_df['Competition'] == selected_comp)].iloc[0]

            comp_data = []
            for label, col in metrics.items():
                val_a = row_a.get(col, 0)
                val_b = row_b.get(col, 0)

                # Determine colors
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

            # Custom HTML Table for coloring
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

            # Filter for Match and Team A
            # Re-get ID
            team_info = team_df[(team_df['team_name'] == team_a) & (team_df['Competition'] == selected_comp)]
            if not team_info.empty:
                team_a_id = team_info.iloc[0]['teamId']
                match_events = events_df[(events_df['matchId'] == selected_match) & (events_df['teamId'] == team_a_id)]

                # 1. Pass Map
                st.markdown("### Pass Map")
                # Filter passes
                passes = match_events[match_events['subEventName'].str.contains('pass', case=False, na=False)]

                # Setup Pitch
                pitch = Pitch(pitch_type='wyscout', pitch_color='#222222', line_color='#c7d5cc')
                fig, ax = pitch.draw(figsize=(10, 7))

                # Draw Successful
                # Need column for success/accuracy. Wyscout tags? or 'tags' column?
                # Tags: {id: 1801} is accurate pass.
                # Let's check columns. Usually there's tags.
                # If we can't parse tags easily here, we might just plot all arrows.
                # Assuming 'tags' column exists and is a list of dicts or string.
                # Simplification: Plot all passes for now, color by 'id' if possible or just blue.

                # Attempt to find success tag (1801)
                # Wyscout CSV usually has a 'tags' column which is a string like "[{'id': 1801}]"

                passes['is_successful'] = passes['tags'].astype(str).str.contains('1801')

                succ_passes = passes[passes['is_successful']]
                unsucc_passes = passes[~passes['is_successful']]

                pitch.arrows(succ_passes.x, succ_passes.y, succ_passes.end_x, succ_passes.end_y, ax=ax, color='green', width=2, headwidth=3, alpha=0.6, label='Successful')
                pitch.arrows(unsucc_passes.x, unsucc_passes.y, unsucc_passes.end_x, unsucc_passes.end_y, ax=ax, color='red', width=2, headwidth=3, alpha=0.6, label='Unsuccessful')

                ax.legend(facecolor='#222222', edgecolor='None', labelcolor='white')
                st.pyplot(fig)

                # 2. Shot Map
                st.markdown("### Shot Map")
                shots = match_events[match_events['subEventName'] == 'Shot']
                goals = shots[shots['tags'].astype(str).str.contains('101')] # Tag 101 is Goal

                fig2, ax2 = pitch.draw(figsize=(10, 7))

                # Check for xG column ('xgShot' in some Wyscout versions, or need to calculate/mock)
                # If available, use it for size.
                # The prompt mentions "Plot shot locations with size determined by xG (if available)".
                # Wyscout Event CSV standard V2 usually doesn't have xG directly unless enriched.
                # However, the repo *Stats* file has total_xgShot.
                # If the EVENT file has xG, we use it. If not, we default size.
                # Let's check columns defensively.

                if 'xg' in shots.columns:
                    sizes = shots['xg'] * 500
                elif 'xgShot' in shots.columns:
                     sizes = shots['xgShot'] * 500
                else:
                    sizes = 100 # Default size

                # Plot all shots
                pitch.scatter(shots.x, shots.y, ax=ax2, color='blue', edgecolors='white', s=sizes, alpha=0.7, label='Shot')
                # Plot goals
                pitch.scatter(goals.x, goals.y, ax=ax2, color='green', edgecolors='gold', s=sizes if isinstance(sizes, int) else 200, marker='*', label='Goal')

                ax2.legend(facecolor='#222222', edgecolor='None', labelcolor='white')
                st.pyplot(fig2)

        elif selected_match == "All":
             st.info("Select a specific match to view Pass Maps and Shot Maps.")
        else:
             st.warning("No Event Data available for this competition.")
