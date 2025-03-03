import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import cv2
import tempfile
import os
import json
from datetime import datetime

# ---- Dashboard Configuration ----
st.set_page_config(
    page_title="Football Analysis System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .highlight {
        color: #EF4444;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---- Helper Functions ----
def load_match_data(match_id):
    """Load player and match data from processed files"""
    try:
        # In production, this would load from a database or API
        # For now, we'll simulate with sample data
        player_data = pd.read_csv(f"data/processed/{match_id}/player_stats.csv")
        match_data = json.load(open(f"data/processed/{match_id}/match_info.json"))
        return player_data, match_data
    except:
        # Use sample data if files don't exist
        return generate_sample_data()

def generate_sample_data():
    """Generate sample football match data for demonstration"""
    # Create sample player data
    players_home = [f"Player H{i}" for i in range(1, 12)]
    players_away = [f"Player A{i}" for i in range(1, 12)]
    
    player_data = pd.DataFrame({
        'player_id': list(range(1, 23)),
        'name': players_home + players_away,
        'team': ['Home'] * 11 + ['Away'] * 11,
        'position': ['GK', 'DEF', 'DEF', 'DEF', 'DEF', 'MID', 'MID', 'MID', 'FWD', 'FWD', 'FWD'] * 2,
        'distance_covered': np.random.normal(8.5, 1.5, 22),
        'top_speed': np.random.normal(32, 3, 22),
        'sprints': np.random.randint(10, 30, 22),
        'passes_completed': np.random.randint(20, 70, 22),
        'pass_accuracy': np.random.normal(75, 10, 22),
        'shots': np.random.randint(0, 6, 22),
        'xG': np.random.normal(0.3, 0.5, 22).round(2),
        'goals': np.random.randint(0, 2, 22),
        'tackles': np.random.randint(0, 8, 22),
        'interceptions': np.random.randint(0, 10, 22),
        'heat_map_data': [json.dumps(np.random.rand(10, 7).tolist()) for _ in range(22)]
    })
    
    # Create sample match data
    match_data = {
        'match_id': 'SAMPLE001',
        'home_team': 'Home FC',
        'away_team': 'Away United',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'venue': 'Sample Stadium',
        'score': {'home': 2, 'away': 1},
        'possession': {'home': 58, 'away': 42},
        'shots': {'home': 15, 'away': 8},
        'shots_on_target': {'home': 7, 'away': 3},
        'corners': {'home': 6, 'away': 4},
        'fouls': {'home': 10, 'away': 14},
        'formation': {'home': '4-3-3', 'away': '4-4-2'},
        'events': [
            {'time': '23', 'team': 'home', 'type': 'goal', 'player': 'Player H9', 'xG': 0.72},
            {'time': '45+2', 'team': 'away', 'type': 'goal', 'player': 'Player A11', 'xG': 0.34},
            {'time': '67', 'team': 'home', 'type': 'goal', 'player': 'Player H10', 'xG': 0.56}
        ]
    }
    
    return player_data, match_data

def create_pitch_visualization(match_data, player_data, selected_metric):
    """Create a football pitch visualization with player positions and selected metrics"""
    # Create a blank pitch image
    pitch = np.ones((700, 1000, 3), dtype=np.uint8) * 255
    
    # Draw pitch lines
    pitch = cv2.rectangle(pitch, (50, 50), (950, 650), (0, 128, 0), -1)  # Green pitch
    pitch = cv2.rectangle(pitch, (50, 50), (950, 650), (255, 255, 255), 2)  # Outer line
    pitch = cv2.rectangle(pitch, (50, 225), (200, 475), (255, 255, 255), 2)  # Left penalty area
    pitch = cv2.rectangle(pitch, (800, 225), (950, 475), (255, 255, 255), 2)  # Right penalty area
    pitch = cv2.circle(pitch, (500, 350), 100, (255, 255, 255), 2)  # Center circle
    pitch = cv2.line(pitch, (500, 50), (500, 650), (255, 255, 255), 2)  # Middle line
    
    # Convert to PIL Image for Streamlit
    img = Image.fromarray(pitch)
    
    # In a real implementation, we would place dots for players based on their actual tracking data
    # For the sample, we'll use a simplified formation-based positioning
    
    return img

def generate_heatmap(player_id, player_data):
    """Generate a heatmap for a player's positions during the match"""
    # In a real implementation, this would use actual position data
    # For the sample, we'll create a random heatmap
    player_row = player_data[player_data['player_id'] == player_id]
    if not player_row.empty:
        heat_data = json.loads(player_row['heat_map_data'].values[0])
        fig = px.imshow(heat_data, 
                        labels=dict(x="Pitch Width", y="Pitch Length", color="Presence"),
                        x=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%'],
                        y=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
                        color_continuous_scale="Viridis")
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        return fig
    return None

def create_player_comparison(player_data, selected_players, metric):
    """Create a bar chart comparing selected players on a specific metric"""
    if not selected_players:
        return None
    
    filtered_data = player_data[player_data['name'].isin(selected_players)]
    fig = px.bar(filtered_data, x='name', y=metric, color='team',
                title=f"Player Comparison: {metric}",
                labels={'name': 'Player', metric: metric.replace('_', ' ').title()})
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def visualize_events(match_data):
    """Create a timeline visualization of match events"""
    events = match_data.get('events', [])
    if not events:
        return None
    
    df_events = pd.DataFrame(events)
    
    # Create a timeline
    fig = go.Figure()
    
    # Add events to timeline
    for i, event in enumerate(events):
        team_color = 'blue' if event['team'] == 'home' else 'red'
        marker_symbol = 'circle' if event['type'] == 'goal' else 'square'
        
        fig.add_trace(go.Scatter(
            x=[int(event['time'])], 
            y=[1],
            mode='markers+text',
            marker=dict(symbol=marker_symbol, size=15, color=team_color),
            text=[f"{event['player']}"],
            textposition="top center",
            name=f"{event['type']} - {event['player']}",
            hoverinfo='text',
            hovertext=f"Time: {event['time']}'\nPlayer: {event['player']}\nType: {event['type']}\nxG: {event.get('xG', 'N/A')}"
        ))
    
    # Configure layout
    fig.update_layout(
        title="Match Timeline",
        xaxis=dict(title="Minute", range=[0, 95]),
        yaxis=dict(showticklabels=False),
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig

# ---- Main Application ----
def main():
    # Sidebar for match selection and filters
    st.sidebar.title("⚽ Football Analysis")
    
    # Match selection (in a real app, this would be from a database)
    match_options = ["Sample Match", "Match 2", "Match 3"]
    selected_match = st.sidebar.selectbox("Select Match", match_options)
    
    # Load match data
    player_data, match_data = load_match_data("SAMPLE001")
    
    # Analysis mode selection
    analysis_mode = st.sidebar.radio("Analysis Mode", 
                                    ["Match Overview", "Player Analysis", "Team Statistics", "Event Analysis"])
    
    # Additional filters
    if analysis_mode == "Player Analysis":
        selected_team = st.sidebar.selectbox("Select Team", ["All", "Home", "Away"])
        available_players = player_data['name'].tolist() if selected_team == "All" else \
                            player_data[player_data['team'] == selected_team]['name'].tolist()
        selected_player = st.sidebar.selectbox("Select Player", available_players)
        
        metrics = ['distance_covered', 'top_speed', 'sprints', 'passes_completed', 
                  'pass_accuracy', 'shots', 'xG', 'goals', 'tackles', 'interceptions']
        selected_metrics = st.sidebar.multiselect("Select Metrics", metrics, default=['distance_covered', 'top_speed'])
    
    elif analysis_mode == "Team Statistics":
        selected_metric = st.sidebar.selectbox("Select Metric", 
                                              ['Formation', 'Possession', 'Distance Covered', 
                                               'Pass Completion', 'Defensive Pressure'])
    
    # Main content area
    st.markdown("<div class='main-header'>Football Match Analysis System</div>", unsafe_allow_html=True)
    
    # Match header info
    col1, col2, col3 = st.columns([2,3,2])
    with col1:
        st.subheader(match_data.get('home_team', 'Home Team'))
    with col2:
        st.markdown(f"<h2 style='text-align: center;'>{match_data.get('score', {}).get('home', 0)} - {match_data.get('score', {}).get('away', 0)}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{match_data.get('date', 'Date')} | {match_data.get('venue', 'Venue')}</p>", unsafe_allow_html=True)
    with col3:
        st.subheader(match_data.get('away_team', 'Away Team'))
    
    # Display content based on selected mode
    if analysis_mode == "Match Overview":
        # Match summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Possession", f"{match_data.get('possession', {}).get('home', 0)}% - {match_data.get('possession', {}).get('away', 0)}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Shots", f"{match_data.get('shots', {}).get('home', 0)} - {match_data.get('shots', {}).get('away', 0)}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Shots on Target", f"{match_data.get('shots_on_target', {}).get('home', 0)} - {match_data.get('shots_on_target', {}).get('away', 0)}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Corners", f"{match_data.get('corners', {}).get('home', 0)} - {match_data.get('corners', {}).get('away', 0)}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Formation visualization
        st.markdown("<div class='sub-header'>Team Formations</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{match_data.get('home_team')}**: {match_data.get('formation', {}).get('home', 'N/A')}")
            pitch_viz = create_pitch_visualization(match_data, player_data, 'home')
            st.image(pitch_viz, caption="Home Team Formation", use_column_width=True)
        
        with col2:
            st.write(f"**{match_data.get('away_team')}**: {match_data.get('formation', {}).get('away', 'N/A')}")
            pitch_viz = create_pitch_visualization(match_data, player_data, 'away')
            st.image(pitch_viz, caption="Away Team Formation", use_column_width=True)
        
        # Events timeline
        st.markdown("<div class='sub-header'>Match Timeline</div>", unsafe_allow_html=True)
        timeline = visualize_events(match_data)
        if timeline:
            st.plotly_chart(timeline, use_container_width=True)
        
        # Team stats comparison
        st.markdown("<div class='sub-header'>Team Performance Comparison</div>", unsafe_allow_html=True)
        # Calculate team totals from player data
        home_players = player_data[player_data['team'] == 'Home']
        away_players = player_data[player_data['team'] == 'Away']
        
        team_stats = pd.DataFrame({
            'Metric': ['Distance (km)', 'Passes', 'Pass Accuracy (%)', 'Shots', 'Tackles', 'Interceptions'],
            'Home': [
                home_players['distance_covered'].sum().round(1),
                home_players['passes_completed'].sum(),
                home_players['pass_accuracy'].mean().round(1),
                home_players['shots'].sum(),
                home_players['tackles'].sum(),
                home_players['interceptions'].sum()
            ],
            'Away': [
                away_players['distance_covered'].sum().round(1),
                away_players['passes_completed'].sum(),
                away_players['pass_accuracy'].mean().round(1),
                away_players['shots'].sum(),
                away_players['tackles'].sum(),
                away_players['interceptions'].sum()
            ]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=team_stats['Metric'],
            x=team_stats['Home'],
            name=match_data.get('home_team', 'Home'),
            orientation='h',
            marker=dict(color='blue')
        ))
        fig.add_trace(go.Bar(
            y=team_stats['Metric'],
            x=team_stats['Away'],
            name=match_data.get('away_team', 'Away'),
            orientation='h',
            marker=dict(color='red')
        ))
        
        fig.update_layout(
            barmode='group',
            title='Team Performance Metrics',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_mode == "Player Analysis":
        # Filter for selected player
        player = player_data[player_data['name'] == selected_player].iloc[0]
        
        # Player header
        st.markdown(f"<div class='sub-header'>{player['name']} ({player['position']}) - {player['team']} Team</div>", unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Distance", f"{player['distance_covered']:.1f} km")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Top Speed", f"{player['top_speed']:.1f} km/h")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Pass Accuracy", f"{player['pass_accuracy']:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Goals/xG", f"{player['goals']}/{player['xG']:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Player heatmap
        st.markdown("<div class='sub-header'>Player Heatmap</div>", unsafe_allow_html=True)
        heatmap = generate_heatmap(player['player_id'], player_data)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)
        
        # Player comparison
        st.markdown("<div class='sub-header'>Player Comparison</div>", unsafe_allow_html=True)
        comparison_players = st.multiselect("Select players to compare with", 
                                           player_data[player_data['name'] != selected_player]['name'].tolist(),
                                           max_selections=3)
        
        if comparison_players:
            all_selected = [selected_player] + comparison_players
            for metric in selected_metrics:
                comparison_chart = create_player_comparison(player_data, all_selected, metric)
                if comparison_chart:
                    st.plotly_chart(comparison_chart, use_container_width=True)
        
    elif analysis_mode == "Team Statistics":
        st.markdown("<div class='sub-header'>Team Performance Analysis</div>", unsafe_allow_html=True)
        
        if selected_metric == "Formation":
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{match_data.get('home_team')} Formation:** {match_data.get('formation', {}).get('home', 'N/A')}")
                pitch_viz = create_pitch_visualization(match_data, player_data, 'home')
                st.image(pitch_viz, caption="Home Team Formation", use_column_width=True)
            
            with col2:
                st.write(f"**{match_data.get('away_team')} Formation:** {match_data.get('formation', {}).get('away', 'N/A')}")
                pitch_viz = create_pitch_visualization(match_data, player_data, 'away')
                st.image(pitch_viz, caption="Away Team Formation", use_column_width=True)
                
        elif selected_metric == "Possession":
            # Possession breakdown
            possession_data = {
                'Team': [match_data.get('home_team', 'Home'), match_data.get('away_team', 'Away')],
                'Possession': [match_data.get('possession', {}).get('home', 50), 
                              match_data.get('possession', {}).get('away', 50)]
            }
            
            fig = px.pie(possession_data, values='Possession', names='Team', 
                        title='Ball Possession', color='Team',
                        color_discrete_map={match_data.get('home_team', 'Home'): 'blue', 
                                          match_data.get('away_team', 'Away'): 'red'})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Zone-based possession (simulated data)
            st.subheader("Possession by Pitch Zone")
            zone_data = {
                'Zone': ['Defensive Third', 'Middle Third', 'Attacking Third'],
                match_data.get('home_team', 'Home'): [20, 25, 13],
                match_data.get('away_team', 'Away'): [15, 18, 9]
            }
            
            df_zone = pd.DataFrame(zone_data)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_zone['Zone'],
                x=df_zone[match_data.get('home_team', 'Home')],
                name=match_data.get('home_team', 'Home'),
                orientation='h',
                marker=dict(color='blue')
            ))
            fig.add_trace(go.Bar(
                y=df_zone['Zone'],
                x=df_zone[match_data.get('away_team', 'Away')],
                name=match_data.get('away_team', 'Away'),
                orientation='h',
                marker=dict(color='red')
            ))
            
            fig.update_layout(barmode='group', title='Possession by Pitch Zone (in seconds)', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        elif selected_metric == "Distance Covered":
            # Team distance covered
            home_distance = player_data[player_data['team'] == 'Home']['distance_covered'].sum()
            away_distance = player_data[player_data['team'] == 'Away']['distance_covered'].sum()
            
            distance_data = {
                'Team': [match_data.get('home_team', 'Home'), match_data.get('away_team', 'Away')],
                'Distance (km)': [home_distance, away_distance]
            }
            
            fig = px.bar(distance_data, x='Team', y='Distance (km)', 
                        title='Total Distance Covered by Team',
                        color='Team', 
                        color_discrete_map={match_data.get('home_team', 'Home'): 'blue', 
                                          match_data.get('away_team', 'Away'): 'red'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Player distance breakdown
            player_distance = player_data.sort_values('distance_covered', ascending=False)
            fig = px.bar(player_distance, x='name', y='distance_covered', color='team',
                        title='Distance Covered by Each Player',
                        labels={'name': 'Player', 'distance_covered': 'Distance (km)'},
                        height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        elif selected_metric == "Pass Completion":
            # Team passing stats
            home_passes = player_data[player_data['team'] == 'Home']['passes_completed'].sum()
            away_passes = player_data[player_data['team'] == 'Away']['passes_completed'].sum()
            home_accuracy = player_data[player_data['team'] == 'Home']['pass_accuracy'].mean()
            away_accuracy = player_data[player_data['team'] == 'Away']['pass_accuracy'].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Total Passes", f"{home_passes} vs {away_passes}", 
                         f"{home_passes - away_passes:+} {match_data.get('home_team', 'Home')}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Pass Accuracy", f"{home_accuracy:.1f}% vs {away_accuracy:.1f}%", 
                         f"{home_accuracy - away_accuracy:+.1f}% {match_data.get('home_team', 'Home')}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Player passing comparison
            player_passing = player_data.sort_values('passes_completed', ascending=False)
            fig = px.scatter(player_passing, x='passes_completed', y='pass_accuracy', 
                            color='team', hover_name='name', size='distance_covered',
                            labels={'passes_completed': 'Passes Completed', 
                                   'pass_accuracy': 'Pass Accuracy (%)',
                                   'distance_covered': 'Distance Covered (km)'},
                            title='Player Passing Performance')
            st.plotly_chart(fig, use_container_width=True)
        
        elif selected_metric == "Defensive Pressure":
            # Defensive stats by team
            home_tackles = player_data[player_data['team'] == 'Home']['tackles'].sum()
            away_tackles = player_data[player_data['team'] == 'Away']['tackles'].sum()
            home_interceptions = player_data[player_data['team'] == 'Home']['interceptions'].sum()
            away_interceptions = player_data[player_data['team'] == 'Away']['interceptions'].sum()
            
            defense_data = pd.DataFrame({
                'Metric': ['Tackles', 'Interceptions'],
                match_data.get('home_team', 'Home'): [home_tackles, home_interceptions],
                match_data.get('away_team', 'Away'): [away_tackles, away_interceptions]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=defense_data['Metric'],
                y=defense_data[match_data.get('home_team', 'Home')],
                name=match_data.get('home_team', 'Home'),
                marker=dict(color='blue')
            ))
            fig.add_trace(go.Bar(
                x=defense_data['Metric'],
                y=defense_data[match_data.get('away_team', 'Away')],
                name=match_data.get('away_team', 'Away'),
                marker=dict(color='red')
            ))
            
            fig.update_layout(barmode='group', title='Defensive Actions by Team')
            st.plotly_chart(fig, use_container_width=True)
            
            # Top defensive players
            defensive_metrics = player_data.copy()
            defensive_metrics['defensive_actions'] = defensive_metrics['tackles'] + defensive_metrics['interceptions']
            top_defenders = defensive_metrics.sort_values('defensive_actions', ascending=False).head(10)
            
            fig = px.bar(top_defenders, x='name', y=['tackles', 'interceptions'], 
                        title='Top 10 Defensive Players',
                        labels={'name': 'Player', 'value': 'Actions', 'variable': 'Type'},
                        barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_mode == "Event Analysis":
        st.markdown("<div class='sub-header'>Match Events Analysis</div>", unsafe_allow_html=True)
        
        # Goals and xG timeline
        events = match_data.get('events', [])
        if events:
            df_events = pd.DataFrame(events)
            
            # Filter for goals
            goals = df_events[df_events['type'] == 'goal']
            
            # Create xG timeline
            fig = go.Figure()
            
            # Add home team goals
            home_goals = goals[goals['team'] == 'home']
            fig.add_trace(go.Scatter(
                x=home_goals['time'].astype(int),
                y=home_goals['xG'].astype(float),
                mode='markers',
                name=f"{match_data.get('home_team', 'Home')} Goals",
                marker=dict(symbol='circle', size=15, color='blue'),
                text=home_goals['player'],
                hoverinfo='text',
                hovertext=[f"Time: {row['time']}'\nPlayer: {row['player']}\nxG: {row['xG']}" 
                          for _, row in home_goals.iterrows()]
            ))
            
            # Add away team goals
            away_goals = goals[goals['team'] == 'away']
            fig.add_trace(go.Scatter(
                x=away_goals['time'].astype(int),
                y=away_goals['xG'].astype(float),
                mode='markers',
                name=f"{match_data.get('away_team', 'Away')} Goals",
                marker=dict(symbol='circle', size=15, color='red'),
                text=away_goals['player'],
                hoverinfo='text',
                hovertext=[f"Time: {row['time']}'\nPlayer: {row['player']}\nxG: {row['xG']}" 
                          for _, row in away_goals.iterrows()]
            ))
            
            # Configure layout
            fig.update_layout(
                title="Goals and Expected Goals (xG)",
                xaxis=dict(title="Match Minute", range=[0, 95]),
                yaxis=dict(title="Expected Goals (xG)", range=[0, 1]),
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Shot map
            st.markdown("<div class='sub-header'>Shot Map</div>", unsafe_allow_html=True)
            st.write("In a complete implementation, this would show actual shot locations on the pitch.")
            
            # Create a pitch visualization for shot map (simplified version)
            pitch = np.ones((700, 1000, 3), dtype=np.uint8) * 255
            pitch = cv2.rectangle(pitch, (50, 50), (950, 650), (0, 128, 0), -1)  # Green pitch
            pitch = cv2.rectangle(pitch, (50, 50), (950, 650), (255, 255, 255), 2)  # Outer line
            pitch = cv2.rectangle(pitch, (50, 225), (200, 475), (255, 255, 255), 2)  # Left penalty area
            pitch = cv2.rectangle(pitch, (800, 225), (950, 475), (255, 255, 255), 2)  # Right penalty area
            pitch = cv2.circle(pitch, (500, 350), 100, (255, 255, 255), 2)  # Center circle
            pitch = cv2.line(pitch, (500, 50), (500, 650), (255, 255, 255), 2)  # Middle line
            
            # Add sample shots (this would use real coordinates in production)
            # Home team shots (blue)
            for i in range(match_data.get('shots', {}).get('home', 0)):
                x = np.random.randint(500, 900)
                y = np.random.randint(200, 500)
                size = np.random.randint(5, 15)  # Size based on xG
                pitch = cv2.circle(pitch, (x, y), size, (255, 0, 0), -1)  # Blue for home team
            
            # Away team shots (red)
            for i in range(match_data.get('shots', {}).get('away', 0)):
                x = np.random.randint(100, 500)
                y = np.random.randint(200, 500)
                size = np.random.randint(5, 15)  # Size based on xG
                pitch = cv2.circle(pitch, (x, y), size, (0, 0, 255), -1)  # Red for away team
            
            # Convert to PIL Image for Streamlit
            img = Image.fromarray(pitch)
            st.image(img, caption="Shot Map (Circle size represents xG)", use_column_width=True)
            
            # Legend
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<span style='color:blue'>●</span> {match_data.get('home_team', 'Home')} - {match_data.get('shots', {}).get('home', 0)} shots", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<span style='color:red'>●</span> {match_data.get('away_team', 'Away')} - {match_data.get('shots', {}).get('away', 0)} shots", unsafe_allow_html=True)
        
        else:
            st.write("No event data available for this match.")

if __name__ == "__main__":
    main()