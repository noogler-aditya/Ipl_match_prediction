import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configuration
st.set_page_config(
    page_title="IPL Match Prediction",
    page_icon="🏏",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    matches_df = pd.read_csv('matches.csv')
    deliveries_df = pd.read_csv('deliveries.csv')
    return matches_df, deliveries_df

# Load models and encoders
@st.cache_resource
def load_models():
    model = joblib.load('models/ipl_prediction_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    team_encoder = joblib.load('models/team_encoder.pkl')
    venue_encoder = joblib.load('models/venue_encoder.pkl')
    city_encoder = joblib.load('models/city_encoder.pkl')
    team_stats = joblib.load('models/team_stats.pkl')
    return model, scaler, team_encoder, venue_encoder, city_encoder, team_stats

# Check if models exist, if not, train them
if not os.path.exists('models/ipl_prediction_model.pkl'):
    st.warning("Models not found. Training models now...")
    import subprocess
    subprocess.run(['python3', 'train_model.py'])
    st.success("Models trained successfully!")

# Load data and models
try:
    matches_df, deliveries_df = load_data()
    model, scaler, team_encoder, venue_encoder, city_encoder, team_stats = load_models()
    teams = sorted(team_encoder.classes_)
    venues = sorted(venue_encoder.classes_)
    cities = sorted([city for city in city_encoder.classes_ if isinstance(city, str)])
except Exception as e:
    st.error(f"Error loading data or models: {e}")
    st.stop()

# App title and description
st.title("🏏 IPL Match Prediction and Analysis")
st.markdown("""
This app predicts the outcome of IPL matches and provides team analysis based on historical data.
""")

# Create tabs
tab1, tab2 = st.tabs(["Match Prediction", "Team Analysis"])

# Tab 1: Match Prediction
with tab1:
    st.header("Predict Match Outcome")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("Select Team 1 (Home Team)", teams)
        venue = st.selectbox("Select Venue", venues)
        city = st.selectbox("Select City", cities)
    
    with col2:
        team2 = st.selectbox("Select Team 2 (Away Team)", teams)
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
    
    if st.button("Predict Match Outcome"):
        if team1 == team2:
            st.error("Please select different teams")
        else:
            # Prepare input data
            input_data = {
                'team1_encoded': team_encoder.transform([team1])[0],
                'team2_encoded': team_encoder.transform([team2])[0],
                'toss_winner_encoded': team_encoder.transform([toss_winner])[0],
                'is_batting_first': 1 if toss_decision == 'bat' else 0,
                'team1_win_rate': team_stats[team1]['win_rate'],
                'team2_win_rate': team_stats[team2]['win_rate'],
                'team1_toss_win_rate': team_stats[team1]['toss_win_rate'],
                'team2_toss_win_rate': team_stats[team2]['toss_win_rate'],
                'team1_home_win_rate': team_stats[team1]['home_win_rate'],
                'team2_away_win_rate': team_stats[team2]['away_win_rate'],
                'venue_encoded': venue_encoder.transform([venue])[0],
                'city_encoded': city_encoder.transform([city])[0]
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Scale the input
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            win_probability = model.predict_proba(input_scaled)[0]
            
            # Display prediction
            st.subheader("Match Prediction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Winner", team1 if prediction == 1 else team2)
            
            with col2:
                if prediction == 1:
                    st.metric(f"{team1} Win Probability", f"{win_probability[1]:.2%}")
                    st.metric(f"{team2} Win Probability", f"{win_probability[0]:.2%}")
                else:
                    st.metric(f"{team1} Win Probability", f"{win_probability[1]:.2%}")
                    st.metric(f"{team2} Win Probability", f"{win_probability[0]:.2%}")
            
            # Display key factors
            st.subheader("Key Factors")
            
            # Convert all rates to float to avoid type comparison issues
            team1_home_rate = float(team_stats[team1]['home_win_rate'])
            team1_win_rate = float(team_stats[team1]['win_rate'])
            team1_toss_rate = float(team_stats[team1]['toss_win_rate'])
            team2_away_rate = float(team_stats[team2]['away_win_rate'])
            team2_win_rate = float(team_stats[team2]['win_rate'])
            team2_toss_rate = float(team_stats[team2]['toss_win_rate'])
            
            factors_df = pd.DataFrame({
                'Factor': ['Home/Away Advantage', 'Team Form', 'Toss Advantage', 'Venue Familiarity'],
                'Team 1 Value': [
                    f"Home: {team1_home_rate:.2%}",
                    f"Win Rate: {team1_win_rate:.2%}",
                    f"Toss Win Rate: {team1_toss_rate:.2%}",
                    "High" if venue in matches_df[matches_df['team1'] == team1]['venue'].values else "Low"
                ],
                'Team 2 Value': [
                    f"Away: {team2_away_rate:.2%}",
                    f"Win Rate: {team2_win_rate:.2%}",
                    f"Toss Win Rate: {team2_toss_rate:.2%}",
                    "High" if venue in matches_df[matches_df['team1'] == team2]['venue'].values else "Low"
                ]
            })
            
            st.table(factors_df)

# Tab 2: Team Analysis
with tab2:
    st.header("Team Performance Analysis")
    
    selected_team = st.selectbox("Select Team for Analysis", teams, key="team_analysis")
    
    # Team overview
    st.subheader(f"{selected_team} - Team Overview")
    
    # Team stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Matches Played", team_stats[selected_team]['matches_played'])
    
    with col2:
        st.metric("Matches Won", team_stats[selected_team]['matches_won'])
    
    with col3:
        win_rate = float(team_stats[selected_team]['win_rate'])
        st.metric("Win Rate", f"{win_rate:.2%}")
    
    with col4:
        toss_rate = float(team_stats[selected_team]['toss_win_rate'])
        st.metric("Toss Win Rate", f"{toss_rate:.2%}")
    
    # Team performance over seasons
    st.subheader("Performance Over Seasons")
    
    team_seasons = matches_df[(matches_df['team1'] == selected_team) | (matches_df['team2'] == selected_team)]
    team_seasons_wins = team_seasons[team_seasons['winner'] == selected_team]
    
    season_performance = pd.DataFrame({
        'Season': sorted(team_seasons['season'].unique()),
    })
    
    season_performance['Matches'] = season_performance['Season'].apply(
        lambda x: team_seasons[team_seasons['season'] == x].shape[0]
    )
    
    season_performance['Wins'] = season_performance['Season'].apply(
        lambda x: team_seasons_wins[team_seasons_wins['season'] == x].shape[0]
    )
    
    season_performance['Win Rate'] = season_performance['Wins'] / season_performance['Matches']
    
    # Plot season performance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Season', y='Win Rate', data=season_performance, ax=ax)
    ax.set_title(f"{selected_team} - Win Rate by Season")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Rate")
    ax.set_xlabel("Season")
    
    for i, v in enumerate(season_performance['Win Rate']):
        ax.text(i, v + 0.05, f"{v:.2%}", ha='center')
    
    st.pyplot(fig)
    
    # Head-to-head analysis
    st.subheader("Head-to-Head Analysis")
    
    h2h_stats = {}
    
    for team in teams:
        if team == selected_team:
            continue
        
        # Matches between selected_team and team
        matches_h2h = matches_df[
            ((matches_df['team1'] == selected_team) & (matches_df['team2'] == team)) |
            ((matches_df['team1'] == team) & (matches_df['team2'] == selected_team))
        ]
        
        # Wins by selected_team against team
        wins_h2h = matches_h2h[matches_h2h['winner'] == selected_team].shape[0]
        
        # Total matches
        total_h2h = matches_h2h.shape[0]
        
        # Win rate
        win_rate_h2h = wins_h2h / total_h2h if total_h2h > 0 else 0
        
        h2h_stats[team] = {
            'matches': total_h2h,
            'wins': wins_h2h,
            'losses': total_h2h - wins_h2h,
            'win_rate': win_rate_h2h
        }
    
    h2h_df = pd.DataFrame.from_dict(h2h_stats, orient='index')
    h2h_df = h2h_df.reset_index().rename(columns={'index': 'Team'})
    h2h_df = h2h_df.sort_values('win_rate', ascending=False)
    
    # Plot head-to-head win rates
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Team', y='win_rate', data=h2h_df, ax=ax)
    ax.set_title(f"{selected_team} - Win Rate Against Other Teams")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Rate")
    ax.set_xlabel("Opponent Team")
    plt.xticks(rotation=45, ha='right')
    
    for i, v in enumerate(h2h_df['win_rate']):
        ax.text(i, v + 0.05, f"{v:.2%}", ha='center')
    
    st.pyplot(fig)
    
    # Display head-to-head stats table
    h2h_display = h2h_df.copy()
    h2h_display['win_rate'] = h2h_display['win_rate'].apply(lambda x: f"{float(x):.2%}")
    h2h_display = h2h_display.rename(columns={
        'matches': 'Matches',
        'wins': 'Wins',
        'losses': 'Losses',
        'win_rate': 'Win Rate'
    })
    
    st.table(h2h_display)


