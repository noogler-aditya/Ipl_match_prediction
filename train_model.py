import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load data
print("Loading data...")
matches_df = pd.read_csv('matches.csv')
deliveries_df = pd.read_csv('deliveries.csv')

# Data preprocessing
print("Preprocessing data...")

# Filter out matches with no winner (tied, no result)
matches_df = matches_df.dropna(subset=['winner'])

# Extract features from matches data
matches_df['toss_win_match_win'] = (matches_df['toss_winner'] == matches_df['winner']).astype(int)
matches_df['is_batting_first'] = (matches_df['toss_decision'] == 'bat').astype(int)

# Create team encoders
team_encoder = LabelEncoder()
team_encoder.fit(pd.concat([matches_df['team1'], matches_df['team2']]).unique())

# Encode teams
matches_df['team1_encoded'] = team_encoder.transform(matches_df['team1'])
matches_df['team2_encoded'] = team_encoder.transform(matches_df['team2'])
matches_df['toss_winner_encoded'] = team_encoder.transform(matches_df['toss_winner'])
matches_df['winner_encoded'] = team_encoder.transform(matches_df['winner'])

# Create venue encoder
venue_encoder = LabelEncoder()
matches_df['venue_encoded'] = venue_encoder.fit_transform(matches_df['venue'])

# Create city encoder
city_encoder = LabelEncoder()
matches_df['city_encoded'] = city_encoder.fit_transform(matches_df['city'])

# Calculate team performance metrics
team_stats = {}

for team in team_encoder.classes_:
    # Matches won
    team_wins = matches_df[matches_df['winner'] == team].shape[0]
    
    # Matches played
    team_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)].shape[0]
    
    # Win rate
    win_rate = team_wins / team_matches if team_matches > 0 else 0
    
    # Toss win rate
    toss_wins = matches_df[matches_df['toss_winner'] == team].shape[0]
    toss_win_rate = toss_wins / team_matches if team_matches > 0 else 0
    
    # Home matches
    home_matches = matches_df[matches_df['team1'] == team].shape[0]
    
    # Away matches
    away_matches = matches_df[matches_df['team2'] == team].shape[0]
    
    # Home win rate
    home_wins = matches_df[(matches_df['team1'] == team) & (matches_df['winner'] == team)].shape[0]
    home_win_rate = home_wins / home_matches if home_matches > 0 else 0
    
    # Away win rate
    away_wins = matches_df[(matches_df['team2'] == team) & (matches_df['winner'] == team)].shape[0]
    away_win_rate = away_wins / away_matches if away_matches > 0 else 0
    
    team_stats[team] = {
        'matches_played': team_matches,
        'matches_won': team_wins,
        'win_rate': float(win_rate),
        'toss_win_rate': float(toss_win_rate),
        'home_win_rate': float(home_win_rate),
        'away_win_rate': float(away_win_rate)
    }

# Add team stats to matches
for idx, row in matches_df.iterrows():
    team1 = row['team1']
    team2 = row['team2']
    
    matches_df.at[idx, 'team1_win_rate'] = team_stats[team1]['win_rate']
    matches_df.at[idx, 'team2_win_rate'] = team_stats[team2]['win_rate']
    matches_df.at[idx, 'team1_toss_win_rate'] = team_stats[team1]['toss_win_rate']
    matches_df.at[idx, 'team2_toss_win_rate'] = team_stats[team2]['toss_win_rate']
    matches_df.at[idx, 'team1_home_win_rate'] = team_stats[team1]['home_win_rate']
    matches_df.at[idx, 'team2_away_win_rate'] = team_stats[team2]['away_win_rate']

# Create target variable: 1 if team1 wins, 0 if team2 wins
matches_df['team1_win'] = (matches_df['winner'] == matches_df['team1']).astype(int)

# Select features for the model
features = [
    'team1_encoded', 'team2_encoded', 'toss_winner_encoded', 'is_batting_first',
    'team1_win_rate', 'team2_win_rate', 'team1_toss_win_rate', 'team2_toss_win_rate',
    'team1_home_win_rate', 'team2_away_win_rate', 'venue_encoded', 'city_encoded'
]

X = matches_df[features]
y = matches_df['team1_win']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Save the model and encoders
print("Saving model and encoders...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/ipl_prediction_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(team_encoder, 'models/team_encoder.pkl')
joblib.dump(venue_encoder, 'models/venue_encoder.pkl')
joblib.dump(city_encoder, 'models/city_encoder.pkl')
joblib.dump(team_stats, 'models/team_stats.pkl')

print("Model training and saving complete!")
