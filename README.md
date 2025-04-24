# IPL Match Prediction and Analysis

This project uses machine learning to predict IPL match outcomes and provides team performance analysis through a Streamlit web application.

## Features

- **Match Prediction**: Predict the outcome of IPL matches based on historical data
- **Team Analysis**: View detailed performance metrics for each team
- **Dataset Overview**: Explore the IPL dataset with visualizations

## Data

The project uses two datasets:
- `matches.csv`: Contains match-level data for IPL matches (2008-2017)
- `deliveries.csv`: Contains ball-by-ball data for IPL matches

## Setup and Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the machine learning model:
   ```
   python3 train_model.py
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Project Structure

- `train_model.py`: Script to preprocess data and train the machine learning model
- `app.py`: Streamlit application for match prediction and team analysis
- `models/`: Directory containing trained models and encoders
- `requirements.txt`: List of Python dependencies

## Machine Learning Model

The project uses a Random Forest Classifier to predict match outcomes based on features like:
- Team performance metrics
- Toss information
- Venue statistics
- Home/away advantage

## Usage

1. **Match Prediction**: Select teams, venue, and toss information to predict the match outcome
2. **Team Analysis**: View performance metrics and head-to-head statistics for each team
3. **Dataset Overview**: Explore the IPL dataset with visualizations
# Ipl_match_prediction
