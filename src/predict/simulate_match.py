import pickle
import numpy as np
from scipy.stats import poisson
import pandas as pd

def load_models(model_home_path, model_away_path):
    with open(model_home_path, 'rb') as f:
        model_home = pickle.load(f)
    with open(model_away_path, 'rb') as f:
        model_away = pickle.load(f)
    return model_home, model_away

def predict_match_outcome(model_home, model_away, features):
    """
    features: DataFrame with ONE row, including all columns needed by both models
    """
    # Home model expects 'home_avg_xg', 'away_avg_xg', 'home_field', and all home_form_... features
    # Away model expects 'away_avg_xg', 'home_avg_xg', and all away_form_... features
    lambda_home = model_home.predict(features)[0]  # Expected home goals
    lambda_away = model_away.predict(features)[0]  # Expected away goals

    max_goals = 6
    prob_matrix = np.outer(poisson.pmf(range(max_goals + 1), lambda_home),
                           poisson.pmf(range(max_goals + 1), lambda_away))
    home_win_prob = np.tril(prob_matrix, -1).sum()
    draw_prob = np.trace(prob_matrix)
    away_win_prob = np.triu(prob_matrix, 1).sum()

    return {
        'lambda_home': lambda_home,
        'lambda_away': lambda_away,
        'home_win_prob': home_win_prob,
        'draw_prob': draw_prob,
        'away_win_prob': away_win_prob,
        'score_probabilities': prob_matrix
    }

if __name__ == "__main__":
    # Example: load models and predict for a new game
    model_home, model_away = load_models("models/model_home.pkl", "models/model_away.pkl")

    # You must prepare a features DataFrame row (with same sequence/columns as used in training)
    input_features = {
        'home_avg_xg': 1.7,  # replace with your values
        'away_avg_xg': 1.4,  # replace with your values
        'home_field': 1      # always 1 here
        # ... add your 'home_form_goals_scored', etc., with the right recent values for the two teams!
    }
    # Add any other rolling/form features your models use
    # For a real prediction, populate these from recent matches of both teams

    test_df = pd.DataFrame([input_features])
    pred = predict_match_outcome(model_home, model_away, test_df)

    print(f"Expected home goals: {pred['lambda_home']:.2f}")
    print(f"Expected away goals: {pred['lambda_away']:.2f}")
    print(f"Win probabilities: Home {pred['home_win_prob']:.3f}, Draw {pred['draw_prob']:.3f}, Away {pred['away_win_prob']:.3f}")
