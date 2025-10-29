"""
Train Poisson GLM model for football match prediction.

Based on methodology from: https://artiebits.com/blog/predicting-football-results-with-statistical-modelling/

The model uses the formula:
    log(λ_i) = α_0 + α_team_i + β_opponent_i + γ * home_i

Where:
    - α_0 is the intercept (baseline goal rate)
    - α_team_i is the attack strength of team i
    - β_opponent_i is the defense strength of opponent i
    - γ is the home advantage coefficient
    - home_i is 1 if playing at home, 0 otherwise
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple


def create_model_data(
    home_team: pd.Series,
    away_team: pd.Series,
    home_goals: pd.Series,
    away_goals: pd.Series,
) -> pd.DataFrame:
    """
    Transform match data into format suitable for Poisson GLM training.
    
    Each match is converted into two rows:
    - One row for the home team (home=1)
    - One row for the away team (home=0)
    
    Args:
        home_team: Series of home team names
        away_team: Series of away team names
        home_goals: Series of goals scored by home team
        away_goals: Series of goals scored by away team
        
    Returns:
        DataFrame with columns: team, opponent, goals, home
    """
    home_df = pd.DataFrame(
        data={
            "team": home_team,
            "opponent": away_team,
            "goals": home_goals,
            "home": 1,
        }
    )
    
    away_df = pd.DataFrame(
        data={
            "team": away_team,
            "opponent": home_team,
            "goals": away_goals,
            "home": 0,
        }
    )
    
    return pd.concat([home_df, away_df], ignore_index=True)


def fit_model(model_data: pd.DataFrame) -> sm.GLM:
    """
    Fit Poisson GLM model using team, opponent, and home advantage.
    
    Formula: goals ~ home + team + opponent
    
    Args:
        model_data: DataFrame with columns team, opponent, goals, home
        
    Returns:
        Fitted statsmodels GLM object
    """
    model = smf.glm(
        formula="goals ~ home + team + opponent",
        data=model_data,
        family=sm.families.Poisson(),
    )
    
    return model.fit()


def extract_coefficients(fitted_model: sm.GLM) -> dict:
    """
    Extract coefficients from fitted model into a clean dictionary format.
    
    Args:
        fitted_model: Fitted statsmodels GLM object
        
    Returns:
        Dictionary with intercept, home_advantage, team_attack, and opponent_defense
    """
    params = fitted_model.params
    
    # Extract intercept and home advantage
    intercept = params['Intercept']
    home_advantage = params['home']
    
    # Extract team attack strengths (relative to baseline)
    team_attack = {}
    for param_name, value in params.items():
        if param_name.startswith('team[T.'):
            team_name = param_name.replace('team[T.', '').rstrip(']')
            team_attack[team_name] = value
    
    # Extract opponent defense strengths (relative to baseline)
    opponent_defense = {}
    for param_name, value in params.items():
        if param_name.startswith('opponent[T.'):
            team_name = param_name.replace('opponent[T.', '').rstrip(']')
            opponent_defense[team_name] = value
    
    return {
        'intercept': intercept,
        'home_advantage': home_advantage,
        'team_attack': team_attack,
        'opponent_defense': opponent_defense,
        'trained_at': datetime.now().isoformat(),
        'formula': 'goals ~ home + team + opponent'
    }


def train_poisson_model(
    data_path: str = "data/raw/epl_historical_results.csv",
    output_dir: str = "models",
) -> Tuple[sm.GLM, dict]:
    """
    Train Poisson GLM model on historical EPL data.
    
    Args:
        data_path: Path to CSV file with historical match data
        output_dir: Directory where model files will be saved
        
    Returns:
        Tuple of (fitted_model, coefficients_dict)
    """
    # Load historical data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, dtype={'Season': str})
    
    print(f"Loaded {len(df)} matches")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Transform data into model format
    print("\nTransforming data for Poisson GLM...")
    model_df = create_model_data(
        home_team=df['HomeTeam'],
        away_team=df['AwayTeam'],
        home_goals=df['FTHG'],  # Full Time Home Goals
        away_goals=df['FTAG'],  # Full Time Away Goals
    )
    
    print(f"Created {len(model_df)} training samples ({len(df)} matches × 2)")
    
    # Fit the model
    print("\nFitting Poisson GLM model...")
    print("Formula: goals ~ home + team + opponent")
    fitted_model = fit_model(model_df)
    
    # Extract coefficients
    coefficients = extract_coefficients(fitted_model)
    
    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETE")
    print("="*80)
    print(f"\nIntercept (baseline): {coefficients['intercept']:.4f}")
    print(f"Home advantage: {coefficients['home_advantage']:.4f}")
    print(f"Team attack coefficients: {len(coefficients['team_attack'])} teams")
    print(f"Opponent defense coefficients: {len(coefficients['opponent_defense'])} teams")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the full model
    model_file = output_path / "poisson_glm_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(fitted_model, f)
    print(f"\n✓ Model saved to {model_file}")
    
    # Save coefficients as pickle
    coef_pkl_file = output_path / "poisson_coefficients.pkl"
    with open(coef_pkl_file, 'wb') as f:
        pickle.dump(coefficients, f)
    print(f"✓ Coefficients saved to {coef_pkl_file}")
    
    # Save coefficients as JSON (human-readable)
    coef_json_file = output_path / "poisson_coefficients.json"
    with open(coef_json_file, 'w') as f:
        json.dump(coefficients, f, indent=2)
    print(f"✓ Coefficients saved to {coef_json_file}")
    
    # Print model summary
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(fitted_model.summary())
    
    return fitted_model, coefficients


if __name__ == "__main__":
    train_poisson_model()
