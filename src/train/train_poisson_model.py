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
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional


def calculate_time_weights(dates: pd.Series, xi: float = 0.012) -> np.ndarray:
    """
    Calculate time-weighting using Dixon-Coles exponential decay function.
    
    Recent matches get higher weights, older matches get lower weights.
    This addresses the limitation that teams change over time due to
    new players, coaches, and evolving tactics.
    
    Formula: φ(t) = exp(-ξ * t)
    
    Where:
        - t is time elapsed since the match (in half-weeks, i.e., 3.5 days)
        - ξ (xi) is the decay parameter controlling how quickly weights decrease
        
    Reference: Dixon & Coles (1997) "Modelling Association Football Scores"
    http://web.math.ku.dk/~rolf/teaching/thesis/DixonColes.pdf
    
    Args:
        dates: Series of match dates (datetime or string format)
        xi: Decay parameter (default 0.012 based on modern EPL data)
            - Higher xi = more weight on recent matches
            - Original paper used 0.0065
            - 0.012 found optimal for 2010-2021 EPL data
            
    Returns:
        Array of weights (one per match), with recent matches weighted higher
        
    Example:
        >>> dates = pd.Series(['2024-01-01', '2024-06-01', '2024-11-01'])
        >>> weights = calculate_time_weights(dates, xi=0.012)
        >>> # Most recent match (2024-11-01) will have weight ≈ 1.0
        >>> # Older matches will have exponentially decreasing weights
    """
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates)
    
    # Calculate time difference from most recent match
    latest_date = dates.max()
    days_elapsed = (latest_date - dates).dt.days
    
    # Convert days to half-weeks (Dixon-Coles time unit)
    # 1 half-week = 3.5 days
    half_weeks_elapsed = days_elapsed / 3.5
    
    # Apply exponential decay: φ(t) = exp(-ξ * t)
    weights = np.exp(-xi * half_weeks_elapsed)
    
    return weights


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


def fit_model(model_data: pd.DataFrame, weights: Optional[np.ndarray] = None) -> sm.GLM:
    """
    Fit Poisson GLM model using team, opponent, and home advantage.
    
    Formula: goals ~ home + team + opponent
    
    Args:
        model_data: DataFrame with columns team, opponent, goals, home
        weights: Optional array of frequency weights for time-weighting.
                If provided, recent matches will be weighted more heavily.
        
    Returns:
        Fitted statsmodels GLM object
    """
    model = smf.glm(
        formula="goals ~ home + team + opponent",
        data=model_data,
        family=sm.families.Poisson(),
        freq_weights=weights,  # Apply time-weighting here
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
    xi: float = 0.0030,
    use_time_weighting: bool = True,
) -> Tuple[sm.GLM, dict]:
    """
    Train Poisson GLM model on historical EPL data with time-weighting.
    
    Args:
        data_path: Path to CSV file with historical match data
        output_dir: Directory where model files will be saved
        xi: Decay parameter for time-weighting (default 0.0030)
            - Higher xi = more weight on recent matches
            - Optimal value found via hyperparameter tuning: 0.0030
            - Original Dixon-Coles paper: 0.0065
            - Modern EPL data (article): 0.012
        use_time_weighting: Whether to apply Dixon-Coles time-weighting
            
    Returns:
        Tuple of (fitted_model, coefficients_dict)
    """
    # Load historical data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, dtype={'Season': str})
    
    print(f"Loaded {len(df)} matches")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Calculate time weights if enabled
    weights = None
    if use_time_weighting:
        print(f"\n📊 Calculating time weights (Dixon-Coles method, ξ={xi})...")
        match_weights = calculate_time_weights(df['Date'], xi=xi)
        
        # Duplicate weights since we have 2 rows per match (home + away)
        weights = np.concatenate([match_weights, match_weights])
        
        print(f"✓ Time-weighting enabled")
        print(f"  Most recent matches: weight ≈ {match_weights.max():.4f}")
        print(f"  Oldest matches: weight ≈ {match_weights.min():.4f}")
        print(f"  Mean weight: {match_weights.mean():.4f}")
    else:
        print("\n⚠️  Time-weighting disabled (all matches weighted equally)")
    
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
    if use_time_weighting:
        print(f"Using time-weighted training (ξ={xi})")
    fitted_model = fit_model(model_df, weights=weights)
    
    # Extract coefficients
    coefficients = extract_coefficients(fitted_model)
    
    # Add training metadata
    coefficients['total_matches'] = len(df)
    coefficients['total_samples'] = len(model_df)
    coefficients['time_weighted'] = use_time_weighting
    if use_time_weighting:
        coefficients['xi'] = xi
        coefficients['weighting_method'] = 'Dixon-Coles exponential decay'
    
    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETE")
    print("="*80)
    print(f"\nTotal matches: {len(df)}")
    print(f"Total samples (2 per match): {len(model_df)}")
    print(f"Intercept (baseline): {coefficients['intercept']:.4f}")
    print(f"Home advantage: {coefficients['home_advantage']:.4f}")
    print(f"Team attack coefficients: {len(coefficients['team_attack'])} teams")
    print(f"Opponent defense coefficients: {len(coefficients['opponent_defense'])} teams")
    if use_time_weighting:
        print(f"\n⏱️  Time-weighting: Enabled (ξ={xi})")
        print(f"  Method: Dixon-Coles exponential decay")
    else:
        print(f"\n⏱️  Time-weighting: Disabled")
    
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
