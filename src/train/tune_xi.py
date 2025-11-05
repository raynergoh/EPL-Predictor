"""
Hyperparameter tuning for Dixon-Coles time-weighting decay parameter (ξ).

This script performs time-series cross-validation to find the optimal decay
parameter for the Poisson model. Unlike standard cross-validation, we MUST
preserve temporal ordering to avoid data leakage.

Reference: https://artiebits.com/blog/improving-poisson-model-using-time-weighting/

Usage:
    python3 src/train/tune_xi.py
    
Output:
    - Console: Results table with log-likelihood for each ξ
    - CSV: data/tuning/xi_tuning_results.csv
    - Best ξ value recommendation
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from pathlib import Path
import json
from datetime import datetime
from typing import List, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from train.train_poisson_model import (
    create_model_data,
    fit_model,
    calculate_time_weights
)


def calculate_log_likelihood(
    test_df: pd.DataFrame,
    model,
    baseline_team: str
) -> Tuple[float, int]:
    """
    Calculate average log-likelihood on test set.
    
    Log-likelihood measures how well the model's predicted probabilities
    match actual outcomes. Higher is better.
    
    Args:
        test_df: Test set with actual match results
        model: Fitted statsmodels GLM object
        baseline_team: Name of baseline team (first alphabetically)
        
    Returns:
        Tuple of (average_log_likelihood, valid_matches_count)
    """
    ll_values = []
    
    for _, row in test_df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_goals = row['FTHG']
        away_goals = row['FTAG']
        
        try:
            # Predict expected goals using model
            home_xg = predict_expected_goals(
                model, home_team, away_team, is_home=True, baseline_team=baseline_team
            )
            away_xg = predict_expected_goals(
                model, home_team, away_team, is_home=False, baseline_team=baseline_team
            )
            
            # Check for valid predictions
            if not (home_xg > 0 and away_xg > 0 and 
                    np.isfinite(home_xg) and np.isfinite(away_xg)):
                continue
            
            # Calculate log-likelihood for this match
            home_ll = poisson.logpmf(home_goals, home_xg)
            away_ll = poisson.logpmf(away_goals, away_xg)
            
            if np.isfinite(home_ll) and np.isfinite(away_ll):
                ll_values.append(home_ll + away_ll)
                
        except Exception as e:
            # Skip matches where prediction fails (e.g., unknown team)
            continue
    
    if not ll_values:
        return np.nan, 0
    
    return np.mean(ll_values), len(ll_values)


def predict_expected_goals(
    model,
    home_team: str,
    away_team: str,
    is_home: bool,
    baseline_team: str
) -> float:
    """
    Predict expected goals for a team using the Poisson model.
    
    Formula: λ = exp(α₀ + α_team + β_opponent + γ * home)
    
    Args:
        model: Fitted statsmodels GLM object
        home_team: Name of home team
        away_team: Name of away team
        is_home: True if predicting for home team, False for away
        baseline_team: Name of baseline team (coefficient = 0)
        
    Returns:
        Expected goals (lambda for Poisson distribution)
    """
    params = model.params
    
    # Get coefficients
    intercept = params['Intercept']
    home_coef = params['home']
    
    if is_home:
        # Home team attacking at home
        team = home_team
        opponent = away_team
        home_advantage = home_coef
    else:
        # Away team attacking away
        team = away_team
        opponent = home_team
        home_advantage = 0.0
    
    # Get attack strength (0 if baseline team)
    if team == baseline_team:
        attack = 0.0
    else:
        param_name = f'team[T.{team}]'
        if param_name in params:
            attack = params[param_name]
        else:
            raise ValueError(f"Team '{team}' not in model")
    
    # Get defense strength (0 if baseline team)
    if opponent == baseline_team:
        defense = 0.0
    else:
        param_name = f'opponent[T.{opponent}]'
        if param_name in params:
            defense = params[param_name]
        else:
            raise ValueError(f"Opponent '{opponent}' not in model")
    
    # Calculate expected goals
    log_lambda = intercept + attack + defense + home_advantage
    expected_goals = np.exp(log_lambda)
    
    return expected_goals


def time_series_cross_validation(
    df: pd.DataFrame,
    xi: float,
    n_splits: int = 5,
    min_train_size: int = 1000
) -> Tuple[float, int]:
    """
    Perform time-series cross-validation for a given xi value.
    
    CRITICAL: We use TEMPORAL splits, not random splits, to avoid data leakage.
    Each fold trains on past data and tests on future data.
    
    Args:
        df: Full dataset with Date column
        xi: Decay parameter to test
        n_splits: Number of cross-validation folds
        min_train_size: Minimum training set size (in matches)
        
    Returns:
        Tuple of (average_log_likelihood, total_valid_matches)
    """
    # Sort by date to ensure temporal ordering
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Get baseline team (first alphabetically)
    all_teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
    baseline_team = all_teams[0]
    
    # Calculate split points
    total_matches = len(df)
    test_size = (total_matches - min_train_size) // n_splits
    
    if test_size < 100:
        print(f"⚠️  Warning: Test set too small ({test_size} matches per fold)")
        n_splits = max(1, (total_matches - min_train_size) // 100)
        test_size = (total_matches - min_train_size) // n_splits
        print(f"   Reducing to {n_splits} splits")
    
    ll_values = []
    total_valid = 0
    
    for fold in range(n_splits):
        # Define train/test split (temporal)
        train_end = min_train_size + fold * test_size
        test_start = train_end
        test_end = test_start + test_size
        
        if test_end > total_matches:
            test_end = total_matches
        
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        if len(test_df) == 0:
            continue
        
        # Calculate time weights for training set
        train_weights = calculate_time_weights(train_df['Date'], xi=xi)
        
        # Transform to model format
        model_df = create_model_data(
            home_team=train_df['HomeTeam'],
            away_team=train_df['AwayTeam'],
            home_goals=train_df['FTHG'],
            away_goals=train_df['FTAG'],
        )
        
        # Duplicate weights (2 rows per match)
        weights = np.concatenate([train_weights, train_weights])
        
        # Fit model
        try:
            model = fit_model(model_df, weights=weights)
        except Exception as e:
            print(f"   Fold {fold+1}: Model fitting failed - {e}")
            continue
        
        # Evaluate on test set
        avg_ll, valid_count = calculate_log_likelihood(test_df, model, baseline_team)
        
        if not np.isnan(avg_ll):
            ll_values.append(avg_ll)
            total_valid += valid_count
    
    if not ll_values:
        return np.nan, 0
    
    return np.mean(ll_values), total_valid


def test_xi_values(
    xi_values: List[float],
    data_path: str = "data/raw/epl_historical_results.csv",
    n_splits: int = 5
) -> pd.DataFrame:
    """
    Test multiple xi values using time-series cross-validation.
    
    Args:
        xi_values: List of decay parameters to test
        data_path: Path to historical data CSV
        n_splits: Number of CV folds
        
    Returns:
        DataFrame with results for each xi
    """
    print("="*80)
    print("DIXON-COLES TIME-WEIGHTING HYPERPARAMETER TUNING")
    print("="*80)
    print(f"\nTesting {len(xi_values)} values of ξ (xi)")
    print(f"Cross-validation: {n_splits} temporal folds")
    print(f"Evaluation metric: Log-likelihood (higher is better)\n")
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, dtype={'Season': str})
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"✓ Loaded {len(df)} matches")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"  Teams: {len(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))}\n")
    
    # Test each xi value
    results = []
    
    print("Testing ξ values:")
    print("-" * 80)
    
    for i, xi in enumerate(xi_values, 1):
        print(f"[{i}/{len(xi_values)}] ξ = {xi:.4f}... ", end='', flush=True)
        
        avg_ll, valid_count = time_series_cross_validation(
            df, xi=xi, n_splits=n_splits
        )
        
        if np.isfinite(avg_ll):
            print(f"Log-likelihood = {avg_ll:.4f} ({valid_count} matches)")
            results.append({
                'xi': xi,
                'log_likelihood': avg_ll,
                'valid_matches': valid_count
            })
        else:
            print("FAILED")
    
    print("-" * 80)
    
    if not results:
        print("\n❌ No valid results obtained")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Find best xi
    best_idx = results_df['log_likelihood'].idxmax()
    best_xi = results_df.loc[best_idx, 'xi']
    best_ll = results_df.loc[best_idx, 'log_likelihood']
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))
    print(f"\n{'='*80}")
    print(f"🏆 BEST ξ: {best_xi:.4f}")
    print(f"   Log-likelihood: {best_ll:.4f}")
    print(f"{'='*80}\n")
    
    return results_df


def save_results(results_df: pd.DataFrame, output_dir: str = "data/tuning"):
    """Save tuning results to CSV and JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save CSV
    csv_file = output_path / f'xi_tuning_results_{timestamp}.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"✓ Results saved to {csv_file}")
    
    # Save JSON with metadata
    best_idx = results_df['log_likelihood'].idxmax()
    best_xi = results_df.loc[best_idx, 'xi']
    best_ll = results_df.loc[best_idx, 'log_likelihood']
    
    summary = {
        'best_xi': float(best_xi),
        'best_log_likelihood': float(best_ll),
        'tested_values': results_df.to_dict('records'),
        'tuned_at': datetime.now().isoformat()
    }
    
    json_file = output_path / f'xi_tuning_summary_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to {json_file}")


def main():
    """Run hyperparameter tuning."""
    # Define range of xi values to test
    # Based on article: optimal was 0.012 for 2010-2021 EPL data
    # We test around that range
    xi_values = np.arange(0.003, 0.0205, 0.0005)  # 0.003 to 0.020 in steps of 0.0005
    
    # Alternative: Fewer values for faster testing
    # xi_values = [0.003, 0.006, 0.009, 0.012, 0.015, 0.018]
    
    # Run tuning
    results_df = test_xi_values(
        xi_values=xi_values,
        n_splits=5
    )
    
    if not results_df.empty:
        # Save results
        save_results(results_df)
        
        # Print recommendation
        best_xi = results_df.loc[results_df['log_likelihood'].idxmax(), 'xi']
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Update train_poisson_model.py to use ξ={best_xi:.4f}")
        print(f"   Then retrain: python3 src/train/train_poisson_model.py\n")


if __name__ == "__main__":
    main()
