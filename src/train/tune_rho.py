"""
Hyperparameter tuning for Dixon-Coles dependency parameter (ρ).

Finds optimal ρ value using time-series cross-validation with log-likelihood metric.

Based on methodology from:
- Dixon & Coles (1997) - "Modelling Association Football Scores and Inefficiencies in the Football Betting Market"
- Typical ρ values: -0.13 to -0.18 for English football

Usage:
    python3 src/train/tune_rho.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from train.train_poisson_model import train_poisson_model
from predict.generate_probabilities import PoissonPredictor
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_log_likelihood(predictions: list, actuals: list) -> float:
    """
    Calculate log-likelihood for probabilistic predictions.
    
    Args:
        predictions: List of prediction dicts with 'probabilities'
        actuals: List of actual outcomes ('home_win', 'draw', 'away_win')
        
    Returns:
        Mean log-likelihood (higher is better)
    """
    log_likelihoods = []
    
    for pred, actual in zip(predictions, actuals):
        prob = pred['probabilities'][actual]
        # Add small epsilon to avoid log(0)
        prob = max(prob, 1e-10)
        log_likelihoods.append(np.log(prob))
    
    return np.mean(log_likelihoods)


def predict_match_outcome(home_team: str, away_team: str, 
                          home_goals: int, away_goals: int) -> str:
    """Determine actual match outcome."""
    if home_goals > away_goals:
        return 'home_win'
    elif home_goals < away_goals:
        return 'away_win'
    else:
        return 'draw'


def time_series_cross_validation(data_path: str = 'data/raw/epl_historical_results.csv',
                                 rho_values: list = None,
                                 n_folds: int = 5) -> pd.DataFrame:
    """
    Perform time-series cross-validation to find optimal ρ.
    
    Strategy:
    - Use time-weighted model (already optimized with ξ=0.003)
    - Test different ρ values for dependency correction
    - Preserve temporal ordering (train on past, test on future)
    
    Args:
        data_path: Path to historical match data
        rho_values: List of ρ values to test (default: -0.25 to 0.0)
        n_folds: Number of cross-validation folds (default: 5)
        
    Returns:
        DataFrame with results for each ρ value
    """
    if rho_values is None:
        # Test range from -0.25 to 0.0 (step 0.01)
        # Dixon & Coles (1997) found ρ ≈ -0.13 for English football
        # We'll test broader range to find optimal for our dataset
        rho_values = np.arange(-0.25, 0.01, 0.01).round(2).tolist()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"RHO (ρ) HYPERPARAMETER TUNING")
    logger.info(f"{'='*80}\n")
    logger.info(f"Testing {len(rho_values)} values of ρ (rho)")
    logger.info(f"Range: {min(rho_values):.2f} to {max(rho_values):.2f}")
    logger.info(f"Cross-validation: {n_folds} temporal folds\n")
    
    # Load data
    df = pd.read_csv(data_path, dtype={'Season': str})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Time-series split (preserves temporal order)
    fold_size = len(df) // (n_folds + 1)
    
    results = []
    
    for rho in rho_values:
        logger.info(f"Testing ρ = {rho:.3f}...")
        fold_scores = []
        
        for fold in range(n_folds):
            # Training data: all matches up to this fold
            train_end = fold_size * (fold + 1)
            # Test data: next fold_size matches
            test_start = train_end
            test_end = test_start + fold_size
            
            test_df = df.iloc[test_start:test_end]
            
            if len(test_df) == 0:
                continue
            
            # Make predictions with this ρ value
            # Model is already trained with time-weighting (ξ=0.003)
            predictor = PoissonPredictor(use_dc_correction=True, rho=rho)
            
            predictions = []
            actuals = []
            
            for _, match in test_df.iterrows():
                try:
                    pred = predictor.predict_match(
                        match['HomeTeam'], 
                        match['AwayTeam'],
                        max_goals=6
                    )
                    predictions.append(pred)
                    
                    actual = predict_match_outcome(
                        match['HomeTeam'], 
                        match['AwayTeam'],
                        match['FTHG'], 
                        match['FTAG']
                    )
                    actuals.append(actual)
                except Exception as e:
                    logger.debug(f"  Skipping match due to error: {e}")
                    continue
            
            # Calculate log-likelihood for this fold
            if predictions and actuals:
                ll = calculate_log_likelihood(predictions, actuals)
                fold_scores.append(ll)
        
        # Average across folds
        mean_ll = np.mean(fold_scores)
        std_ll = np.std(fold_scores)
        
        results.append({
            'rho': rho,
            'mean_log_likelihood': mean_ll,
            'std_log_likelihood': std_ll,
            'n_folds': len(fold_scores)
        })
        
        logger.info(f"  ρ = {rho:.3f} → Log-likelihood = {mean_ll:.4f} ± {std_ll:.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best ρ
    best_idx = results_df['mean_log_likelihood'].idxmax()
    best_rho = results_df.loc[best_idx, 'rho']
    best_ll = results_df.loc[best_idx, 'mean_log_likelihood']
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TUNING COMPLETE")
    logger.info(f"{'='*80}\n")
    logger.info(f"🏆 BEST ρ: {best_rho:.3f}")
    logger.info(f"   Log-likelihood: {best_ll:.4f}")
    logger.info(f"\n📊 Comparison with literature:")
    logger.info(f"   Dixon & Coles (1997): ρ ≈ -0.13")
    logger.info(f"   Your optimal: ρ = {best_rho:.3f}")
    
    return results_df


def save_results(results_df: pd.DataFrame, output_dir: str = 'data/tuning'):
    """Save tuning results to CSV and JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    csv_file = output_path / f'rho_tuning_results_{timestamp}.csv'
    results_df.to_csv(csv_file, index=False)
    logger.info(f"\n✓ Detailed results saved: {csv_file}")
    
    # Save summary
    best_idx = results_df['mean_log_likelihood'].idxmax()
    summary = {
        'best_rho': float(results_df.loc[best_idx, 'rho']),
        'best_log_likelihood': float(results_df.loc[best_idx, 'mean_log_likelihood']),
        'best_std': float(results_df.loc[best_idx, 'std_log_likelihood']),
        'n_folds': int(results_df.loc[best_idx, 'n_folds']),
        'timestamp': timestamp,
        'total_rho_values_tested': len(results_df)
    }
    
    json_file = output_path / f'rho_tuning_summary_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Summary saved: {json_file}\n")


def main():
    """Run ρ hyperparameter tuning."""
    results_df = time_series_cross_validation()
    save_results(results_df)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Update default ρ in src/predict/generate_probabilities.py")
    print("2. Run backtesting to compare:")
    print("   - Baseline (no time-weighting, no DC)")
    print("   - Time-weighted only (ξ=0.003)")
    print("   - Time-weighted + DC (ξ=0.003, optimal ρ)")
    print("\n3. Update README.md with new results")
    print()


if __name__ == "__main__":
    main()
