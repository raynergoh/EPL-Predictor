"""
Backtesting framework for Poisson prediction models.

This script performs walk-forward backtesting to evaluate model performance
on historical out-of-sample data. It compares:
1. Baseline (no time-weighting)
2. Time-weighted with optimal ξ

Metrics:
- Prediction accuracy (correct outcome)
- Brier score (calibration quality)
- Log-likelihood
- Return on investment (ROI) if betting

Usage:
    python3 src/backtest/backtest_models.py
    
Output:
    - Console: Performance metrics comparison
    - CSV: data/backtest/backtest_results_{timestamp}.csv
    - JSON: data/backtest/backtest_summary_{timestamp}.json
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Tuple
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from train.train_poisson_model import (
    create_model_data,
    fit_model,
    calculate_time_weights
)


class PoissonBacktester:
    """Backtesting framework for Poisson models."""
    
    def __init__(self, data_path: str = "data/raw/epl_historical_results.csv"):
        """Initialize backtester with historical data."""
        self.df = pd.read_csv(data_path, dtype={'Season': str})
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Get baseline team (first alphabetically)
        all_teams = sorted(set(self.df['HomeTeam'].unique()) | set(self.df['AwayTeam'].unique()))
        self.baseline_team = all_teams[0]
        
        print(f"Loaded {len(self.df)} matches")
        print(f"Date range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
        print(f"Baseline team: {self.baseline_team}")
    
    def predict_match(
        self,
        model,
        home_team: str,
        away_team: str
    ) -> Dict[str, float]:
        """
        Predict match outcome using trained model.
        
        Returns dict with:
        - home_xg, away_xg: Expected goals
        - home_win, draw, away_win: Outcome probabilities
        """
        try:
            # Predict expected goals
            home_xg = self._predict_expected_goals(model, home_team, away_team, is_home=True)
            away_xg = self._predict_expected_goals(model, home_team, away_team, is_home=False)
            
            # Calculate outcome probabilities using Poisson PMF
            home_win_prob = 0.0
            draw_prob = 0.0
            away_win_prob = 0.0
            
            for home_goals in range(7):
                for away_goals in range(7):
                    prob = poisson.pmf(home_goals, home_xg) * poisson.pmf(away_goals, away_xg)
                    
                    if home_goals > away_goals:
                        home_win_prob += prob
                    elif home_goals == away_goals:
                        draw_prob += prob
                    else:
                        away_win_prob += prob
            
            return {
                'home_xg': home_xg,
                'away_xg': away_xg,
                'home_win': home_win_prob,
                'draw': draw_prob,
                'away_win': away_win_prob
            }
        except Exception as e:
            # Return uniform probabilities if prediction fails
            return {
                'home_xg': np.nan,
                'away_xg': np.nan,
                'home_win': 1/3,
                'draw': 1/3,
                'away_win': 1/3
            }
    
    def _predict_expected_goals(
        self,
        model,
        home_team: str,
        away_team: str,
        is_home: bool
    ) -> float:
        """Predict expected goals for a team."""
        params = model.params
        
        intercept = params['Intercept']
        home_coef = params['home']
        
        if is_home:
            team = home_team
            opponent = away_team
            home_advantage = home_coef
        else:
            team = away_team
            opponent = home_team
            home_advantage = 0.0
        
        # Get attack strength
        if team == self.baseline_team:
            attack = 0.0
        else:
            param_name = f'team[T.{team}]'
            attack = params.get(param_name, 0.0)
        
        # Get defense strength
        if opponent == self.baseline_team:
            defense = 0.0
        else:
            param_name = f'opponent[T.{opponent}]'
            defense = params.get(param_name, 0.0)
        
        log_lambda = intercept + attack + defense + home_advantage
        return np.exp(log_lambda)
    
    def calculate_brier_score(self, predictions: List[Dict], actuals: List[str]) -> float:
        """
        Calculate Brier score (lower is better).
        
        Measures calibration quality of probabilistic predictions.
        Perfect predictions = 0, random = 0.67
        """
        brier_scores = []
        
        for pred, actual in zip(predictions, actuals):
            # Convert actual outcome to probability vector
            if actual == 'H':  # Home win
                actual_vec = [1.0, 0.0, 0.0]
            elif actual == 'D':  # Draw
                actual_vec = [0.0, 1.0, 0.0]
            else:  # Away win
                actual_vec = [0.0, 0.0, 1.0]
            
            # Predicted probabilities
            pred_vec = [pred['home_win'], pred['draw'], pred['away_win']]
            
            # Brier score for this prediction
            brier = np.mean([(p - a)**2 for p, a in zip(pred_vec, actual_vec)])
            brier_scores.append(brier)
        
        return np.mean(brier_scores)
    
    def calculate_accuracy(self, predictions: List[Dict], actuals: List[str]) -> float:
        """Calculate prediction accuracy (correct outcome)."""
        correct = 0
        
        for pred, actual in zip(predictions, actuals):
            # Get most likely outcome
            probs = [pred['home_win'], pred['draw'], pred['away_win']]
            predicted_outcome = ['H', 'D', 'A'][np.argmax(probs)]
            
            if predicted_outcome == actual:
                correct += 1
        
        return correct / len(predictions)
    
    def calculate_log_likelihood(self, predictions: List[Dict], actuals: pd.DataFrame) -> float:
        """Calculate average log-likelihood."""
        ll_values = []
        
        for pred, (_, row) in zip(predictions, actuals.iterrows()):
            home_xg = pred['home_xg']
            away_xg = pred['away_xg']
            
            if np.isnan(home_xg) or np.isnan(away_xg):
                continue
            
            home_goals = row['FTHG']
            away_goals = row['FTAG']
            
            home_ll = poisson.logpmf(home_goals, home_xg)
            away_ll = poisson.logpmf(away_goals, away_xg)
            
            if np.isfinite(home_ll) and np.isfinite(away_ll):
                ll_values.append(home_ll + away_ll)
        
        return np.mean(ll_values) if ll_values else np.nan
    
    def backtest_model(
        self,
        train_end_idx: int,
        test_start_idx: int,
        test_end_idx: int,
        xi: float = None,
        use_time_weighting: bool = True
    ) -> Dict:
        """
        Train model on historical data and test on future matches.
        
        Args:
            train_end_idx: Last index of training data
            test_start_idx: First index of test data
            test_end_idx: Last index of test data
            xi: Decay parameter (None = no weighting)
            use_time_weighting: Whether to apply time-weighting
            
        Returns:
            Dict with predictions and actual results
        """
        # Split data
        train_df = self.df.iloc[:train_end_idx].copy()
        test_df = self.df.iloc[test_start_idx:test_end_idx].copy()
        
        # Calculate time weights if enabled
        weights = None
        if use_time_weighting and xi is not None:
            train_weights = calculate_time_weights(train_df['Date'], xi=xi)
            weights = np.concatenate([train_weights, train_weights])
        
        # Train model
        model_df = create_model_data(
            home_team=train_df['HomeTeam'],
            away_team=train_df['AwayTeam'],
            home_goals=train_df['FTHG'],
            away_goals=train_df['FTAG'],
        )
        
        try:
            model = fit_model(model_df, weights=weights)
        except Exception as e:
            print(f"  Model fitting failed: {e}")
            return None
        
        # Predict on test set
        predictions = []
        for _, row in test_df.iterrows():
            pred = self.predict_match(model, row['HomeTeam'], row['AwayTeam'])
            predictions.append(pred)
        
        return {
            'predictions': predictions,
            'actuals': test_df['FTR'].tolist(),
            'test_df': test_df
        }
    
    def walk_forward_backtest(
        self,
        initial_train_size: int = 3000,
        test_window_size: int = 380,
        n_folds: int = 5,
        models_to_test: List[Dict] = None
    ) -> pd.DataFrame:
        """
        Perform walk-forward backtesting.
        
        This simulates real-world usage: train on past, predict future, then
        move forward in time and repeat.
        
        Args:
            initial_train_size: Minimum training samples
            test_window_size: Size of each test window (~1 season)
            n_folds: Number of forward steps
            models_to_test: List of model configs to compare
                [{'name': 'Baseline', 'xi': None, 'use_time_weighting': False},
                 {'name': 'Optimal', 'xi': 0.003, 'use_time_weighting': True}]
        
        Returns:
            DataFrame with results for each model on each fold
        """
        if models_to_test is None:
            models_to_test = [
                {'name': 'Baseline (No Weighting)', 'xi': None, 'use_time_weighting': False},
                {'name': 'Time-Weighted (ξ=0.003)', 'xi': 0.003, 'use_time_weighting': True}
            ]
        
        print("\n" + "="*80)
        print("WALK-FORWARD BACKTESTING")
        print("="*80)
        print(f"Initial training size: {initial_train_size} matches")
        print(f"Test window size: {test_window_size} matches (~1 season)")
        print(f"Number of folds: {n_folds}")
        print(f"Models to test: {len(models_to_test)}")
        print()
        
        results = []
        total_matches = len(self.df)
        
        for fold in range(n_folds):
            train_end = initial_train_size + fold * test_window_size
            test_start = train_end
            test_end = min(test_start + test_window_size, total_matches)
            
            if test_end - test_start < 100:
                print(f"Fold {fold+1}: Insufficient test data, skipping")
                break
            
            train_dates = self.df.iloc[train_end-1]['Date'].date()
            test_start_date = self.df.iloc[test_start]['Date'].date()
            test_end_date = self.df.iloc[test_end-1]['Date'].date()
            
            print(f"\nFold {fold+1}/{n_folds}")
            print(f"  Training: {train_end} matches (up to {train_dates})")
            print(f"  Testing: {test_end - test_start} matches ({test_start_date} to {test_end_date})")
            
            for model_config in models_to_test:
                model_name = model_config['name']
                print(f"    {model_name}...", end=' ', flush=True)
                
                result = self.backtest_model(
                    train_end_idx=train_end,
                    test_start_idx=test_start,
                    test_end_idx=test_end,
                    xi=model_config['xi'],
                    use_time_weighting=model_config['use_time_weighting']
                )
                
                if result is None:
                    print("FAILED")
                    continue
                
                # Calculate metrics
                accuracy = self.calculate_accuracy(result['predictions'], result['actuals'])
                brier_score = self.calculate_brier_score(result['predictions'], result['actuals'])
                log_likelihood = self.calculate_log_likelihood(result['predictions'], result['test_df'])
                
                print(f"Acc={accuracy:.1%}, Brier={brier_score:.3f}, LL={log_likelihood:.3f}")
                
                results.append({
                    'fold': fold + 1,
                    'model': model_name,
                    'xi': model_config['xi'],
                    'train_size': train_end,
                    'test_size': test_end - test_start,
                    'test_start_date': test_start_date,
                    'test_end_date': test_end_date,
                    'accuracy': accuracy,
                    'brier_score': brier_score,
                    'log_likelihood': log_likelihood
                })
        
        return pd.DataFrame(results)
    
    def print_summary(self, results_df: pd.DataFrame):
        """Print summary statistics for backtesting results."""
        print("\n" + "="*80)
        print("BACKTESTING SUMMARY")
        print("="*80)
        
        for model_name in results_df['model'].unique():
            model_results = results_df[results_df['model'] == model_name]
            
            print(f"\n{model_name}:")
            print(f"  Accuracy:       {model_results['accuracy'].mean():.2%} ± {model_results['accuracy'].std():.2%}")
            print(f"  Brier Score:    {model_results['brier_score'].mean():.4f} ± {model_results['brier_score'].std():.4f}")
            print(f"  Log-Likelihood: {model_results['log_likelihood'].mean():.4f} ± {model_results['log_likelihood'].std():.4f}")
        
        # Compare models
        if len(results_df['model'].unique()) >= 2:
            print("\n" + "-"*80)
            print("MODEL COMPARISON")
            print("-"*80)
            
            models = results_df['model'].unique()
            baseline = results_df[results_df['model'] == models[0]]
            optimized = results_df[results_df['model'] == models[1]]
            
            acc_improvement = (optimized['accuracy'].mean() - baseline['accuracy'].mean()) * 100
            brier_improvement = (baseline['brier_score'].mean() - optimized['brier_score'].mean()) * 1000
            ll_improvement = (optimized['log_likelihood'].mean() - baseline['log_likelihood'].mean()) * 100
            
            print(f"\nImprovement ({models[1]} vs {models[0]}):")
            print(f"  Accuracy:       {acc_improvement:+.2f} percentage points")
            print(f"  Brier Score:    {brier_improvement:+.2f} points (×1000, lower better)")
            print(f"  Log-Likelihood: {ll_improvement:+.2f} points (×100, higher better)")
            
            if acc_improvement > 0:
                print(f"\n✓ Time-weighting improves accuracy by {acc_improvement:.2f}%")
            else:
                print(f"\n⚠️  Time-weighting reduces accuracy by {abs(acc_improvement):.2f}%")


def save_results(results_df: pd.DataFrame, output_dir: str = "data/backtest"):
    """Save backtesting results to CSV and JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save CSV
    csv_file = output_path / f'backtest_results_{timestamp}.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"\n✓ Results saved to {csv_file}")
    
    # Save JSON summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_folds': len(results_df['fold'].unique()),
        'models_tested': results_df['model'].unique().tolist(),
        'metrics_by_model': {}
    }
    
    for model_name in results_df['model'].unique():
        model_results = results_df[results_df['model'] == model_name]
        summary['metrics_by_model'][model_name] = {
            'accuracy_mean': float(model_results['accuracy'].mean()),
            'accuracy_std': float(model_results['accuracy'].std()),
            'brier_score_mean': float(model_results['brier_score'].mean()),
            'brier_score_std': float(model_results['brier_score'].std()),
            'log_likelihood_mean': float(model_results['log_likelihood'].mean()),
            'log_likelihood_std': float(model_results['log_likelihood'].std())
        }
    
    json_file = output_path / f'backtest_summary_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to {json_file}")


def main():
    """Run backtesting."""
    # Initialize backtester
    backtester = PoissonBacktester()
    
    # Run walk-forward backtest
    results_df = backtester.walk_forward_backtest(
        initial_train_size=3000,  # ~8 seasons
        test_window_size=380,     # ~1 season
        n_folds=5,
        models_to_test=[
            {'name': 'Baseline (No Weighting)', 'xi': None, 'use_time_weighting': False},
            {'name': 'Time-Weighted (ξ=0.003)', 'xi': 0.003, 'use_time_weighting': True}
        ]
    )
    
    # Print summary
    backtester.print_summary(results_df)
    
    # Save results
    save_results(results_df)


if __name__ == "__main__":
    main()
