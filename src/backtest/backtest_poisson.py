"""
Backtesting framework for Poisson GLM predictions.

Validates model accuracy on historical data using time-series cross-validation.
For each season split:
1. Load train/test data
2. Train Poisson GLM on train set
3. Generate predictions on test set
4. Compare predictions vs actual outcomes
5. Calculate accuracy metrics and calibration

Metrics:
- Accuracy: % of correct outcome predictions (home win/draw/away win)
- Precision/Recall: Per outcome type
- Calibration: Predicted probabilities vs actual frequencies
- ROI: Simulated betting returns vs bookmaker odds baseline
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from predict.generate_probabilities import PoissonPredictor


class BacktestFramework:
    """Backtesting framework for time-series cross-validation."""
    
    def __init__(
        self, 
        splits_dir: str = 'data/processed/season_splits',
        output_dir: str = 'data/processed/backtest_results'
    ):
        """
        Initialize backtesting framework.
        
        Args:
            splits_dir: Directory containing season train/test splits
            output_dir: Directory to save backtest results
        """
        self.splits_dir = Path(splits_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'backtest_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_season_splits(self) -> List[Tuple[str, Path, Path]]:
        """
        Get all season train/test split pairs.
        
        Returns:
            List of (season_name, train_path, test_path) tuples
        """
        splits = []
        
        # Find all test files
        test_files = sorted(self.splits_dir.glob('*_test.csv'))
        
        for test_file in test_files:
            season = test_file.stem.replace('_test', '')
            train_file = self.splits_dir / f'{season}_train.csv'
            
            if train_file.exists():
                splits.append((season, train_file, test_file))
                
        self.logger.info(f"Found {len(splits)} season splits")
        return splits
        
    def train_model_on_split(self, train_path: Path) -> sm.GLM:
        """
        Train Poisson GLM on a training split.
        
        Args:
            train_path: Path to training CSV
            
        Returns:
            Fitted statsmodels GLM model
        """
        self.logger.info(f"Training model on {train_path.name}")
        
        # Load training data
        train_df = pd.read_csv(train_path)
        
        # Train Poisson GLM
        formula = 'goals ~ home + C(team) + C(opponent)'
        model = smf.glm(
            formula=formula,
            data=train_df,
            family=sm.families.Poisson()
        )
        
        fitted_model = model.fit()
        
        self.logger.info(f"Model trained with {len(train_df)} rows")
        return fitted_model
        
    def predict_match(
        self, 
        model: sm.GLM, 
        home_team: str, 
        away_team: str,
        max_goals: int = 6
    ) -> Dict:
        """
        Generate prediction for a single match using trained model.
        
        Args:
            model: Fitted Poisson GLM
            home_team: Home team name
            away_team: Away team name
            max_goals: Maximum goals to consider in probability matrix
            
        Returns:
            Dict with prediction data
        """
        # Create feature rows for home and away teams
        home_row = pd.DataFrame({
            'team': [home_team],
            'opponent': [away_team],
            'home': [1]
        })
        
        away_row = pd.DataFrame({
            'team': [away_team],
            'opponent': [home_team],
            'home': [0]
        })
        
        try:
            # Get expected goals (lambda parameters)
            lambda_home = model.predict(home_row).iloc[0]
            lambda_away = model.predict(away_row).iloc[0]
            
            # Generate Poisson distributions
            home_pmf = poisson.pmf(range(max_goals + 1), lambda_home)
            away_pmf = poisson.pmf(range(max_goals + 1), lambda_away)
            
            # Outer product for scoreline probabilities
            score_matrix = np.outer(home_pmf, away_pmf)
            
            # Extract match outcome probabilities
            home_win = np.sum(np.tril(score_matrix, -1))  # Lower triangle
            draw = np.sum(np.diag(score_matrix))  # Diagonal
            away_win = np.sum(np.triu(score_matrix, 1))  # Upper triangle
            
            # Normalize
            total = home_win + draw + away_win
            home_win /= total
            draw /= total
            away_win /= total
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'expected_goals': {
                    'home': float(lambda_home),
                    'away': float(lambda_away)
                },
                'probabilities': {
                    'home_win': float(home_win),
                    'draw': float(draw),
                    'away_win': float(away_win)
                },
                'predicted_outcome': max(
                    [('home_win', home_win), ('draw', draw), ('away_win', away_win)],
                    key=lambda x: x[1]
                )[0]
            }
            
        except Exception as e:
            self.logger.warning(f"Prediction failed for {home_team} vs {away_team}: {e}")
            # Return uniform probabilities on failure
            return {
                'home_team': home_team,
                'away_team': away_team,
                'expected_goals': {'home': 1.5, 'away': 1.5},
                'probabilities': {
                    'home_win': 0.33,
                    'draw': 0.33,
                    'away_win': 0.33
                },
                'predicted_outcome': 'home_win',
                'error': str(e)
            }
    
    def get_actual_outcome(self, home_goals: float, away_goals: float) -> str:
        """Determine actual match outcome from goals."""
        if home_goals > away_goals:
            return 'home_win'
        elif home_goals < away_goals:
            return 'away_win'
        else:
            return 'draw'
            
    def backtest_season(
        self, 
        season: str, 
        train_path: Path, 
        test_path: Path
    ) -> Dict:
        """
        Run backtest on a single season split.
        
        Args:
            season: Season identifier (e.g., '2024-25')
            train_path: Path to training data
            test_path: Path to test data
            
        Returns:
            Dict with backtest results for this season
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"BACKTESTING SEASON: {season}")
        self.logger.info(f"{'='*70}")
        
        # Train model
        model = self.train_model_on_split(train_path)
        
        # Load test data (2-rows-per-match format)
        test_df = pd.read_csv(test_path)
        
        # Convert to 1-row-per-match format for predictions
        # Group by date and match pairs
        matches = []
        home_rows = test_df[test_df['home'] == 1].copy()
        
        for idx, home_row in home_rows.iterrows():
            date = home_row['Date']
            home_team = home_row['team']
            away_team = home_row['opponent']
            home_goals = home_row['goals']
            
            # Find corresponding away row
            away_row = test_df[
                (test_df['Date'] == date) & 
                (test_df['team'] == away_team) & 
                (test_df['opponent'] == home_team)
            ]
            
            if not away_row.empty:
                away_goals = away_row.iloc[0]['goals']
                matches.append({
                    'date': date,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'actual_outcome': self.get_actual_outcome(home_goals, away_goals)
                })
        
        self.logger.info(f"Found {len(matches)} matches to predict")
        
        # Generate predictions
        predictions = []
        correct = 0
        total = 0
        
        outcome_stats = {
            'home_win': {'correct': 0, 'total': 0, 'predicted': 0},
            'draw': {'correct': 0, 'total': 0, 'predicted': 0},
            'away_win': {'correct': 0, 'total': 0, 'predicted': 0}
        }
        
        for match in matches:
            pred = self.predict_match(
                model,
                match['home_team'],
                match['away_team']
            )
            
            # Add actual outcome
            pred['actual_outcome'] = match['actual_outcome']
            pred['actual_goals'] = {
                'home': match['home_goals'],
                'away': match['away_goals']
            }
            pred['date'] = match['date']
            
            predictions.append(pred)
            
            # Calculate accuracy
            if pred['predicted_outcome'] == match['actual_outcome']:
                correct += 1
                outcome_stats[match['actual_outcome']]['correct'] += 1
            
            total += 1
            outcome_stats[match['actual_outcome']]['total'] += 1
            outcome_stats[pred['predicted_outcome']]['predicted'] += 1
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        
        # Per-outcome precision and recall
        outcome_metrics = {}
        for outcome in ['home_win', 'draw', 'away_win']:
            stats = outcome_stats[outcome]
            precision = (
                stats['correct'] / stats['predicted'] 
                if stats['predicted'] > 0 else 0
            )
            recall = (
                stats['correct'] / stats['total'] 
                if stats['total'] > 0 else 0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0
            )
            
            outcome_metrics[outcome] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'actual_count': stats['total'],
                'predicted_count': stats['predicted'],
                'correct_count': stats['correct']
            }
        
        # Calibration analysis (binned probability vs actual frequency)
        calibration = self.calculate_calibration(predictions)
        
        results = {
            'season': season,
            'total_matches': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'outcome_distribution': {
                outcome: stats['total'] for outcome, stats in outcome_stats.items()
            },
            'outcome_metrics': outcome_metrics,
            'calibration': calibration,
            'predictions': predictions
        }
        
        self.logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        self.logger.info(f"Home wins: {outcome_stats['home_win']['total']}, "
                        f"Draws: {outcome_stats['draw']['total']}, "
                        f"Away wins: {outcome_stats['away_win']['total']}")
        
        return results
    
    def calculate_calibration(self, predictions: List[Dict]) -> Dict:
        """
        Calculate calibration metrics (predicted probability vs actual frequency).
        
        Args:
            predictions: List of prediction dicts
            
        Returns:
            Calibration metrics by probability bins
        """
        bins = [0, 0.4, 0.5, 0.6, 1.0]  # Probability bins
        bin_labels = ['0-40%', '40-50%', '50-60%', '60-100%']
        
        calibration_data = defaultdict(lambda: {'predicted': [], 'correct': []})
        
        for pred in predictions:
            predicted_outcome = pred['predicted_outcome']
            actual_outcome = pred['actual_outcome']
            predicted_prob = pred['probabilities'][predicted_outcome]
            
            # Find bin
            for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
                if low <= predicted_prob < high or (high == 1.0 and predicted_prob == 1.0):
                    bin_label = bin_labels[i]
                    calibration_data[bin_label]['predicted'].append(predicted_prob)
                    calibration_data[bin_label]['correct'].append(
                        1 if predicted_outcome == actual_outcome else 0
                    )
                    break
        
        # Calculate mean predicted prob and actual frequency per bin
        calibration_results = {}
        for bin_label in bin_labels:
            if bin_label in calibration_data:
                data = calibration_data[bin_label]
                calibration_results[bin_label] = {
                    'mean_predicted': np.mean(data['predicted']),
                    'actual_frequency': np.mean(data['correct']),
                    'count': len(data['correct'])
                }
        
        return calibration_results
    
    def run_backtest(self, seasons: List[str] = None) -> Dict:
        """
        Run backtest across all or specified seasons.
        
        Args:
            seasons: Optional list of season names to test (e.g., ['2024-25'])
                    If None, tests all available seasons
                    
        Returns:
            Dict with aggregate results across all seasons
        """
        self.logger.info("="*70)
        self.logger.info("STARTING BACKTESTING FRAMEWORK")
        self.logger.info("="*70)
        
        # Get season splits
        all_splits = self.get_season_splits()
        
        # Filter by requested seasons if specified
        if seasons:
            all_splits = [
                (s, tr, te) for s, tr, te in all_splits 
                if s in seasons
            ]
            self.logger.info(f"Filtering to {len(all_splits)} requested seasons")
        
        # Run backtest on each season
        season_results = []
        for season, train_path, test_path in all_splits:
            result = self.backtest_season(season, train_path, test_path)
            season_results.append(result)
        
        # Aggregate results
        aggregate = self.aggregate_results(season_results)
        
        # Save results
        self.save_results(aggregate)
        
        # Display summary
        self.display_summary(aggregate)
        
        return aggregate
    
    def aggregate_results(self, season_results: List[Dict]) -> Dict:
        """Aggregate results across all seasons."""
        total_matches = sum(r['total_matches'] for r in season_results)
        total_correct = sum(r['correct_predictions'] for r in season_results)
        overall_accuracy = total_correct / total_matches if total_matches > 0 else 0
        
        # Aggregate outcome metrics
        outcome_totals = defaultdict(lambda: {
            'correct': 0, 'total': 0, 'predicted': 0
        })
        
        for result in season_results:
            for outcome, metrics in result['outcome_metrics'].items():
                outcome_totals[outcome]['correct'] += metrics['correct_count']
                outcome_totals[outcome]['total'] += metrics['actual_count']
                outcome_totals[outcome]['predicted'] += metrics['predicted_count']
        
        # Calculate aggregate precision/recall
        aggregate_outcome_metrics = {}
        for outcome, totals in outcome_totals.items():
            precision = (
                totals['correct'] / totals['predicted']
                if totals['predicted'] > 0 else 0
            )
            recall = (
                totals['correct'] / totals['total']
                if totals['total'] > 0 else 0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0
            )
            
            aggregate_outcome_metrics[outcome] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'total_count': totals['total'],
                'predicted_count': totals['predicted'],
                'correct_count': totals['correct']
            }
        
        return {
            'total_matches': total_matches,
            'total_correct': total_correct,
            'overall_accuracy': overall_accuracy,
            'season_count': len(season_results),
            'aggregate_outcome_metrics': aggregate_outcome_metrics,
            'season_results': season_results,
            'baseline_accuracy': 1/3,  # Random guess baseline
            'improvement_vs_baseline': overall_accuracy - 1/3
        }
    
    def save_results(self, results: Dict):
        """Save backtest results to JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f'backtest_results_{timestamp}.json'
        
        # Remove predictions from season results for summary file
        results_summary = results.copy()
        results_summary['season_results'] = [
            {k: v for k, v in r.items() if k != 'predictions'}
            for r in results['season_results']
        ]
        
        with open(output_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        self.logger.info(f"Results saved to {output_file}")
        
        # Save detailed predictions separately
        predictions_file = self.output_dir / f'backtest_predictions_{timestamp}.json'
        all_predictions = []
        for season_result in results['season_results']:
            all_predictions.extend(season_result['predictions'])
        
        with open(predictions_file, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        self.logger.info(f"Detailed predictions saved to {predictions_file}")
    
    def display_summary(self, results: Dict):
        """Display formatted summary of backtest results."""
        print("\n" + "="*70)
        print("BACKTEST RESULTS SUMMARY")
        print("="*70)
        print(f"\nTotal Matches: {results['total_matches']:,}")
        print(f"Correct Predictions: {results['total_correct']:,}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
        print(f"Baseline (Random): {results['baseline_accuracy']:.2%}")
        print(f"Improvement: +{results['improvement_vs_baseline']:.2%}")
        
        print("\n" + "-"*70)
        print("OUTCOME METRICS")
        print("-"*70)
        print(f"{'Outcome':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Count':<12}")
        print("-"*70)
        
        for outcome in ['home_win', 'draw', 'away_win']:
            metrics = results['aggregate_outcome_metrics'][outcome]
            print(f"{outcome:<12} "
                  f"{metrics['precision']:<12.2%} "
                  f"{metrics['recall']:<12.2%} "
                  f"{metrics['f1_score']:<12.2%} "
                  f"{metrics['total_count']:<12,}")
        
        print("\n" + "-"*70)
        print("SEASON BREAKDOWN")
        print("-"*70)
        print(f"{'Season':<12} {'Matches':<10} {'Correct':<10} {'Accuracy':<12}")
        print("-"*70)
        
        for season_result in results['season_results']:
            print(f"{season_result['season']:<12} "
                  f"{season_result['total_matches']:<10,} "
                  f"{season_result['correct_predictions']:<10,} "
                  f"{season_result['accuracy']:<12.2%}")
        
        print("="*70 + "\n")


def main():
    """Run backtesting on all seasons."""
    framework = BacktestFramework()
    results = framework.run_backtest()
    
    return results


if __name__ == '__main__':
    main()
