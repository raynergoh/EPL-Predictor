"""
Phase 3: Prediction Pipeline - Poisson Outer Product

This module generates match outcome predictions using the Poisson distribution
and the outer product technique to create scoreline probability matrices.

Key Components:
1. Load trained Poisson GLM model and coefficients
2. Generate expected goals for both teams
3. Create Poisson probability distributions
4. Build scoreline probability matrix (7×7)
5. Extract win/draw/loss probabilities
6. Identify most likely scorelines

Mathematical Basis:
- Expected goals (λ) calculated from model: log(λ) = model prediction
- Scoreline probability: P(i,j) = P(home_goals=i) × P(away_goals=j)
- Win probability: sum of matrix below diagonal
- Draw probability: sum of matrix diagonal
- Loss probability: sum of matrix above diagonal
"""

import logging
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from scipy.stats import poisson
from datetime import datetime
from typing import Dict, Tuple, List

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Setup logging
log_file = LOGS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Core Probability Generation
# ============================================================================

class PoissonPredictor:
    """Generate match predictions using Poisson regression."""
    
    def __init__(self, model_path=None, coefficients_path=None):
        """
        Initialize predictor with trained model and coefficients.
        
        Args:
            model_path: Path to trained model pickle file (str or Path)
            coefficients_path: Path to coefficients JSON file (str or Path)
        """
        # Convert to Path objects if strings provided
        if model_path is None:
            self.model_path = MODELS_DIR / "poisson_glm_model.pkl"
        else:
            self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        
        if coefficients_path is None:
            self.coefficients_path = MODELS_DIR / "poisson_coefficients.json"
        else:
            self.coefficients_path = Path(coefficients_path) if isinstance(coefficients_path, str) else coefficients_path
        
        self.model = None
        self.coefficients = None
        
        self.load_model()
        
        logger.info("✓ PoissonPredictor initialized")
    
    def load_model(self):
        """Load trained model and coefficients."""
        logger.info(f"Loading model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info("✓ Model loaded")
        
        if self.coefficients_path.exists():
            with open(self.coefficients_path, 'r') as f:
                self.coefficients = json.load(f)
            logger.info("✓ Coefficients loaded")
    
    def get_expected_goals(self, team: str, opponent: str, home: bool) -> float:
        """
        Calculate expected goals using model prediction.
        Handles unknown teams by using league-average coefficients.
        
        Note: In the Poisson GLM, one team (the baseline/reference category)
        has an implicit coefficient of 0 and won't appear in the coefficient
        dictionaries. This team is still "known" and can be predicted normally.
        
        Args:
            team: Team name
            opponent: Opponent team name
            home: True if home team, False if away
        
        Returns:
            Expected goals (λ) for the team
        """
        # Get all teams that appear in the training data
        # This includes teams with explicit coefficients AND the baseline team
        teams_with_attack = set(self.coefficients.get('team_attack', {}).keys())
        teams_with_defense = set(self.coefficients.get('opponent_defense', {}).keys())
        
        # All teams in the model (those with coefficients)
        teams_with_coefficients = teams_with_attack | teams_with_defense
        
        # Try to predict using the model directly for both known AND baseline teams
        # The model's predict() function handles baseline teams automatically
        try:
            match_data = pd.DataFrame({
                'home': [1 if home else 0],
                'team': [team],
                'opponent': [opponent]
            })
            lambda_val = self.model.predict(match_data).values[0]
            return max(lambda_val, 0.01)
        except Exception as e:
            # Only fall back to league-average if prediction fails
            # This catches truly unknown teams that aren't in the training data
            logger.warning(f"⚠ Unknown team detected: {team}")
            logger.info(f"  Using league-average coefficients for prediction")
            logger.debug(f"  Prediction error: {e}")
            
            # Get baseline from intercept and home advantage
            intercept = self.coefficients.get('intercept', 1.35)
            home_coef = self.coefficients.get('home_advantage', 0.20) if home else 0
            
            # Use league-average team effects (mean of all coefficients)
            team_attack = list(self.coefficients.get('team_attack', {}).values())
            opponent_defense = list(self.coefficients.get('opponent_defense', {}).values())
            
            avg_team_attack = sum(team_attack) / len(team_attack) if team_attack else 0
            avg_opponent_defense = sum(opponent_defense) / len(opponent_defense) if opponent_defense else 0
            
            # Use actual coefficients for known teams, average for unknown
            # Note: baseline team (e.g. Arsenal) has implicit coefficient of 0
            team_coef = self.coefficients.get('team_attack', {}).get(team, avg_team_attack)
            opponent_coef = self.coefficients.get('opponent_defense', {}).get(opponent, avg_opponent_defense)
            
            # Calculate lambda using manual coefficient combination
            log_lambda = intercept + home_coef + team_coef + opponent_coef
            lambda_val = np.exp(log_lambda)
        
        return max(lambda_val, 0.01)  # Avoid zero or negative values
    
    def predict_match(self, home_team: str, away_team: str, 
                     max_goals: int = 6) -> Dict:
        """
        Generate complete match prediction with probabilities.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            max_goals: Maximum goals to consider (default 6 for reasonable tail)
        
        Returns:
            Dictionary with predictions
        """
        logger.info(f"\nPredicting: {home_team} (H) vs {away_team} (A)")
        
        # Validate team names
        try:
            # Get expected goals
            lambda_home = self.get_expected_goals(home_team, away_team, home=True)
            lambda_away = self.get_expected_goals(away_team, home_team, home=False)
        except Exception as e:
            logger.error(f"✗ Error predicting {home_team} vs {away_team}: {str(e)}")
            logger.error(f"  Please ensure team names match training data exactly")
            raise
        
        logger.info(f"  Expected goals: {home_team}={lambda_home:.3f}, {away_team}={lambda_away:.3f}")
        
        # Generate probability distributions (0 to max_goals)
        home_pmf = poisson.pmf(range(max_goals + 1), lambda_home)
        away_pmf = poisson.pmf(range(max_goals + 1), lambda_away)
        
        # Outer product: scoreline probability matrix
        score_matrix = np.outer(home_pmf, away_pmf)
        
        # Calculate outcome probabilities
        # Home win: home_goals > away_goals (lower triangle excluding diagonal)
        home_win = np.sum(np.tril(score_matrix, -1))
        # Draw: home_goals == away_goals (diagonal)
        draw = np.sum(np.diag(np.diag(score_matrix)))
        # Away win: away_goals > home_goals (upper triangle excluding diagonal)
        away_win = np.sum(np.triu(score_matrix, 1))
        
        # Normalize to ensure sum = 1.0
        total = home_win + draw + away_win
        home_win /= total
        draw /= total
        away_win /= total
        
        # Find most likely scorelines
        most_likely_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        most_likely_score = (most_likely_idx[0], most_likely_idx[1])
        most_likely_prob = score_matrix[most_likely_idx]
        
        # Get top 3 scorelines
        flat_probs = score_matrix.flatten()
        top_3_indices = np.argsort(flat_probs)[-3:][::-1]
        top_3_scorelines = [
            {
                'home_goals': int(idx // (max_goals + 1)),
                'away_goals': int(idx % (max_goals + 1)),
                'probability': float(flat_probs[idx])
            }
            for idx in top_3_indices
        ]
        
        result = {
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
            'most_likely_scoreline': {
                'home_goals': int(most_likely_score[0]),
                'away_goals': int(most_likely_score[1]),
                'probability': float(most_likely_prob)
            },
            'top_3_scorelines': top_3_scorelines,
            'score_matrix': [[float(x) for x in row] for row in score_matrix],
            'timestamp': datetime.now().isoformat()
        }
        
        # Log results
        logger.info(f"  Probabilities: {home_team} win={home_win:.1%}, Draw={draw:.1%}, {away_team} win={away_win:.1%}")
        logger.info(f"  Most likely: {home_team} {most_likely_score[0]}-{most_likely_score[1]} {away_team} ({most_likely_prob:.2%})")
        
        return result


# ============================================================================
# Batch Predictions
# ============================================================================

def predict_fixtures(fixtures: List[Dict], predictor: PoissonPredictor = None) -> List[Dict]:
    """
    Generate predictions for multiple fixtures.
    
    Args:
        fixtures: List of fixture dicts with 'home_team' and 'away_team'
        predictor: Optional PoissonPredictor instance (creates new if None)
    
    Returns:
        List of prediction results
    """
    if predictor is None:
        predictor = PoissonPredictor()
    
    results = []
    for fixture in fixtures:
        prediction = predictor.predict_match(
            fixture['home_team'],
            fixture['away_team']
        )
        results.append(prediction)
    
    return results


# ============================================================================
# Output Formatting
# ============================================================================

def format_prediction_table(prediction: Dict) -> str:
    """Format a single prediction as a readable table."""
    home = prediction['home_team']
    away = prediction['away_team']
    
    output = f"\n{'='*60}\n"
    output += f"{home:20s} vs {away:20s}\n"
    output += f"{'='*60}\n"
    
    # Expected goals
    eg_home = prediction['expected_goals']['home']
    eg_away = prediction['expected_goals']['away']
    output += f"Expected Goals:        {eg_home:>6.2f}  -  {eg_away:<6.2f}\n"
    
    # Outcome probabilities
    prob = prediction['probabilities']
    output += f"\n{'Outcome':<20} {'Probability':<15} {'Odds':<10}\n"
    output += f"{'-'*45}\n"
    output += f"{home:>20} {prob['home_win']:>14.1%}  ({1/max(prob['home_win'],0.001):>6.2f}x)\n"
    output += f"{'Draw':>20} {prob['draw']:>14.1%}  ({1/max(prob['draw'],0.001):>6.2f}x)\n"
    output += f"{away:>20} {prob['away_win']:>14.1%}  ({1/max(prob['away_win'],0.001):>6.2f}x)\n"
    
    # Most likely scoreline
    ml = prediction['most_likely_scoreline']
    output += f"\nMost Likely Scoreline: {ml['home_goals']}-{ml['away_goals']} ({ml['probability']:.2%})\n"
    
    # Top 3 scorelines
    output += f"\nTop 3 Scorelines:\n"
    for i, score in enumerate(prediction['top_3_scorelines'], 1):
        output += f"  {i}. {score['home_goals']}-{score['away_goals']:>2} ({score['probability']:>6.2%})\n"
    
    output += f"{'='*60}\n"
    
    return output


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Demo: Test predictions on example matches."""
    logger.info("\n" + "="*70)
    logger.info("PHASE 3: POISSON PREDICTION PIPELINE")
    logger.info("="*70)
    
    # Initialize predictor
    predictor = PoissonPredictor()
    
    # Example matches (using correct team names from training data)
    fixtures = [
        {'home_team': 'Man City', 'away_team': 'West Ham'},
        {'home_team': 'Liverpool', 'away_team': 'Chelsea'},
        {'home_team': 'Arsenal', 'away_team': 'Man United'},
        {'home_team': 'Newcastle', 'away_team': 'Tottenham'},
    ]
    
    logger.info(f"\nGenerating predictions for {len(fixtures)} matches...")
    predictions = predict_fixtures(fixtures, predictor)
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("PREDICTIONS")
    logger.info("="*70)
    
    for pred in predictions:
        table = format_prediction_table(pred)
        logger.info(table)
        print(table)  # Also print to console
    
    # Save predictions to JSON
    output_file = PROJECT_ROOT / "data" / "processed" / "predictions.json"
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"✓ Predictions saved to {output_file}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ PHASE 3: PREDICTIONS COMPLETE")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
