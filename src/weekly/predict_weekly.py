"""
Weekly prediction pipeline.

End-to-end system for generating weekly EPL match predictions:
1. Fetch upcoming fixtures
2. Load trained model
3. Generate predictions
4. Format and save results
"""

import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict
import webbrowser

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from weekly.scrape_fixtures import FixtureScraper
from predict.generate_probabilities import PoissonPredictor


class WeeklyPredictor:
    """Complete weekly prediction pipeline."""
    
    # Premier League logo
    PREMIER_LEAGUE_LOGO = 'https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg'
    
    # Team badge URLs from Wikipedia (consistently updated, high quality)
    # Includes all teams from historical data (2005-2026) to cover all possible teams
    TEAM_BADGES = {
        # Current Premier League teams (2025/26)
        'Arsenal': 'https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg',
        'Aston Villa': 'https://upload.wikimedia.org/wikipedia/en/9/9a/Aston_Villa_FC_new_crest.svg',
        'Bournemouth': 'https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg',
        'Brentford': 'https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg',
        'Brighton': 'https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg',
        'Burnley': 'https://upload.wikimedia.org/wikipedia/en/6/6d/Burnley_FC_Logo.svg',
        'Chelsea': 'https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg',
        'Crystal Palace': 'https://upload.wikimedia.org/wikipedia/en/a/a2/Crystal_Palace_FC_logo_%282022%29.svg',
        'Everton': 'https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg',
        'Fulham': 'https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg',
        'Ipswich': 'https://upload.wikimedia.org/wikipedia/en/4/43/Ipswich_Town.svg',
        'Leeds': 'https://upload.wikimedia.org/wikipedia/en/5/54/Leeds_United_F.C._logo.svg',
        'Leicester': 'https://upload.wikimedia.org/wikipedia/en/2/2d/Leicester_City_crest.svg',
        'Liverpool': 'https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg',
        'Luton': 'https://upload.wikimedia.org/wikipedia/en/9/9d/Luton_Town_logo.svg',
        'Man City': 'https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg',
        'Man United': 'https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg',
        'Newcastle': 'https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg',
        "Nott'm Forest": 'https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg',
        'Sheffield Utd': 'https://upload.wikimedia.org/wikipedia/en/9/9c/Sheffield_United_FC_logo.svg',
        'Southampton': 'https://upload.wikimedia.org/wikipedia/en/c/c9/FC_Southampton.svg',
        'Sunderland': 'https://upload.wikimedia.org/wikipedia/en/7/77/Logo_Sunderland.svg',
        'Tottenham': 'https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg',
        'West Ham': 'https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg',
        'Wolves': 'https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg',
        
        # Full name variations (for API compatibility)
        'Manchester City': 'https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg',
        'Manchester United': 'https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg',
        'Newcastle United': 'https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg',
        'Nottingham Forest': 'https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg',
        'Sheffield United': 'https://upload.wikimedia.org/wikipedia/en/9/9c/Sheffield_United_FC_logo.svg',
        'Wolverhampton Wanderers': 'https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg',
        'Leeds United': 'https://upload.wikimedia.org/wikipedia/en/5/54/Leeds_United_F.C._logo.svg',
        'Leicester City': 'https://upload.wikimedia.org/wikipedia/en/2/2d/Leicester_City_crest.svg',
        'Luton Town': 'https://upload.wikimedia.org/wikipedia/en/9/9d/Luton_Town_logo.svg',
        
        # Historical teams (2005-2024)
        'Barnsley': 'https://upload.wikimedia.org/wikipedia/en/c/c9/Barnsley_FC.svg',
        'Birmingham': 'https://upload.wikimedia.org/wikipedia/en/6/68/Birmingham_City_FC_logo.svg',
        'Blackburn': 'https://upload.wikimedia.org/wikipedia/en/0/0f/Blackburn_Rovers.svg',
        'Blackpool': 'https://upload.wikimedia.org/wikipedia/en/d/df/Blackpool_FC_logo.svg',
        'Bolton': 'https://upload.wikimedia.org/wikipedia/en/8/82/Bolton_Wanderers_FC_logo.svg',
        'Cardiff': 'https://upload.wikimedia.org/wikipedia/en/3/3c/Cardiff_City_crest.svg',
        'Charlton': 'https://upload.wikimedia.org/wikipedia/en/5/5c/Charlton_Athletic.svg',
        'Derby': 'https://upload.wikimedia.org/wikipedia/en/4/4a/Derby_County_crest.svg',
        'Huddersfield': 'https://upload.wikimedia.org/wikipedia/en/7/7d/Huddersfield_Town_A.F.C._logo.svg',
        'Hull': 'https://upload.wikimedia.org/wikipedia/en/5/54/Hull_City_A.F.C._logo.svg',
        'Middlesbrough': 'https://upload.wikimedia.org/wikipedia/en/2/2c/Middlesbrough_FC_crest.svg',
        'Norwich': 'https://upload.wikimedia.org/wikipedia/en/8/8c/Norwich_City.svg',
        'Portsmouth': 'https://upload.wikimedia.org/wikipedia/en/3/38/Portsmouth_FC_crest.svg',
        'QPR': 'https://upload.wikimedia.org/wikipedia/en/3/31/Queens_Park_Rangers_crest.svg',
        'Reading': 'https://upload.wikimedia.org/wikipedia/en/1/11/Reading_FC.svg',
        'Stoke': 'https://upload.wikimedia.org/wikipedia/en/2/29/Stoke_City_FC.svg',
        'Swansea': 'https://upload.wikimedia.org/wikipedia/en/f/f9/Swansea_City_AFC_logo.svg',
        'Swindon': 'https://upload.wikimedia.org/wikipedia/en/7/77/Swindon_Town_FC.svg',
        'Watford': 'https://upload.wikimedia.org/wikipedia/en/e/e2/Watford.svg',
        'West Brom': 'https://upload.wikimedia.org/wikipedia/en/8/8b/West_Bromwich_Albion.svg',
        'Wigan': 'https://upload.wikimedia.org/wikipedia/en/4/43/Wigan_Athletic.svg',
    }
    
    def __init__(
        self,
        model_path: str = 'models/poisson_glm_model.pkl',
        coefficients_path: str = 'models/poisson_coefficients.json',
        output_dir: str = 'data/weekly'
    ):
        """
        Initialize weekly predictor.
        
        Args:
            model_path: Path to trained Poisson GLM model
            coefficients_path: Path to model coefficients JSON
            output_dir: Directory to save prediction outputs
        """
        self.model_path = model_path
        self.coefficients_path = coefficients_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.fixture_scraper = FixtureScraper()
        self.predictor = None
        
    def load_predictor(self):
        """Load trained model for predictions."""
        print("Loading trained model...")
        self.predictor = PoissonPredictor(
            model_path=self.model_path,
            coefficients_path=self.coefficients_path
        )
        print("✅ Model loaded successfully")
    
    def fetch_fixtures(self, matchweek: int = None) -> List[Dict]:
        """
        Fetch fixtures for prediction.
        
        Args:
            matchweek: Specific matchweek or None for upcoming
            
        Returns:
            List of fixture dictionaries
        """
        print(f"\n{'='*70}")
        print(f"FETCHING FIXTURES")
        print(f"{'='*70}")
        
        fixtures = self.fixture_scraper.fetch_fixtures(matchweek)
        
        print(f"\nFound {len(fixtures)} fixtures for prediction:")
        for i, fixture in enumerate(fixtures, 1):
            print(f"  {i}. {fixture['home_team']} vs {fixture['away_team']}")
        
        return fixtures
    
    def generate_predictions(self, fixtures: List[Dict]) -> List[Dict]:
        """
        Generate predictions for all fixtures.
        
        Args:
            fixtures: List of fixture dictionaries
            
        Returns:
            List of prediction dictionaries with fixture details
        """
        if self.predictor is None:
            self.load_predictor()
        
        print(f"\n{'='*70}")
        print(f"GENERATING PREDICTIONS")
        print(f"{'='*70}\n")
        
        predictions = []
        
        for fixture in fixtures:
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            
            # Generate prediction
            pred = self.predictor.predict_match(home_team, away_team)
            
            # Add fixture metadata
            pred['fixture_info'] = {
                'matchweek': fixture.get('matchweek'),
                'date': fixture.get('date'),
                'time': fixture.get('time'),
                'venue': fixture.get('venue')
            }
            
            predictions.append(pred)
            
            # Print quick summary
            probs = pred['probabilities']
            print(f"✓ {home_team} vs {away_team}")
            print(f"  Home: {probs['home_win']:.1%} | "
                  f"Draw: {probs['draw']:.1%} | "
                  f"Away: {probs['away_win']:.1%}")
            print(f"  Most likely: {pred['most_likely_scoreline']}")
            print()
        
        return predictions
    
    def save_predictions(
        self, 
        predictions: List[Dict],
        matchweek: int = None
    ) -> Dict[str, Path]:
        """
        Save predictions as HTML report only and auto-open in browser.
        
        Args:
            predictions: List of prediction dictionaries
            matchweek: Matchweek number for filename
            
        Returns:
            Dict with 'html' key mapping to output file path
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if matchweek:
            base_name = f'predictions_mw{matchweek}_{timestamp}'
        else:
            base_name = f'predictions_{timestamp}'
        
        # Save HTML only
        html_file = self.output_dir / f'{base_name}.html'
        self._save_html_report(predictions, html_file, matchweek)
        
        # Auto-open HTML in browser
        try:
            webbrowser.open(f'file://{html_file.absolute()}')
        except Exception as e:
            print(f"⚠️  Could not auto-open HTML: {e}")
        
        return {'html': html_file}
    
    def _save_csv_summary(self, predictions: List[Dict], output_file: Path):
        """Save predictions as CSV summary table."""
        rows = []
        
        for pred in predictions:
            fixture_info = pred.get('fixture_info', {})
            probs = pred['probabilities']
            xg = pred['expected_goals']
            
            # Determine predicted outcome from probabilities
            outcome_probs = [
                ('Home Win', probs['home_win']),
                ('Draw', probs['draw']),
                ('Away Win', probs['away_win'])
            ]
            predicted_outcome = max(outcome_probs, key=lambda x: x[1])[0]
            
            # Format most likely scoreline
            ml_score = pred.get('most_likely_scoreline', {})
            if isinstance(ml_score, dict):
                scoreline = f"{ml_score.get('home_goals', '?')}-{ml_score.get('away_goals', '?')}"
            else:
                scoreline = str(ml_score)
            
            rows.append({
                'Date': fixture_info.get('date', ''),
                'Time': fixture_info.get('time', ''),
                'Home Team': pred['home_team'],
                'Away Team': pred['away_team'],
                'Home xG': f"{xg['home']:.2f}",
                'Away xG': f"{xg['away']:.2f}",
                'Home Win %': f"{probs['home_win']*100:.1f}",
                'Draw %': f"{probs['draw']*100:.1f}",
                'Away Win %': f"{probs['away_win']*100:.1f}",
                'Most Likely Score': scoreline,
                'Predicted Outcome': predicted_outcome
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
    
    def _get_team_badge(self, team_name: str) -> str:
        """Get badge URL for team, with fallback to generic badge."""
        return self.TEAM_BADGES.get(team_name, 'https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg')
    
    def _save_html_report(
        self, 
        predictions: List[Dict], 
        output_file: Path,
        matchweek: int = None
    ):
        """Generate HTML report with formatted predictions."""
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EPL Predictions - Matchweek {matchweek}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }}
        
        .header .pl-logo {{
            width: 80px;
            height: auto;
            margin-bottom: 15px;
        }}
        
        .header h1 {{
            color: #2d3748;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            color: #718096;
            font-size: 1.2em;
        }}
        
        .stats-bar {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            display: flex;
            justify-content: space-around;
            text-align: center;
        }}
        
        .stat {{
            flex: 1;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            color: #718096;
            margin-top: 5px;
        }}
        
        .match-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .match-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        
        .match-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
        }}
        
        .match-date {{
            color: #718096;
            font-size: 0.9em;
        }}
        
        .teams {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .team {{
            flex: 1;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .team-badge {{
            width: 80px;
            height: 80px;
            object-fit: contain;
            margin-bottom: 10px;
        }}
        
        .team-name {{
            font-size: 1.4em;
            font-weight: bold;
            color: #2d3748;
            margin-bottom: 5px;
        }}
        
        .team-xg {{
            color: #718096;
            font-size: 0.95em;
        }}
        
        .vs {{
            font-size: 1.2em;
            color: #a0aec0;
            font-weight: bold;
            padding: 0 20px;
        }}
        
        .probabilities {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }}
        
        .prob-bar {{
            flex: 1;
            text-align: center;
            padding: 12px;
            border-radius: 8px;
            color: white;
            font-weight: bold;
        }}
        
        .prob-home {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        
        .prob-draw {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        
        .prob-away {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        
        .prob-label {{
            font-size: 0.85em;
            opacity: 0.9;
        }}
        
        .prob-value {{
            font-size: 1.3em;
            margin-top: 5px;
        }}
        
        .prediction-summary {{
            background: #f7fafc;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .prediction-summary strong {{
            color: #2d3748;
        }}
        
        .scoreline {{
            font-size: 1.2em;
            color: #667eea;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        .footer {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            margin-top: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
            color: #718096;
        }}
        
        .top-scorelines {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
            justify-content: center;
        }}
        
        .scoreline-badge {{
            background: #edf2f7;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            color: #4a5568;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="{pl_logo}" alt="Premier League" class="pl-logo">
            <h1>PL Match Predictions</h1>
            <div class="subtitle">Matchweek {matchweek} · {date}</div>
        </div>
        
        <div class="stats-bar">
            <div class="stat">
                <div class="stat-value">{total_matches}</div>
                <div class="stat-label">Fixtures</div>
            </div>
            <div class="stat">
                <div class="stat-value">51.95%</div>
                <div class="stat-label">Model Accuracy</div>
            </div>
            <div class="stat">
                <div class="stat-value">ξ=0.003</div>
                <div class="stat-label">Time-Weighted</div>
            </div>
        </div>
        
        {match_cards}
        
        <div class="footer">
            <p>Predictions generated using Time-Weighted Poisson GLM (Dixon-Coles) trained on {training_matches} historical matches</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Model: 51.95% accuracy · Time-weighting: ξ=0.003 · Validated on 1,900 test matches (5-fold CV)
            </p>
            <p style="margin-top: 10px; font-size: 0.85em; opacity: 0.7;">
                Generated on {timestamp}
            </p>
        </div>
    </div>
</body>
</html>
        """
        
        # Generate match cards HTML
        match_cards_html = []
        
        for pred in predictions:
            fixture_info = pred.get('fixture_info', {})
            probs = pred['probabilities']
            xg = pred['expected_goals']
            
            date_str = fixture_info.get('date', '')
            time_str = fixture_info.get('time', '')
            
            # Get team badge URLs
            home_badge = self._get_team_badge(pred['home_team'])
            away_badge = self._get_team_badge(pred['away_team'])
            
            # Format most likely scoreline
            ml_score = pred.get('most_likely_scoreline', {})
            if isinstance(ml_score, dict):
                scoreline = f"{ml_score.get('home_goals', '?')}-{ml_score.get('away_goals', '?')}"
            else:
                scoreline = str(ml_score)
            
            # Format top 3 scorelines
            top_3 = pred.get('top_3_scorelines', [])
            scorelines_html = ''.join([
                f'<span class="scoreline-badge">{s.get("home_goals", "?")}-{s.get("away_goals", "?")} ({s.get("probability", 0)*100:.1f}%)</span>'
                for s in top_3[:3]
            ])
            
            card_html = f"""
        <div class="match-card">
            <div class="match-header">
                <div class="match-date">{date_str} · {time_str}</div>
                <div class="match-date">{fixture_info.get('venue', '')}</div>
            </div>
            
            <div class="teams">
                <div class="team">
                    <img src="{home_badge}" alt="{pred['home_team']}" class="team-badge" onerror="this.src='https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg'">
                    <div class="team-name">{pred['home_team']}</div>
                    <div class="team-xg">xG: {xg['home']:.2f}</div>
                </div>
                <div class="vs">VS</div>
                <div class="team">
                    <img src="{away_badge}" alt="{pred['away_team']}" class="team-badge" onerror="this.src='https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg'">
                    <div class="team-name">{pred['away_team']}</div>
                    <div class="team-xg">xG: {xg['away']:.2f}</div>
                </div>
            </div>
            
            <div class="probabilities">
                <div class="prob-bar prob-home">
                    <div class="prob-label">Home Win</div>
                    <div class="prob-value">{probs['home_win']*100:.1f}%</div>
                </div>
                <div class="prob-bar prob-draw">
                    <div class="prob-label">Draw</div>
                    <div class="prob-value">{probs['draw']*100:.1f}%</div>
                </div>
                <div class="prob-bar prob-away">
                    <div class="prob-label">Away Win</div>
                    <div class="prob-value">{probs['away_win']*100:.1f}%</div>
                </div>
            </div>
            
            <div class="prediction-summary">
                <strong>Most Likely:</strong>
                <span class="scoreline">{scoreline}</span>
                <div class="top-scorelines">
                    {scorelines_html}
                </div>
            </div>
        </div>
            """
            
            match_cards_html.append(card_html)
        
        # Fill template
        html_content = html_template.format(
            pl_logo=self.PREMIER_LEAGUE_LOGO,
            matchweek=matchweek or 'Upcoming',
            date=datetime.now().strftime('%B %d, %Y'),
            total_matches=len(predictions),
            training_matches='4,270',
            match_cards=''.join(match_cards_html),
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def run_weekly_predictions(self, matchweek: int = None) -> Dict:
        """
        Run complete weekly prediction pipeline.
        
        Args:
            matchweek: Specific matchweek or None for upcoming
            
        Returns:
            Dict with predictions and output file paths
        """
        print("\n" + "="*70)
        print("EPL WEEKLY PREDICTIONS PIPELINE")
        print("="*70)
        
        # Step 1: Fetch fixtures
        fixtures = self.fetch_fixtures(matchweek)
        
        if not fixtures:
            print("\n❌ No fixtures found!")
            return {'fixtures': [], 'predictions': [], 'output_files': {}}
        
        # Step 2: Generate predictions
        predictions = self.generate_predictions(fixtures)
        
        # Step 3: Save outputs
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}\n")
        
        mw = fixtures[0].get('matchweek') if fixtures else None
        output_files = self.save_predictions(predictions, mw)
        
        print(f"✅ HTML Report: {output_files['html']}")
        print(f"🌐 Opening in browser...")
        
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE!")
        print(f"{'='*70}")
        
        return {
            'fixtures': fixtures,
            'predictions': predictions,
            'output_files': output_files
        }


def main():
    """Run weekly prediction pipeline."""
    import sys
    
    # Parse matchweek argument if provided
    matchweek = None
    if len(sys.argv) > 1:
        try:
            matchweek = int(sys.argv[1])
        except ValueError:
            print(f"Warning: Invalid matchweek '{sys.argv[1]}', using all upcoming fixtures")
    
    predictor = WeeklyPredictor()
    results = predictor.run_weekly_predictions(matchweek=matchweek)
    
    print(f"\n✅ Generated predictions for {len(results['predictions'])} matches")
    print(f"\n📂 Output files:")
    for format_name, file_path in results['output_files'].items():
        print(f"   {format_name}: {file_path}")
    
    return results


if __name__ == '__main__':
    main()
