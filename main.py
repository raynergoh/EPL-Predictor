"""
EPL Match Predictor - Main Entry Point

Single command to:
1. Fetch latest match results from current season
2. Update model with all historical data (including current season)
3. Generate predictions for next upcoming matchweek

Usage:
    python main.py                  # Predict next upcoming matchweek
    python main.py --matchweek 11   # Predict specific matchweek
    python main.py --retrain        # Force full model retraining
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.fetch_historical_data import fetch_and_update_data
from train.train_poisson_model import train_poisson_model
from weekly.predict_weekly import WeeklyPredictor


def setup_logging():
    """Setup logging configuration."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'main_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def check_model_needs_update(data_path: str = 'data/raw/epl_historical_results.csv') -> bool:
    """
    Check if model needs retraining based on latest data.
    
    Returns True if:
    - Model file doesn't exist
    - New match data is available since last training
    """
    model_file = Path('models/poisson_glm_model.pkl')
    
    # If model doesn't exist, needs training
    if not model_file.exists():
        return True
    
    # Check if data file is newer than model
    data_file = Path(data_path)
    if not data_file.exists():
        return True
        
    model_time = model_file.stat().st_mtime
    data_time = data_file.stat().st_mtime
    
    # If data is newer than model, needs retraining
    return data_time > model_time


def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(
        description='EPL Match Predictor - Generate predictions with latest data'
    )
    parser.add_argument(
        '--matchweek',
        type=int,
        help='Specific matchweek to predict (default: next upcoming matchweek)'
    )
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Force full model retraining even if not needed'
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    
    print("=" * 80)
    print("EPL MATCH PREDICTOR")
    print("=" * 80)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Fetch latest match results
        logger.info("STEP 1: Fetching latest match results...")
        
        try:
            updated = fetch_and_update_data()
        except Exception as e:
            logger.error(f"Fatal error fetching data: {e}")
            print(f"❌ Cannot proceed without data file")
            return 1
        
        print()
        
        # Step 2: Check if model needs retraining
        logger.info("STEP 2: Checking if model needs retraining...")
        needs_update = check_model_needs_update()
        
        if args.retrain or needs_update:
            if args.retrain:
                print("🔄 Forced retraining requested...")
            else:
                print("🔄 New data detected, model needs retraining...")
                
            logger.info("Training model with ALL historical data...")
            print("🤖 Training Poisson GLM model...")
            print("   (This includes all historical data + current season results)")
            print()
            
            # Train model with all data
            fitted_model, coefficients = train_poisson_model()
            
            print()
            print("✓ Model training complete!")
            print(f"   - Trained on {coefficients.get('total_matches', 'N/A')} matches")
            print(f"   - {len(coefficients['team_attack'])} teams with attack coefficients")
            print(f"   - Home advantage: {coefficients['home_advantage']:.4f}")
        else:
            print("✓ Model is up to date (no retraining needed)")
        
        print()
        
        # Step 3: Generate predictions
        logger.info("STEP 3: Generating predictions...")
        print("⚽ Generating match predictions...")
        
        predictor = WeeklyPredictor()
        
        if args.matchweek:
            print(f"   Target: Matchweek {args.matchweek}")
            results = predictor.run_weekly_predictions(matchweek=args.matchweek)
        else:
            print("   Target: Next upcoming matchweek")
            results = predictor.run_weekly_predictions()
        
        print()
        print("=" * 80)
        print("PREDICTIONS COMPLETE")
        print("=" * 80)
        
        # Display summary
        num_predictions = len(results.get('predictions', []))
        print(f"\n✓ Generated {num_predictions} match predictions")
        
        # Show output files
        output_files = results.get('output_files', {})
        if output_files:
            print("\n📄 Output files:")
            for format_type, filepath in output_files.items():
                print(f"   - {format_type.upper()}: {filepath}")
                
        # Show HTML file specifically
        html_file = output_files.get('html')
        if html_file:
            print("\n🌐 View predictions in browser:")
            print(f"   open {html_file}")
            
        print()
        logger.info("Pipeline completed successfully")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        logger.warning("Pipeline interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
