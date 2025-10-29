# AI Copilot Instructions - EPL-Predictor

This is a **Poisson regression-based football prediction system** that forecasts EPL match outcomes using lineup-based expected goals (xG), rolling team form, and historical data.

## Architecture Overview

The project follows a **data pipeline architecture** with five key stages:

1. **Data Collection** (`src/scraping/`, `src/data/`) → JSON/CSV
2. **Feature Engineering** (`src/processing/`, `src/features/`) → Fixture-level features
3. **Model Training** (`src/train/`) → Pickle-serialized Poisson models
4. **Inference** (`src/predict/`) → Match predictions & probabilities
5. **Main Orchestrator** (`src/main.py`) → Runs entire pipeline end-to-end

### Data Flow

```
scrape_fixtures_lineups() → team_fixtures_lineups.json
  ↓ (player name cleaning via cleaning.py)
calculate_team_xg() → team_xg_estimates.json (per-team, per-match xG)
  ↓ (merge home/away)
team_fixture_xg_estimates.json → predict_upcoming_matches()
  ↓ (load rolling form stats from historical data)
Predict match outcome + scoreline probabilities
```

**Key Files by Stage:**
- Scraping: `src/scraping/scrape_fixtures_lineups.py` (Fantasy Football Scout)
- xG Calculation: `src/processing/calculate_xg.py` (player stats → team xG with H/A adjustment)
- Feature Building: `src/features/build_features.py` (historical xG + rolling form stats)
- Model Training: `src/train/poisson_model.py` (Poisson GLM via statsmodels)
- Prediction: `src/predict/predict_upcoming.py` (uses simulate_match.py for outcome probabilities)

## Critical Patterns & Conventions

### 1. Data Column Naming & Conventions

- **Historical results CSV** (`data/raw/epl_historical_results.csv`): `home_goals`, `away_goals`, `home_xg`, `away_xg`, `home_team`, `away_team`, `date`
- **Fixture-level xG JSON** formats as list of dicts: `home_team`, `away_team`, `home_avg_xg`, `away_avg_xg`, `home_predicted_lineup`, `away_predicted_lineup`
- **Team xG estimates** format: `team`, `opponent`, `home_away` (H/A flag), `team_xg_avg_adj` (adjusted average xG)
- **Rolling form features** use prefixes: `home_form_*` and `away_form_*` (e.g., `home_form_goals_scored`)
- **Home field indicator**: Always `home_field = 1` in feature matrix

### 2. Home/Away Adjustment Pattern

xG values are adjusted by `1.10` (home) or `0.95` (away) in `calculate_team_xg()`. This adjustment is **baked into** xG estimates before merging fixtures. Ensure this is applied once per team per match, not duplicated in feature engineering.

### 3. Rolling Window Feature Engineering

`build_features.py` and `predict_upcoming.py` both compute rolling averages independently:
- `.groupby('team')` → separate home/away stats
- `.rolling(window).mean().shift(1)` — **shift(1) is critical**: excludes current match to prevent data leakage
- Default `rolling_window=5` (configurable parameter)
- Missing form stats at season start filled with **median imputation**

### 4. Model Input/Output Contract

**Training**: Features → `home_avg_xg`, `away_avg_xg`, `home_field`, `home_form_*`, `away_form_*` → Poisson GLM with formula-based design matrix
**Prediction**: Same feature schema required. Mismatch will raise `KeyError` or fail prediction silently.

Models (`model_home.pkl`, `model_away.pkl`) are serialized with `pickle`. Regenerate when feature schema changes.

### 5. String Cleaning & Name Mapping

Use utilities in `src/utils/cleaning.py`:
- `clean_team_name()`: Handles known aliases (e.g., "Brighton and Hove Albion" → "Brighton")
- `clean_player_name()`: Parses "Surname (FirstName)" → "FirstName Surname"; maintains manual override map
- Player xG lookups are **case-sensitive** and require exact name matches

### 6. Prediction Probability Computation

`simulate_match.py` uses Poisson PMF to generate outcome probabilities:
- `lambda_home`, `lambda_away` = predicted expected goals (Poisson μ)
- Probability matrix = outer product of two Poisson PMFs (0-6 goals)
- Probabilities computed via: `home_win = lower_triangle_sum`, `draw = trace`, `away_win = upper_triangle_sum`
- Output: Full 7×7 matrix + most likely scoreline + top-3 scorelines

## Workflow Commands

All commands run from repository root:

```bash
# Install dependencies
pip install -r requirements.txt

# Download historical EPL data (Understat async)
python -m src.data.extract_historical_data

# Build feature matrix from historical + xG
python -m src.features.build_features

# Train Poisson models (GLM via statsmodels)
python -m src.train.poisson_model

# End-to-end: scrape → xG → predict (recommended)
python -m src.main

# Single prediction (requires models + historical data)
python -m src.predict.predict_upcoming
```

## Common Tasks & How-Tos

**Add new rolling form feature:**
1. Edit `build_features.py`: add computation in `create_team_stats()` groupby chain
2. Ensure name follows `form_*` pattern
3. Add to `feature_cols` list
4. Retrain models

**Update historical data after matchweek:**
1. Add new rows to `data/raw/epl_historical_results.csv`
2. Rerun `python -m src.features.build_features` (regenerates rolling stats)
3. Rerun `python -m src.train.poisson_model` (retrain with full data)
4. Then `python -m src.main` for next predictions

**Debug missing player xG:**
- Check `data/raw/understat_player_xg.csv` for name match
- Verify scraped lineup names via `scrape_fixtures_lineups.py` output
- Add manual mapping to `clean_player_name()` if name format differs

**Analyze prediction outputs:**
- Full scoreline probability matrix in `score_probabilities` (7×7 numpy array)
- Most likely scoreline = argmax of matrix
- Win probabilities sum to 1.0 (sanity check)

## External Dependencies & APIs

- **BeautifulSoup4 + requests**: Scrapes Fantasy Football Scout for predicted lineups
- **Understat library**: Async fetch of EPL historical xG data (requires internet)
- **statsmodels**: GLM Poisson regression (formula-based interface)
- **pandas, numpy, scipy.stats**: Data manipulation & Poisson PMF computation

## Gotchas & Common Errors

1. **Feature schema mismatch**: If `build_features.py` adds/removes rolling features but models weren't retrained, prediction will fail with `KeyError`. Always retrain after schema changes.
2. **Data leakage via rolling window**: Ensure `.shift(1)` is used in rolling mean—don't include current match in averages.
3. **Player name normalization**: xG lookup is case-sensitive; inconsistent names = 0 xG contribution. Use cleaning utilities.
4. **Season start missing form**: First 4 matches lack rolling stats—median imputation fills these. Check for NaN after imputation.
5. **Empty lineup**: If predicted lineup is empty, `team_xg_avg_adj = 0`. This propagates through model; may produce invalid predictions.
6. **Home/away xG confusion**: `calculate_team_xg()` returns per-team xG; `merge_xg_to_fixtures()` converts to per-fixture (home vs away roles). Don't apply adjustment twice.

## File Organization Principles

- **src/scraping/**: Web scraping only (BeautifulSoup, requests)
- **src/data/**: External data fetching (Understat, raw downloads)
- **src/processing/**: Transform scraped data (xG lookups, merging)
- **src/features/**: Feature engineering & dataset construction
- **src/train/**: Model training (statsmodels, serialization)
- **src/predict/**: Inference & simulation logic
- **src/utils/**: Reusable utilities (cleaning, common transformations)
- **data/raw/**: Immutable source data (CSVs, JSON from APIs)
- **data/processed/**: Derived artifacts (features, JSON intermediates)
- **models/**: Pickle-serialized trained models

When adding new functionality, place it in the appropriate `src/` subdirectory based on its role in the pipeline.
