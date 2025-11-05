# AI Copilot Instructions - EPL-Predictor

This is a **Poisson regression-based football prediction system** that forecasts EPL match outcomes using team attack/defense strengths, home advantage, and Dixon-Coles time-weighting.

## Architecture Overview

The project follows a **simplified data pipeline architecture**:

1. **Data Collection** (`src/data/`) → Fetches historical match data from football-data.co.uk
2. **Model Training** (`src/train/`) → Trains Poisson GLM with Dixon-Coles time-weighting
3. **Fixture Scraping** (`src/weekly/`) → Fetches upcoming fixtures from Premier League API
4. **Prediction** (`src/weekly/`, `src/predict/`) → Generates match outcome probabilities
5. **Main Orchestrator** (`main.py`) → Runs entire pipeline: fetch data → train → predict

### Actual Data Flow

```
football-data.co.uk → epl_historical_results.csv
  ↓
Train Poisson GLM (with time-weighting)
  ↓
Save models: poisson_glm_model.pkl, poisson_coefficients.pkl
  ↓
Premier League API → Fetch upcoming fixtures
  ↓
Load model → Predict expected goals → Calculate probabilities
  ↓
Generate HTML report (auto-opens in browser)
```

**Key Files:**
- **Data fetching**: `src/data/fetch_historical_data.py` (football-data.co.uk)
- **Model training**: `src/train/train_poisson_model.py` (Poisson GLM with time-weighting)
- **Fixture scraping**: `src/weekly/scrape_fixtures.py` (Premier League API)
- **Prediction pipeline**: `src/weekly/predict_weekly.py` (orchestrates predictions)
- **Probability calculation**: `src/predict/generate_probabilities.py` (Poisson PMF)
- **Main entry point**: `main.py` (end-to-end workflow)

## Critical Patterns & Conventions

### 1. Historical Data Format

**File**: `data/raw/epl_historical_results.csv`

**Key columns**:
- `Date`: Match date (YYYY-MM-DD format) - **CRITICAL for time-weighting**
- `HomeTeam`, `AwayTeam`: Team names (football-data.co.uk format)
- `FTHG`, `FTAG`: Full-time home/away goals
- `Season`: Season code (e.g., "2425" for 2024/25)

**Team name format** (football-data.co.uk standard):
- "Man City", "Man United" (NOT "Manchester City/United")
- "Tottenham" (NOT "Tottenham Hotspur")
- "Brighton" (NOT "Brighton & Hove Albion")
- "Nott'm Forest" (NOT "Nottingham Forest")
- "Wolves" (NOT "Wolverhampton Wanderers")

### 2. Poisson Model Architecture

**Formula**: `goals ~ home + team + opponent`

**Model structure**:
- **Intercept** (α₀): Baseline goal rate
- **home**: Home advantage coefficient (γ)
- **team[T.TeamName]**: Attack strength relative to baseline team
- **opponent[T.TeamName]**: Defense strength relative to baseline team

**Baseline team**: First alphabetically (usually Arsenal) has coefficient 0
**All other teams**: Coefficients relative to baseline

**Expected goals formula**:
```
λ_home = exp(α₀ + α_team_home + β_opponent_away + γ)
λ_away = exp(α₀ + α_team_away + β_opponent_home)
```

### 3. Dixon-Coles Time-Weighting (CRITICAL)

**Purpose**: Recent matches weighted more heavily than old matches

**Formula**: `φ(t) = exp(-ξ * t)`
- `t` = time elapsed since match (in half-weeks = 3.5 days)
- `ξ` = decay parameter (default 0.012 for modern EPL data)

**Implementation**:
- Time weights calculated in `calculate_time_weights(dates, xi=0.012)`
- Weights duplicated for home/away rows (2 samples per match)
- Passed to GLM via `freq_weights` parameter

**Key insight**: 
- ξ=0.012 means recent matches (2024-2025) have weight ≈1.0
- Old matches (2005-2010) have weight ≈0.0
- Effective sample size: ~469 (vs 15,284 unweighted)

**Tuning**: Hyperparameter ξ should be optimized via cross-validation for your specific dataset

### 4. Model Files

**Location**: `models/`

**Files**:
- `poisson_glm_model.pkl`: Full statsmodels GLM object (use for predictions)
- `poisson_coefficients.pkl`: Dictionary of extracted coefficients
- `poisson_coefficients.json`: Human-readable coefficients with metadata

**Metadata in coefficients**:
- `intercept`, `home_advantage`: Scalar values
- `team_attack`: Dict mapping team names to attack coefficients
- `opponent_defense`: Dict mapping team names to defense coefficients
- `time_weighted`: Boolean (True if Dixon-Coles weighting used)
- `xi`: Decay parameter value (if time-weighted)
- `trained_at`: ISO timestamp

**When to retrain**:
- New matchweek results added to historical data
- Model file older than latest match in data
- Hyperparameter tuning (changing ξ)

### 5. Fixture API & Team Name Mapping

**API**: `https://footballapi.pulselive.com/football/fixtures`

**CRITICAL**: Premier League API returns different team names than football-data.co.uk

**Name mapping** (in `scrape_fixtures.py` → `_normalize_team_name()`):
```python
'Manchester City' → 'Man City'
'Manchester United' → 'Man United'
'Tottenham Hotspur' → 'Tottenham'
'Brighton & Hove Albion' → 'Brighton'
'Nottingham Forest' → 'Nott\'m Forest'
'Wolverhampton Wanderers' → 'Wolves'
'Leeds United' → 'Leeds'
'Leicester City' → 'Leicester'
'AFC Bournemouth' → 'Bournemouth'
```

**Why critical**: Model trained on football-data.co.uk names. Mismatch = unknown team = prediction fails.

### 6. Prediction Probability Computation

**Method**: Poisson distribution probability mass function (PMF)

**Steps**:
1. Predict expected goals: `λ_home`, `λ_away` from Poisson GLM
2. Generate 7×7 probability matrix (0-6 goals each team)
3. Each cell `P(home=i, away=j) = poisson.pmf(i, λ_home) * poisson.pmf(j, λ_away)`

**Outcome probabilities**:
- **Home win**: Sum of lower triangle (home_goals > away_goals)
- **Draw**: Sum of diagonal (home_goals = away_goals)
- **Away win**: Sum of upper triangle (home_goals < away_goals)

**Implementation**: `src/predict/generate_probabilities.py` → `PoissonPredictor.predict()`

### 7. Output Format

**Location**: `data/weekly/predictions_mw{N}_{timestamp}.html`

**Behavior**: 
- **HTML only** (no JSON/CSV clutter)
- **Auto-opens in browser** via `webbrowser.open()`
- Contains: match predictions, probabilities, most likely scorelines

## Workflow Commands

All commands run from repository root:

```bash
# Install dependencies
pip install -r requirements.txt

# End-to-end prediction (RECOMMENDED - does everything)
python3 main.py
# This will:
# 1. Fetch latest EPL data from football-data.co.uk
# 2. Train Poisson GLM with time-weighting (if needed)
# 3. Scrape upcoming fixtures from Premier League API
# 4. Generate predictions and HTML report
# 5. Auto-open HTML in browser

# Optional: Specify matchweek
python3 main.py --matchweek 15

# Optional: Force model retraining
python3 main.py --retrain

# Train model only (for testing/tuning)
python3 src/train/train_poisson_model.py

# Hyperparameter tuning (find optimal ξ)
python3 src/train/tune_xi.py
```

## Common Tasks & How-Tos

**Update data after new matchweek results:**
```bash
python3 main.py
# Automatically fetches latest data, retrains if needed, generates predictions
```

**Change time-weighting decay parameter:**
1. Edit `src/train/train_poisson_model.py`
2. Change default `xi=0.012` in `train_poisson_model()` function
3. Retrain: `python3 src/train/train_poisson_model.py`
4. Or use hyperparameter tuning to find optimal ξ

**Disable time-weighting (for comparison):**
```python
# In train_poisson_model.py
train_poisson_model(use_time_weighting=False)
```

**Add new team to mapping:**
1. Edit `src/weekly/scrape_fixtures.py`
2. Add to `_normalize_team_name()` dictionary
3. Map Premier League API name → football-data.co.uk name

**Debug prediction failures:**
1. Check model was trained: `ls models/poisson_glm_model.pkl`
2. Verify team names match: Check `scrape_fixtures.py` name mapping
3. Ensure data is up-to-date: Check `data/raw/epl_historical_results.csv` last date
4. Check terminal logs for API errors or missing fixtures

**Analyze model coefficients:**
```bash
# View human-readable coefficients
cat models/poisson_coefficients.json

# Key metrics:
# - home_advantage: Typically 0.15-0.25 (log scale)
# - team_attack: Positive = strong attack, negative = weak attack
# - opponent_defense: Positive = weak defense, negative = strong defense
```

## External Dependencies & APIs

**Data sources**:
- **football-data.co.uk**: Historical match results (2005-present)
  - Free, no API key required
  - CSV format download
  - URL pattern: `https://www.football-data.co.uk/mmz4281/{SEASON}/E0.csv`

- **Premier League API** (footballapi.pulselive.com): Upcoming fixtures
  - Free, no API key required
  - JSON format
  - Returns 100+ upcoming fixtures (filter to specific matchweek)

**Python packages**:
- `statsmodels`: Poisson GLM training (`smf.glm()`)
- `pandas`, `numpy`: Data manipulation
- `scipy.stats.poisson`: Probability calculations
- `requests`, `BeautifulSoup4`: API calls (minimal use)
- `webbrowser`: Auto-open HTML reports

## Gotchas & Common Errors

### 1. Team Name Mismatch
**Symptom**: "Team not found in model" or KeyError during prediction
**Cause**: Premier League API team name not mapped to football-data.co.uk format
**Fix**: Add mapping to `_normalize_team_name()` in `scrape_fixtures.py`

### 2. Outdated Model
**Symptom**: Predictions based on old data, poor accuracy
**Cause**: Model file older than latest match results
**Fix**: Run `python3 main.py` (auto-detects and retrains)

### 3. 100 Fixtures Returned
**Symptom**: API returns 100 fixtures instead of ~10 for next matchweek
**Cause**: Automatic matchweek filtering logic
**Fix**: Already implemented - filters to earliest upcoming matchweek

### 4. Time-Weighting Not Applied
**Symptom**: Df Residuals ≈ 15,284 (should be ~469 if weighted)
**Cause**: `use_time_weighting=False` or weights not passed to GLM
**Fix**: Ensure `use_time_weighting=True` in `train_poisson_model()`

### 5. HTML Not Auto-Opening
**Symptom**: HTML generated but doesn't open in browser
**Cause**: `webbrowser.open()` failed (macOS security, headless environment)
**Fix**: Check console for error message, manually open file from `data/weekly/`

### 6. Large Standard Errors in Model
**Symptom**: Some team coefficients have std err > 1000
**Cause**: Teams with very few recent matches (e.g., relegated teams from 2005)
**Fix**: This is expected - time-weighting reduces their effective sample size to near zero

## File Organization (ACTUAL STRUCTURE)

```
EPL-Predictor/
├── main.py                          # Main entry point (end-to-end pipeline)
├── data/
│   ├── raw/
│   │   └── epl_historical_results.csv   # Historical match data (football-data.co.uk)
│   └── weekly/
│       └── predictions_mw{N}_{timestamp}.html  # Prediction outputs (HTML only)
├── models/
│   ├── poisson_glm_model.pkl        # Trained Poisson GLM
│   ├── poisson_coefficients.pkl     # Extracted coefficients
│   └── poisson_coefficients.json    # Human-readable coefficients
└── src/
    ├── data/
    │   └── fetch_historical_data.py     # Fetch from football-data.co.uk
    ├── train/
    │   ├── train_poisson_model.py       # Train Poisson GLM (with time-weighting)
    │   └── update_model.py              # Check if retraining needed
    ├── weekly/
    │   ├── scrape_fixtures.py           # Fetch fixtures from Premier League API
    │   └── predict_weekly.py            # Generate predictions & HTML report
    ├── predict/
    │   └── generate_probabilities.py    # Poisson probability calculations
    └── utils/
        └── clean_team_names.py          # Team name normalization utilities
```

**Note**: `src/scraping/`, `src/processing/`, `src/features/` directories referenced in old docs **do not exist**. Current architecture is simpler.

## Model Improvement Roadmap

### Current Implementation ✅
- Dixon-Coles time-weighting (ξ=0.012)
- Team attack/defense strengths
- Home advantage coefficient
- Historical data from 2005-present

### In Progress 🔄
- Hyperparameter tuning for optimal ξ
- Backtesting framework for model validation

### Future Enhancements 🔮
- Dixon-Coles dependency correction (adjust for low-scoring draws)
- Rolling form features (recent goals scored/conceded)
- Expected goals (xG) integration from Understat
- Lineup-based predictions (player xG contributions)
- Ensemble models (combine multiple approaches)
