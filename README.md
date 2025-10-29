# EPL Match Predictor# EPL Match Predictor# EPL Match Predictor````markdown



A comprehensive Premier League match prediction system using Poisson regression to forecast match outcomes with competitive accuracy.



## OverviewA comprehensive Premier League match prediction system using Poisson regression to forecast match outcomes with competitive accuracy.# EPL-Predictor



**EPL-Predictor** is a data science project for forecasting English Premier League match outcomes using **Poisson regression** with team strength modeling and home advantage factors. The system achieves **50.28% accuracy** in predicting match outcomes (win/draw/loss), significantly outperforming random baseline (33.33%) by **16.95 percentage points**.



### Key Features## OverviewA comprehensive Premier League match prediction system using Poisson regression to forecast match outcomes with competitive accuracy.



- **Poisson GLM Model**: Statistical regression trained on 4,270 historical matches across 12 seasons (2014-2026)

- **Real-Time Fixtures**: Automatically fetches upcoming matches from Premier League API

- **Weekly Predictions**: Generate predictions for specific matchweeks with one commandThis project implements a complete end-to-end pipeline for predicting English Premier League match results. Built on statistical modeling principles, it achieves 50.28% accuracy in predicting match outcomes (win/draw/loss), significantly outperforming random baseline (33.33%) by 16.95 percentage points.**EPL-Predictor** is a data science and machine learning project for forecasting English Premier League (EPL) match outcomes using **Poisson regression** with team form and opponent strength modeling, following the statistical methodology from [artiebits.com](https://artiebits.com/blog/predicting-football-results-with-statistical-modelling/).

- **Professional Visualization**: Interactive HTML reports with team badges and probability breakdowns

- **Multiple Output Formats**: JSON (API), CSV (analysis), and HTML (web reports)

- **44+ Team Coverage**: Includes current and historical Premier League teams with official badges

### Key Features## Overview

## Model Performance



### Overall Statistics

- **Poisson GLM Model**: Statistical regression trained on 4,270 historical matches across 12 seasons> **Status**: 🟢 Phase 1 Complete - Historical Data Infrastructure Ready  

- **Accuracy**: 50.28% (vs 33.33% random baseline)

- **Training Data**: 4,270 matches, 12 seasons (2014-15 to 2025-26)- **Real-Time Fixtures**: Automatically fetches upcoming matches from Premier League API

- **Test Validation**: 3,890 matches across 11 seasons

- **Home Advantage**: +22.48% goal probability increase- **Current Season Updates**: Continuously trains with latest match results from ongoing seasonThis project implements a complete end-to-end pipeline for predicting English Premier League match results. Built on statistical modeling principles, it achieves 50.28% accuracy in predicting match outcomes (win/draw/loss), significantly outperforming random baseline (33.33%) by 16.95 percentage points.> **Current Progress**: Data fetching & transformation pipeline ✅



### Performance by Outcome Type- **Team Badge Integration**: Visual team badges from Wikipedia CDN in HTML reports



| Outcome | Precision | Recall | F1 Score |- **Multiple Output Formats**: JSON (API), CSV (analysis), and HTML (web reports)

|---------|-----------|--------|----------|

| Home Win | 51.18% | **80.88%** | 62.69% |- **Professional Visualization**: Interactive web reports with probability breakdowns

| Draw | 100.00% | 0.11% | 0.22% |

| Away Win | 48.08% | 44.12% | 46.01% |### Key Features---



**Note**: The model excels at identifying home wins (80.88% recall) but struggles with draw predictions, a common limitation of Poisson models.## Model Performance



## Quick Start



### Prerequisites### Overall Statistics



```bash- **Accuracy**: 50.28% (vs 33.33% random baseline)- **Poisson GLM Model**: Statistical regression trained on 4,270 historical matches across 12 seasons## 🎯 Project Goals

Python 3.11+

pip- **Training Data**: 4,270+ matches, 12 seasons (2014-15 to 2025-26)

Git

```- **Test Validation**: 3,890 matches across 11 seasons- **Time-Series Validation**: Rigorous backtesting on 3,890 test matches using 11 season splits



### Installation- **Home Advantage**: +22.48% goal probability increase



```bash- **Automated Pipeline**: End-to-end workflow from data collection to prediction generation- **Modern Poisson Model**: Single unified GLM using team + opponent + home factors (not separate home/away models)

git clone https://github.com/raynergoh/EPL-Predictor.git

cd EPL-Predictor### Performance by Outcome Type

pip install -r requirements.txt

```| Outcome | Precision | Recall | F1 Score |- **Multiple Output Formats**: JSON (API), CSV (analysis), and HTML (web reports)- **Weekly Predictions**: Predict all 10 fixtures per matchweek with scoreline probabilities



### Generate Weekly Predictions|---------|-----------|--------|----------|



**This is the main use case** - generate predictions for a specific matchweek:| Home Win | 51.18% | **80.88%** | 62.69% |- **Professional Visualization**: Interactive web reports with probability breakdowns- **Low Maintenance**: Train infrequently, reuse model across weeks



```bash| Draw | 100.00% | 0.11% | 0.22% |

# Generate predictions for matchweek 10

python3 -m src.weekly.predict_weekly 10| Away Win | 48.08% | 44.12% | 46.01% |- **Transparent Accuracy**: Backtest against historical data to showcase model quality



# Generate predictions for matchweek 11

python3 -m src.weekly.predict_weekly 11

**Note**: The model excels at identifying home wins (80.88% recall) but struggles with draw predictions, a common limitation of Poisson models.## Model Performance- **Lineup-Based xG**: Incorporate predicted lineups with player xG stats

# Generate predictions for all upcoming fixtures (no matchweek specified)

python3 -m src.weekly.predict_weekly

```

## Quick Start- **Opponent Strength**: Factor in opponent defense (Chelsea vs West Ham ≠ Chelsea vs Arsenal)

**What happens:**

1. Fetches fixtures from Premier League API

2. Filters to specified matchweek (or all upcoming fixtures)

3. Loads trained Poisson GLM model### Prerequisites### Overall Statistics

4. Generates predictions with scoreline probabilities

5. Saves results in 3 formats (JSON, CSV, HTML)



**Output files** (saved in `data/weekly/`):```bash- **Accuracy**: 50.28% (vs 33.33% random baseline)---

- `predictions_mw10_*.html` - **Open this in your browser!** 🌐

- `predictions_mw10_*.json` - For API integrationPython 3.11+

- `predictions_mw10_*.csv` - For spreadsheet analysis

Git- **Training Data**: 4,270 matches, 12 seasons (2014-15 to 2025-26)

### View Predictions in Browser

```

After generating predictions, open the HTML report:

- **Test Validation**: 3,890 matches across 11 seasons## 📊 Current Data

```bash

# On macOS:### Installation

open data/weekly/predictions_mw*.html

- **Home Advantage**: +22.48% goal probability increase

# On Linux:

xdg-open data/weekly/predictions_mw*.html```bash



# On Windows:git clone https://github.com/raynergoh/EPL-Predictor.git| Metric | Value |

start data\weekly\predictions_mw*.html

```cd EPL-Predictor



**What you'll see:**pip install -r requirements.txt### Performance by Outcome Type|--------|-------|

- 🏆 Premier League logo and professional branding

- 🎨 Color-coded probability bars for Win/Draw/Loss```

- ⚽ Expected goals (xG) for each team

- 📊 Most likely scoreline + top 3 alternatives| Outcome | Precision | Recall | F1 Score || **Historical Matches** | 4,270 |

- 🛡️ Official team badges for all clubs

- 📈 Model performance statistics### Generate Weekly Predictions (Main Use Case)



## Usage Examples|---------|-----------|--------|----------|| **Seasons** | 2014-15 to 2025-26 |



### Python APIThis is what most users will want to do - generate predictions for upcoming fixtures and view them in a browser.



```python| Home Win | 51.18% | **80.88%** | 62.69% || **Teams** | 35 unique (includes promoted teams) |

from src.weekly.predict_weekly import WeeklyPredictor

```bash

# Initialize predictor

predictor = WeeklyPredictor()# Step 1: Generate predictions for upcoming fixtures| Draw | 100.00% | 0.11% | 0.22% || **Latest Data** | 2025-10-26 |



# Generate predictions for matchweek 10python -m src.weekly.predict_weekly

results = predictor.run_weekly_predictions(matchweek=10)

| Away Win | 48.08% | 44.12% | 46.01% || **Data Quality** | 100% (no missing values) |

# Access prediction data

for pred in results['predictions']:# Step 2: Open the HTML report in your browser

    print(f"{pred['home_team']} vs {pred['away_team']}")

    print(f"  Expected Goals: {pred['expected_goals']['home']:.2f} - {pred['expected_goals']['away']:.2f}")# The filename will be shown in the output (e.g., predictions_mw10_20251030_013157.html)

    print(f"  Home Win: {pred['probabilities']['home_win']:.1%}")

    print(f"  Draw: {pred['probabilities']['draw']:.1%}")

    print(f"  Away Win: {pred['probabilities']['away_win']:.1%}")

    print(f"  Most Likely: {pred['most_likely_scoreline']}")# On macOS:**Note**: The model excels at identifying home wins (80.88% recall) but struggles with draw predictions, a common limitation of Poisson models.**Goal Statistics** (validates model assumptions):

    print()

```open data/weekly/predictions_mw*.html



### Sample Output- Home win rate: **44.7%** ✅



```# On Linux:

Matchweek 10 Predictions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━xdg-open data/weekly/predictions_mw*.html## Quick Start- Draw rate: **23.4%** ✅



Arsenal vs Liverpool

Expected: 1.88 - 1.15

Scoreline: 1-1 (10.4%)# On Windows:- Away win rate: **31.9%** ✅

Probabilities: Home 54% | Draw 23% | Away 23%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━start data\weekly\predictions_mw*.html

```

```### Prerequisites

## Project Structure



```

EPL-Predictor/**What you'll see in the browser:**---

├── src/

│   ├── data/- Team badges for all clubs

│   │   └── extract_historical_data.py    # Fetch historical data

│   ├── features/- Expected goals (xG) for each team  ```bash

│   │   └── build_features.py             # Feature engineering

│   ├── train/- Win/Draw/Loss probabilities with color-coded gradient bars

│   │   └── poisson_model.py              # Model training

│   ├── predict/- Most likely scoreline (e.g., "Liverpool 2-1 Brighton")Python 3.11+## 🚀 Quick Start

│   │   └── generate_probabilities.py     # Prediction engine

│   ├── backtest/- Top 3 alternative scorelines with probabilities

│   │   ├── backtest_poisson.py           # Validation framework

│   │   └── analyze_backtest_results.py   # Performance analysis- Model performance statisticspip install -r requirements.txt

│   ├── weekly/

│   │   ├── scrape_fixtures.py            # Fetch fixtures (Premier League API)- Professional gradient design optimized for readability

│   │   └── predict_weekly.py             # Weekly prediction pipeline

│   └── utils/```### 1. Install Dependencies

│       └── cleaning.py                   # Data utilities

├── models/**Output files created** (saved in `data/weekly/`):

│   ├── poisson_glm_model.pkl             # Trained model (10 MB)

│   └── poisson_coefficients.json         # Model coefficients- `predictions_mw{N}_{timestamp}.html` - **Open this in your browser**

├── data/

│   ├── raw/- `predictions_mw{N}_{timestamp}.json` - For API integration

│   │   └── epl_historical_results.csv    # Historical matches (4,270)

│   └── weekly/- `predictions_mw{N}_{timestamp}.csv` - For spreadsheet analysis### Installation```sh

│       └── predictions_mw*.html          # Generated reports

└── requirements.txt

```

### Update Model with Current Season Datapip install -r requirements.txt

## Methodology



### Model Architecture

The model continuously learns from new matches. Run this periodically (e.g., weekly) to incorporate the latest results:```bash```

The system uses a **Poisson Generalized Linear Model (GLM)**:



```

Formula: goals ~ home + C(team) + C(opponent)```bashgit clone https://github.com/raynergoh/EPL-Predictor.git

Family: Poisson

Link: Log# Fetch latest match results and retrain model

```

python -m src.train.update_modelcd EPL-Predictor### 2. Fetch Historical Data

**Parameters** (70 total):

- Intercept: Baseline goal rate```

- Home advantage: Binary indicator (+22.48% effect)

- Team attack effects: 43 team offensive strengthspip install -r requirements.txt

- Opponent defense effects: 43 team defensive strengths

- Arsenal: Baseline team (implicit coefficient = 0)This will:



### Prediction Algorithm1. Fetch completed matches from current season (via Premier League API)``````sh



1. **Expected Goals Calculation**2. Append new results to historical data

   ```python

   λ_home = exp(intercept + home_advantage + team_attack + opponent_defense)3. Retrain the Poisson model with updated datapython -m src.data.fetch_historical_data

   λ_away = exp(intercept + team_attack + opponent_defense)

   ```4. Save improved model for next predictions



2. **Scoreline Probability Matrix**### Generate Weekly Predictions```

   - Generate Poisson PMF for 0-6 goals per team

   - Create 7×7 matrix: P(i,j) = Poisson(home=i) × Poisson(away=j)**When to run this:**



3. **Match Outcome Probabilities**- After each matchweek completes (weekly)

   - Home Win: Sum of lower triangle (home > away)

   - Draw: Sum of diagonal (home = away)- Before generating predictions for important matches

   - Away Win: Sum of upper triangle (away > home)

- When you notice prediction accuracy declining```bashThis downloads EPL match data from football-data.co.uk (2014-15 to present).

### Data Pipeline



**Phase 1: Data Collection**

- Source: football-data.co.uk historical data (2014-2026)### Full Training Pipeline (Advanced)python -m src.weekly.predict_weekly

- Coverage: 4,270 matches across 12 EPL seasons



**Phase 2: Feature Engineering**

- Transform to 2-rows-per-match format (home/away perspectives)Only needed if you want to retrain from scratch:```### 3. Next Steps (Coming Soon)

- Add home/away indicators

- Create time-series train/test splits



**Phase 3: Model Training**```bash

- Fit Poisson GLM using statsmodels

- Extract team strength coefficients# Step 1: Fetch historical data (if needed)

- Serialize model for production use

python -m src.data.extract_historical_dataOutput files will be saved to `data/weekly/`:- [ ] Transform data to Poisson GLM format

**Phase 4: Validation**

- Time-series cross-validation (11 season splits)

- Calculate accuracy, precision, recall metrics

# Step 2: Train Poisson GLM model- `predictions_mw{N}_*.json` - Detailed prediction data- [ ] Train single Poisson regression model

**Phase 5: Production Deployment**

- Weekly fixture scraping from Premier League APIpython -m src.train.poisson_model

- Automated prediction generation

- Multi-format output (JSON/CSV/HTML)- `predictions_mw{N}_*.csv` - Spreadsheet-friendly format- [ ] Implement Poisson outer product for scoreline probabilities



## Advanced Usage# Step 3: Validate model accuracy (optional)



### Retrain Model (If Needed)python -m src.backtest.backtest_poisson- `predictions_mw{N}_*.html` - Interactive web report- [ ] Backtest model accuracy



Only required if you want to retrain from scratch:



```bash# Step 4: Analyze backtest results (optional)- [ ] Scrape upcoming fixtures from PL website

# Step 1: Fetch historical data

python -m src.data.extract_historical_datapython -m src.backtest.analyze_backtest_results



# Step 2: Build features```## Project Structure- [ ] Generate predictions for next matchweek

python -m src.features.build_features



# Step 3: Train model

python -m src.train.poisson_model## Project Structure



# Step 4: Validate (optional)

python -m src.backtest.backtest_poisson

``````---

# Step 5: Analyze results (optional)

python -m src.backtest.analyze_backtest_resultsEPL-Predictor/

```

├── src/EPL-Predictor/

### Access Model Coefficients

│   ├── data/

```python

import json│   │   └── extract_historical_data.py    # Fetch historical match data (Understat API)├── src/## 📁 Project Structure



# Load model coefficients│   ├── features/

with open('models/poisson_coefficients.json', 'r') as f:

    coefficients = json.load(f)│   │   └── build_features.py             # Feature engineering pipeline│   ├── data/



# View team attack strengths│   ├── train/

print("Team Attack Coefficients:")

for team, coef in sorted(coefficients['team_attack'].items(), │   │   ├── poisson_model.py              # Model training (full retrain)│   │   └── extract_historical_data.py    # Fetch historical match data```

                         key=lambda x: x[1], reverse=True)[:5]:

    print(f"  {team}: {coef:.3f}")│   │   └── update_model.py               # Continuous training with current season



# View home advantage│   ├── predict/│   ├── features/EPL-Predictor/

print(f"\nHome Advantage: {coefficients['home_advantage']:.3f}")

```│   │   └── generate_probabilities.py     # Prediction engine with Poisson PMF



## Validation Results│   ├── backtest/│   │   └── build_features.py             # Feature engineering pipeline├── data/



### Season-by-Season Performance│   │   ├── backtest_poisson.py           # Time-series validation framework



| Season | Accuracy | Matches |│   │   └── analyze_backtest_results.py   # Performance analysis & visualization│   ├── train/│   ├── raw/

|--------|----------|---------|

| 2018-19 | 56.58% | 380 |│   ├── weekly/

| 2016-17 | 55.53% | 380 |

| 2023-24 | 53.95% | 380 |│   │   ├── scrape_fixtures.py            # Fetch upcoming fixtures (PL API)│   │   └── poisson_model.py              # Model training│   │   └── epl_historical_data.csv    ← NEW: 4,270 historical matches

| 2021-22 | 52.89% | 380 |

| 2017-18 | 51.05% | 380 |│   │   └── predict_weekly.py             # Weekly prediction pipeline + HTML report

| 2019-20 | 50.79% | 380 |

| 2022-23 | 50.00% | 380 |│   └── utils/│   ├── predict/│   └── processed/

| 2020-21 | 48.42% | 380 |

| 2015-16 | 42.37% | 380 |│       └── cleaning.py                   # Data cleaning utilities

| 2024-25 | 41.32% | 380 |

├── models/│   │   └── generate_probabilities.py     # Prediction engine├── src/

**Average**: 50.28% across 3,890 test matches

│   ├── poisson_glm_model.pkl             # Trained model (10 MB)

### Benchmarking

│   └── poisson_coefficients.json         # Model coefficients (human-readable)│   ├── backtest/│   ├── data/

- **Random Baseline**: 33.33% (guessing outcome uniformly)

- **Our Model**: 50.28% (+16.95 percentage points)├── data/

- **Industry Standard**: Bookmakers achieve 50-53%

- **Academic Studies**: Typical Poisson models achieve 45-52%│   ├── raw/│   │   ├── backtest_poisson.py           # Validation framework│   │   └── fetch_historical_data.py   ← NEW: Football-data.co.uk fetcher



**Conclusion**: Our model performs competitively with professional systems and achieves the high end of academic benchmarks.│   │   └── epl_historical_results.csv    # Historical match results



## Team Badge Coverage│   ├── processed/│   │   └── analyze_backtest_results.py   # Performance analysis│   ├── features/                       ← Next: Data transformation



The HTML reports include official badges for **44+ teams**:│   │   ├── backtest_results/             # Validation metrics by season



**Current Teams (2025/26):**│   │   └── season_splits/                # Time-series train/test splits│   ├── weekly/│   ├── train/                          ← Next: Poisson GLM training

Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, Burnley, Chelsea, Crystal Palace, Everton, Fulham, Ipswich, Leeds, Leicester, Liverpool, Luton, Man City, Man United, Newcastle, Nott'm Forest, Sheffield Utd, Southampton, Sunderland, Tottenham, West Ham, Wolves

│   └── weekly/

**Historical Teams:**

Barnsley, Birmingham, Blackburn, Blackpool, Bolton, Cardiff, Charlton, Derby, Huddersfield, Hull, Middlesbrough, Norwich, Portsmouth, QPR, Reading, Stoke, Swansea, Swindon, Watford, West Brom, Wigan│       └── predictions_mw*.html          # Generated prediction reports│   │   ├── scrape_fixtures.py            # Fixture collection│   ├── predict/                        ← Next: Prediction pipeline



**Badge Source**: Wikipedia SVG (high-quality, always up-to-date)└── logs/                                 # Execution logs



## Technologies│   │   └── predict_weekly.py             # Weekly prediction pipeline│   └── evaluate/                       ← Next: Backtesting framework



- **Python 3.11+**```

- **pandas**: Data manipulation

- **statsmodels**: Poisson GLM│   └── utils/├── models/                             ← Serialized trained models

- **scipy**: Poisson distribution & probability calculations

- **requests + BeautifulSoup**: Web scraping for fixtures## Usage Guide

- **matplotlib + seaborn**: Visualization (backtesting)

│       └── cleaning.py                   # Data cleaning utilities├── PHASE_1_SUMMARY.md                 ← Detailed Phase 1 report

## Model Limitations

### Basic Workflow

1. **Draw Prediction Challenge**

   - Poisson models inherently struggle with draws├── models/└── README.md                           ← This file

   - Only 1 draw predicted out of 905 actual draws

   - This is a known limitation in football prediction literature```python



2. **Home Field Bias**# Import the weekly predictor│   ├── poisson_glm_model.pkl             # Trained model (10 MB)```

   - Model over-predicts home wins (2,743 predicted vs 1,736 actual)

   - Trade-off: Achieves excellent home win recall (80.88%)from src.weekly.predict_weekly import WeeklyPredictor



3. **Baseline Team (Arsenal)**│   └── poisson_coefficients.json         # Model coefficients

   - Arsenal is the reference category with implicit coefficient 0

   - All other teams measured relative to Arsenal's strength# Initialize predictor

   - Statistically correct for Poisson GLM modeling

predictor = WeeklyPredictor()├── data/---

## Best Practices



### Recommended Use Cases

- ✅ **Strong Home Favorites**: High confidence predictions (80.9% recall)# Generate predictions for upcoming fixtures│   ├── raw/                              # Source data

- ✅ **Expected Goals Analysis**: Reliable xG estimates

- ✅ **Comparative Analysis**: Identify relative team strengthsresults = predictor.run_weekly_predictions()

- ✅ **Long-term Tracking**: Monitor prediction accuracy over seasons

│   ├── processed/                        # Processed datasets## 🔬 Model Architecture (Planned)

### Use with Caution

- ⚠️ **Draw Predictions**: Model rarely predicts draws (0.11% recall)# Access prediction data

- ⚠️ **Newly Promoted Teams**: Limited historical data reduces accuracy

- ⚠️ **Early Season**: Less context from current season performancefor pred in results['predictions']:│   └── weekly/                           # Weekly predictions



### Maintenance Schedule    print(f"{pred['home_team']} vs {pred['away_team']}")

- **Weekly**: Generate predictions for upcoming matchweek

- **Monthly**: Review prediction accuracy against actual results    print(f"  Home Win: {pred['probabilities']['home_win']:.1%}")└── logs/                                 # Execution logs### Training Data Format

- **Every 5-10 matchweeks**: Optional model retraining with new data

- **Season End**: Full retraining with complete season data    print(f"  Draw: {pred['probabilities']['draw']:.1%}")



## Future Enhancements    print(f"  Away Win: {pred['probabilities']['away_win']:.1%}")```Convert historical matches to 2 rows per match (home & away perspectives):



Potential improvements for increased accuracy:    print(f"  Most Likely: {pred['most_likely_scoreline']}")



1. **Advanced Features**```

   - Opponent-adjusted expected goals from current season

   - Recent form indicators (last 5 matches, weighted by recency)

   - Head-to-head historical records

   - Injury/suspension data integration### Predict Specific Matchweek## Usage Guide```python



2. **Model Improvements**

   - Bivariate Poisson model (accounts for score correlation)

   - Dixon-Coles time decay weighting (recent matches weighted higher)```bash# Before (1 row per match):

   - Machine learning ensemble methods (XGBoost, Random Forest)

   - Bayesian updating for in-season adaptation# Generate predictions for a specific matchweek



3. **Data Enrichment**python -m src.weekly.predict_weekly --matchweek 15### Training the ModelDate        HomeTeam      AwayTeam      FTHG  FTAG

   - Player-level statistics and lineup quality

   - Weather conditions and pitch quality```

   - Travel distance and fixture congestion

   - Betting market odds for calibration2025-10-26  Wolves        Burnley       2     3



4. **Production Features**### Access Model Coefficients

   - Automated weekly retraining workflow

   - Prediction tracking dashboard with accuracy over time```bash

   - Confidence intervals for prediction uncertainty

   - Betting strategy simulation and Kelly criterion```python



## Contributingimport json# 1. Fetch historical data# After (2 rows per match):



This is a personal research project. For questions or suggestions, please open an issue on GitHub.



## License# Load model coefficientspython -m src.data.extract_historical_dataDate        team     opponent  goals  home



MIT License - See LICENSE file for detailswith open('models/poisson_coefficients.json', 'r') as f:



## Acknowledgments    coefficients = json.load(f)2025-10-26  Wolves   Burnley   2      1



- Data source: [football-data.co.uk](https://www.football-data.co.uk/), Premier League API

- Team badges: [Wikipedia Commons](https://commons.wikimedia.org/)

- Statistical methods based on academic football prediction literature# View team attack strengths# 2. Build feature matrix2025-10-26  Burnley  Wolves    3      0

- Inspired by Dixon-Coles (1997) modeling framework

print("Team Attack Coefficients:")

## References

for team, coef in sorted(coefficients['team_attack'].items(), key=lambda x: x[1], reverse=True):python -m src.features.build_features```

1. Dixon, M. J., & Coles, S. G. (1997). Modelling Association Football Scores and Inefficiencies in the Football Betting Market. *Journal of the Royal Statistical Society*.

    print(f"  {team}: {coef:.3f}")

2. Maher, M. J. (1982). Modelling Association Football Scores. *Statistica Neerlandica*.



3. Karlis, D., & Ntzoufras, I. (2003). Analysis of Sports Data by Using Bivariate Poisson Models. *Journal of the Royal Statistical Society*.

# View home advantage

---

print(f"\nHome Advantage: {coefficients['home_advantage']:.3f}")# 3. Train Poisson GLM### Poisson GLM Formula

**Project Status**: Production Ready ✅

```

**Last Updated**: October 2025

python -m src.train.poisson_model```

**Contact**: [GitHub Issues](https://github.com/raynergoh/EPL-Predictor/issues)

## Methodology

```log(λ_i) = α₀ + α_team_i + β_opponent_i + γ × home_i

### Model Architecture

```

The system uses a **Poisson Generalized Linear Model (GLM)** with the following specification:

### Running Backtests

```

Formula: goals ~ home + C(team) + C(opponent)Where:

Family: Poisson

Link: Log```bash- **λ_i**: Expected goals for team i

```

# Full backtesting across all seasons- **α₀**: Baseline goal rate

**Parameters** (70+ total):

- Intercept: Baseline goals expectationpython -m src.backtest.backtest_poisson- **α_team**: Attack strength of team i

- Home advantage: Binary indicator (+22.48% effect)

- Team attack effects: 35+ team offensive strengths- **β_opponent**: Defense strength of opponent

- Opponent defense effects: 35+ team defensive strengths

# Generate analysis visualizations- **γ**: Home advantage coefficient (~0.27)

### Prediction Algorithm

python -m src.backtest.analyze_backtest_results- **home**: Binary indicator (1 if team i is home, 0 if away)

1. **Expected Goals Calculation**

   - Calculate lambda (λ) for both teams using model coefficients```

   - Home team: λ_home = exp(intercept + home + team_i + opponent_j)

   - Away team: λ_away = exp(intercept + team_j + opponent_i)### Prediction Pipeline



2. **Scoreline Probability Matrix**### Making Predictions1. **Expected Goals**: Model predicts λ_home and λ_away using Poisson GLM

   - Generate Poisson PMF for 0-6 goals for each team

   - Create 7×7 matrix via outer product: P(i,j) = P(home=i) × P(away=j)2. **Scoreline Matrix**: Outer product of Poisson PMFs (0-6 goals per team)



3. **Match Outcome Probabilities**```python3. **Probabilities**: Sum matrix cells for Home Win / Draw / Away Win

   - Home Win: Sum of lower triangle (home > away)

   - Draw: Sum of diagonal (home = away)from src.weekly.predict_weekly import WeeklyPredictor

   - Away Win: Sum of upper triangle (away > home)

---

### Continuous Learning

# Initialize predictor

The model uses a **continuous training approach** to stay current:

predictor = WeeklyPredictor()## 📈 Current Outputs

1. **Historical Foundation**: Trained on 12 seasons of EPL data (2014-2026)

2. **Current Season Updates**: Incorporates latest match results weekly

3. **Adaptive Coefficients**: Team strengths adjust based on recent form

4. **Promoted Team Handling**: New teams use league-average coefficients until data accumulates# Run weekly predictions**Sample Prediction** (planned):



This ensures predictions for newly promoted teams (e.g., Leeds, Burnley, Sunderland in 2025/26) improve as the season progresses, rather than relying solely on old Championship data.results = predictor.run_weekly_predictions()```



### Data PipelineMatchweek 10



**Phase 1: Data Collection**# Access prediction data━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Historical: Understat.com API (12 seasons, 4,270+ matches)

- Current Season: Premier League API (completed matches, updated weekly)for pred in results['predictions']:Arsenal vs Liverpool

- Format: Match results with team identifiers and goals scored

    print(f"{pred['home_team']} vs {pred['away_team']}")Expected: 1.63 - 1.45

**Phase 2: Feature Engineering**

- Transform to 2-rows-per-match format (home/away perspectives)    print(f"  Home Win: {pred['probabilities']['home_win']:.1%}")Scoreline: 1-1 (12.3%)

- Add home/away indicators

- Create time-series train/test splits    print(f"  Draw: {pred['probabilities']['draw']:.1%}")Probabilities: Home 37% | Draw 28% | Away 35%



**Phase 3: Model Training**    print(f"  Away Win: {pred['probabilities']['away_win']:.1%}")━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Fit Poisson GLM using statsmodels

- Extract team attack/defense coefficients``````

- Serialize model for production use



**Phase 4: Validation**

- Time-series cross-validation (11 season splits)## MethodologyFull outputs include:

- Calculate accuracy, precision, recall metrics

- Generate calibration plots- Predicted scorelines (most likely + top 3)



**Phase 5: Production Deployment**### Model Architecture- Win/Draw/Away probabilities

- Weekly fixture scraping from Premier League API

- Automated prediction generation- Model confidence scores

- Multi-format output (JSON/CSV/HTML with team badges)

The system uses a **Poisson Generalized Linear Model (GLM)** with the following specification:- JSON/CSV exports for analysis

## Output Formats



### HTML Report (Primary)

Interactive web report featuring:```---

- **Team Badges**: High-quality SVG logos from Wikipedia CDN

- **Visual Design**: Modern gradient styling with purple/pink/blue colorsFormula: goals ~ home + C(team) + C(opponent)

- **Probability Bars**: Color-coded win/draw/loss percentages

- **Match Cards**: Hover effects and responsive layoutFamily: Poisson## 🔄 Development Phases

- **Model Stats**: Performance metrics and accuracy tracking

- **Mobile-Friendly**: Responsive design works on all devicesLink: Log



### JSON Output```| Phase | Status | Description |

Detailed machine-readable format containing:

- Expected goals for both teams|-------|--------|-------------|

- Win/draw/loss probabilities

- Most likely scoreline**Parameters** (70 total):| **Phase 1** | ✅ Done | Historical data fetching infrastructure |

- Top 3 alternative scorelines

- Full 7×7 probability matrix- Intercept: Baseline goals expectation| **Phase 1b** | ⏳ Next | Data transformation to Poisson format |

- Fixture metadata (date, venue, teams)

- Home advantage: Binary indicator (+22.48% effect)| **Phase 2** | ⏳ Planned | Poisson GLM training & validation |

### CSV Output

Spreadsheet-friendly format with:- Team effects: 34 team attack strengths| **Phase 3** | ⏳ Planned | Prediction pipeline with lineup xG |

- Match date and time

- Team names- Opponent effects: 34 opponent defense strengths| **Phase 4** | ⏳ Planned | Backtesting framework & accuracy metrics |

- Expected goals (xG)

- Outcome probabilities| **Phase 5** | ⏳ Planned | Weekly prediction output & deployment |

- Predicted winner

### Prediction Algorithm

## Best Practices

See [PHASE_1_SUMMARY.md](PHASE_1_SUMMARY.md) for detailed completion report.

### Recommended Use Cases

- **Strong Home Favorites**: High confidence in home win predictions (80.9% recall)1. **Expected Goals Calculation**

- **Expected Goals Analysis**: Reliable xG estimates for team performance

- **Comparative Analysis**: Identify relative team strengths   - Calculate lambda (λ) for both teams using model coefficients---

- **Long-term Tracking**: Monitor prediction accuracy over seasons

   - Home team: λ_home = exp(intercept + home + team_i + opponent_j)

### Use with Caution

- **Draw Predictions**: Model rarely predicts draws (0.11% recall)   - Away team: λ_away = exp(intercept + team_j + opponent_i)## 🛠 Technologies

- **Newly Promoted Teams**: Limited historical data in first few weeks

- **Early Season**: Less context from current season performance (improves weekly)



### Maintenance Schedule2. **Scoreline Probability Matrix**- **Python 3.11+**

- **Weekly**: Update model with latest results (`python -m src.train.update_model`)

- **After Each Matchweek**: Generate new predictions (`python -m src.weekly.predict_weekly`)   - Generate Poisson PMF for 0-6 goals for each team- **pandas**: Data manipulation

- **Monthly**: Review prediction accuracy using backtest results

- **Season End**: Full retraining with complete season data   - Create 7x7 matrix via outer product: P(i,j) = P(home=i) × P(away=j)- **statsmodels**: Poisson GLM



## Validation Results- **scipy**: Poisson distribution & probability calculations



### Season-by-Season Performance3. **Match Outcome Probabilities**- **requests + BeautifulSoup**: Web scraping



| Season | Accuracy | Matches |   - Home Win: Sum of lower triangle (home > away)

|--------|----------|---------|

| 2018-19 | 56.58% | 380 |   - Draw: Sum of diagonal (home = away)---

| 2016-17 | 55.53% | 380 |

| 2023-24 | 53.95% | 380 |   - Away Win: Sum of upper triangle (away > home)

| 2021-22 | 52.89% | 380 |

| 2017-18 | 51.05% | 380 |## 📚 References

| 2019-20 | 50.79% | 380 |

| 2022-23 | 50.00% | 380 |### Data Pipeline

| 2020-21 | 48.42% | 380 |

| 2015-16 | 42.37% | 380 |- [Predicting Football Results with Poisson Regression](https://artiebits.com/blog/predicting-football-results-with-statistical-modelling/) - Core methodology

| 2024-25 | 41.32% | 380 |

**Phase 1: Data Collection**- [football-data.co.uk](https://www.football-data.co.uk/) - Historical match data

**Average**: 50.28% across 3,890 test matches

- Source: Understat.com historical data- [Understat](https://understat.com/) - Player xG statistics

### Benchmarking

- Coverage: 12 EPL seasons (2014-15 to 2025-26)- [Fantasy Football Scout](https://www.fantasyfootballscout.co.uk/) - Predicted lineups

- **Random Baseline**: 33.33% (guessing outcome uniformly)

- **Our Model**: 50.28% (+16.95 percentage points)- Format: Match results with team identifiers

- **Industry Standard**: Bookmakers achieve 50-53%

- **Academic Studies**: Typical Poisson models achieve 45-52%---



**Conclusion**: Our model performs competitively with professional systems and achieves the high end of academic benchmarks.**Phase 2: Feature Engineering**



## Technical Details- Transform to 2-rows-per-match format## 📝 License



### Dependencies- Add home/away indicators



Core libraries:- Create time-series train/test splitsOpen source for educational purposes.

- `statsmodels` - GLM implementation

- `scipy` - Poisson probability calculations

- `pandas` - Data manipulation

- `numpy` - Numerical operations**Phase 3: Model Training**````

- `requests` - API calls for fixtures/results

- `beautifulsoup4` - Web scraping fallback- Fit Poisson GLM using statsmodels

- `matplotlib` - Visualization- Extract team strength coefficients

- `seaborn` - Statistical plots- Serialize model for production use



### Model Limitations**Phase 4: Validation**

- Time-series cross-validation (11 splits)

1. **Draw Prediction Challenge**- Calculate accuracy, precision, recall metrics

   - Poisson models inherently struggle with draws- Generate calibration plots

   - Only 1 draw predicted out of 905 actual draws

   - This is a known limitation in football prediction literature**Phase 5: Production Deployment**

- Weekly fixture scraping

2. **Home Field Bias**- Automated prediction generation

   - Model over-predicts home wins (2,743 predicted vs 1,736 actual)- Multi-format output (JSON/CSV/HTML)

   - Trade-off: Achieves excellent home win recall (80.88%)

## Output Formats

3. **New Team Handling**

   - Teams not in training data start with league-average coefficients### JSON Output

   - Accuracy improves as season progresses and data accumulatesDetailed machine-readable format containing:

   - By matchweek 10+, promoted teams have sufficient data for accurate predictions- Expected goals for both teams

- Win/draw/loss probabilities

### Team Badge Coverage- Most likely scoreline

- Top 3 alternative scorelines

Badges included for 35+ teams:- Full 7x7 probability matrix

- All current Premier League teams (2025/26)- Fixture metadata

- Historical teams: Cardiff, Huddersfield, Hull, Middlesbrough, Norwich, QPR, Sheffield United, Stoke, Swansea, Watford, West Brom

- Promoted teams: Burnley, Leeds United, Luton, Sunderland### CSV Output

- CDN source: Wikipedia (always up-to-date, high-quality SVG)Spreadsheet-friendly format with:

- Match date and time

## Future Enhancements- Team names

- Expected goals (xG)

Potential improvements for increased accuracy:- Outcome probabilities

- Predicted winner

1. **Advanced Features**

   - Opponent-adjusted expected goals from current season### HTML Report

   - Recent form indicators (last 5 matches, weighted by recency)Interactive web report featuring:

   - Head-to-head historical records- Modern gradient design

   - Injury/suspension data integration- Color-coded probability bars

   - Home advantage decay over time- Match cards with hover effects

- Model performance statistics

2. **Model Improvements**- Mobile-responsive layout

   - Bivariate Poisson model (accounts for score correlation)

   - Dixon-Coles time decay weighting (recent matches weighted higher)## Best Practices

   - Machine learning ensemble methods (XGBoost, Random Forest)

   - Bayesian updating for in-season adaptation### Recommended Use Cases

- **Strong Home Favorites**: High confidence in home win predictions (80.9% recall)

3. **Data Enrichment**- **Expected Goals Analysis**: Reliable xG estimates for team performance

   - Player-level statistics and lineup quality- **Comparative Analysis**: Identify relative team strengths

   - Weather conditions and pitch quality- **Long-term Tracking**: Monitor prediction accuracy over seasons

   - Travel distance and fixture congestion

   - Betting market odds for calibration### Use with Caution

- **Draw Predictions**: Model rarely predicts draws (0.11% recall)

4. **Production Features**- **Newly Promoted Teams**: Limited historical data reduces accuracy

   - Automated weekly retraining workflow- **Early Season**: Less context from current season performance

   - Prediction tracking dashboard with accuracy over time

   - Confidence intervals for prediction uncertainty### Maintenance Schedule

   - Betting strategy simulation and Kelly criterion- **Weekly**: Generate predictions for upcoming fixtures

- **Monthly**: Review prediction accuracy

## Contributing- **Every 5-10 matchweeks**: Retrain model with new data

- **Season End**: Full retraining with complete season data

This is a personal research project. For questions or suggestions, please open an issue on GitHub.

## Validation Results

## License

### Season-by-Season Performance

MIT License - See LICENSE file for details

| Season | Accuracy | Matches |

## Acknowledgments|--------|----------|---------|

| 2018-19 | 56.58% | 380 |

- Data source: Understat.com, Premier League API| 2016-17 | 55.53% | 380 |

- Statistical methods based on academic football prediction literature| 2023-24 | 53.95% | 380 |

- Inspired by Dixon-Coles (1997) modeling framework| 2021-22 | 52.89% | 380 |

- Team badges from Wikipedia Commons| 2017-18 | 51.05% | 380 |

| 2019-20 | 50.79% | 380 |

## References| 2022-23 | 50.00% | 380 |

| 2020-21 | 48.42% | 380 |

1. Dixon, M. J., & Coles, S. G. (1997). Modelling Association Football Scores and Inefficiencies in the Football Betting Market. *Journal of the Royal Statistical Society*.| 2015-16 | 42.37% | 380 |

| 2024-25 | 41.32% | 380 |

2. Maher, M. J. (1982). Modelling Association Football Scores. *Statistica Neerlandica*.

**Average**: 50.28% across 3,890 test matches

3. Karlis, D., & Ntzoufras, I. (2003). Analysis of Sports Data by Using Bivariate Poisson Models. *Journal of the Royal Statistical Society*.

### Benchmarking

---

- **Random Baseline**: 33.33% (guessing outcome uniformly)

**Project Status**: Production Ready- **Our Model**: 50.28% (+16.95 percentage points)

- **Industry Standard**: Bookmakers achieve 50-53%

**Last Updated**: October 2025- **Academic Studies**: Typical Poisson models achieve 45-52%



**Contact**: [GitHub Issues](https://github.com/raynergoh/EPL-Predictor/issues)**Conclusion**: Our model performs competitively with professional systems.


## Technical Details

### Dependencies

Core libraries:
- `statsmodels` - GLM implementation
- `scipy` - Poisson probability calculations
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Visualization
- `seaborn` - Statistical plots

### Model Limitations

1. **Draw Prediction Challenge**
   - Poisson models inherently struggle with draws
   - Only 1 draw predicted out of 905 actual draws
   - This is a known limitation in football prediction literature

2. **Home Field Bias**
   - Model over-predicts home wins (2,743 predicted vs 1,736 actual)
   - Trade-off: Achieves excellent home win recall (80.88%)

3. **New Team Handling**
   - Teams not in training data receive uniform probabilities (33% each)
   - Improves as season progresses and data accumulates

## Future Enhancements

Potential improvements for increased accuracy:

1. **Advanced Features**
   - Opponent-adjusted expected goals
   - Recent form indicators (last 5 matches)
   - Head-to-head historical records
   - Injury/suspension data integration

2. **Model Improvements**
   - Bivariate Poisson model (accounts for correlation)
   - Dixon-Coles time decay weighting
   - Machine learning ensemble methods
   - Bayesian updating for in-season adaptation

3. **Data Enrichment**
   - Live API integration for real-time fixtures
   - Player-level statistics
   - Weather conditions
   - Betting market odds

4. **Production Features**
   - Automated retraining pipeline
   - Prediction tracking dashboard
   - Confidence intervals
   - Betting strategy simulation

## Contributing

This is a personal research project. For questions or suggestions, please open an issue on GitHub.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Data source: Understat.com
- Statistical methods based on academic football prediction literature
- Inspired by Dixon-Coles (1997) modeling framework

## References

1. Dixon, M. J., & Coles, S. G. (1997). Modelling Association Football Scores and Inefficiencies in the Football Betting Market. *Journal of the Royal Statistical Society*.

2. Maher, M. J. (1982). Modelling Association Football Scores. *Statistica Neerlandica*.

3. Karlis, D., & Ntzoufras, I. (2003). Analysis of Sports Data by Using Bivariate Poisson Models. *Journal of the Royal Statistical Society*.

---

**Project Status**: Production Ready

**Last Updated**: October 2025

**Contact**: [GitHub Issues](https://github.com/raynergoh/EPL-Predictor/issues)
