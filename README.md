# EPL-Predictor

**EPL-Predictor** is a data science and machine learning project for forecasting English Premier League (EPL) match outcomes using lineup-based xG models, rolling team form, and Poisson regression. The project scrapes predicted lineups, computes expected goals based on player stats, tracks recent performance for each club, and simulates result probabilities for all upcoming fixtures.

---

## Features

- **Automated Fixture & Lineup Scraper:**  
  Scrapes predicted lineups and next fixtures from Fantasy Football Scout.

- **Lineup-Based xG Calculation:**  
  Computes each team's likely attacking output for the upcoming fixture using player-level xG stats, with home/away adjustments.

- **Historical Data Extraction:**  
  Downloads and stores historic EPL scores and match xG.

- **Feature Engineering:**  
  Calculates rolling averages for team form and defensive strength.

- **Poisson Regression Modeling:**  
  Trains predictive models for home/away goals with xG, form, and home field advantage as features.

- **Full Match Simulation:**  
  Simulates expected goals, win/draw/loss probabilities, and full scoreline probability matrices.

- **Result Storage and Reporting:**  
  Outputs results to tables; saves as CSV/JSON for further analysis or visualization.

---

## Getting Started

1. **Install dependencies:**  

```sh
pip install -r requirements.txt
```

2. **Download EPL historical data:**  

```sh
python -m src.data.extract_historical_data
```

3. **Feature engineering:**  

```sh
python -m src.features.build_features
```

4. **Train prediction models:**  

```sh
python -m src.models.poisson_model
```

5. **Run end-to-end prediction for next matches:**  

```sh
python -m src.main
```
- This will scrape lineups, compute xG, apply prediction models, and output win/draw/loss probabilities as well as most likely scorelines for all forthcoming fixtures.

---

## Keeping Results Updated

- **After each matchweek:**  
- Add new results to `data/raw/epl_historical_results.csv`
- Rerun feature building and model training steps to keep predictions current.

- **Results storage:**  
- Save all predictions in `/results/` as date/gameweek-stamped CSV and JSON files for future reference and evaluation.

---

## Outputs

For each predicted fixture, you receive:

- Expected goals for home and away
- Win/draw/loss probabilities
- Most likely and top-3 scorelines (with probability)
- Full probability matrix for detailed outcome analysis
- All results can be exported as CSV/JSON for further reporting, visualization, or dashboarding

**Sample Output:**

```sh
Manchester United vs Arsenal
Expected goals: 0.41-1.57
Most likely scoreline: 0-1 (21.63%)
Top 3 scorelines:
  0-1: 21.63%
  0-2: 16.93%
  0-0: 13.81%
Probability matrix (rows: home goals 0-6, cols: away goals 0-6):
[[0.138 0.216 0.169 0.088 0.035 0.011 0.003]
 [0.057 0.09  0.07  0.037 0.014 0.004 0.001]
 [0.012 0.019 0.015 0.008 0.003 0.001 0.   ]
 [0.002 0.003 0.002 0.001 0.    0.    0.   ]
 [0.    0.    0.    0.    0.    0.    0.   ]
 [0.    0.    0.    0.    0.    0.    0.   ]
 [0.    0.    0.    0.    0.    0.    0.   ]]
Probabilities: Home 9.46%, Draw 24.33%, Away 66.10%
```
---

## Credits

- Understat, Fantasy Football Scout, and other open football data sources.
- Player and match-level stats used as core inputs.

---