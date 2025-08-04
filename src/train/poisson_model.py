import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
import os

def train_models(features_csv, labels_home_csv, labels_away_csv, model_output_path):
    # Load features and labels
    X = pd.read_csv(features_csv)
    y_home = pd.read_csv(labels_home_csv)
    y_away = pd.read_csv(labels_away_csv)

    # Combine features and labels into training datasets
    data_home = X.copy()
    data_home['goals'] = y_home.squeeze()  # Convert to Series if single column

    data_away = X.copy()
    data_away['goals'] = y_away.squeeze()

    # Define formula including additional rolling features if exist
    # Assumes rolling features have 'home_' or 'away_' prefix, adapt if different
    home_features = "home_avg_xg + away_avg_xg + home_field"
    away_features = "away_avg_xg + home_avg_xg"

    # Add rolling feature columns if present
    rolling_cols_home = [col for col in X.columns if col.startswith('home_form_')]
    rolling_cols_away = [col for col in X.columns if col.startswith('away_form_')]

    if rolling_cols_home:
        home_features += " + " + " + ".join(rolling_cols_home)
    if rolling_cols_away:
        away_features += " + " + " + ".join(rolling_cols_away)

    formula_home = f"goals ~ {home_features}"
    formula_away = f"goals ~ {away_features}"

    # Fit Poisson regression models
    model_home = smf.glm(formula=formula_home, data=data_home, family=sm.families.Poisson()).fit()
    print("Home model summary:")
    print(model_home.summary())

    model_away = smf.glm(formula=formula_away, data=data_away, family=sm.families.Poisson()).fit()
    print("Away model summary:")
    print(model_away.summary())

    # Save models
    os.makedirs(model_output_path, exist_ok=True)
    with open(os.path.join(model_output_path, "model_home.pkl"), 'wb') as f:
        pickle.dump(model_home, f)
    with open(os.path.join(model_output_path, "model_away.pkl"), 'wb') as f:
        pickle.dump(model_away, f)

    print(f"Models saved to {model_output_path}")

if __name__ == "__main__":
    train_models(
        features_csv="data/processed/features.csv",
        labels_home_csv="data/processed/labels_home.csv",
        labels_away_csv="data/processed/labels_away.csv",
        model_output_path="models"
    )
