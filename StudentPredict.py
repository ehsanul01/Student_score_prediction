# student_score_predictor.py
# Generate a 100-row synthetic dataset, train Linear Regression, evaluate, and optionally predict.
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_CSV = "student_scores_100.csv"
PLOT_PNG = "student_score_regression_100.png"
RANDOM_SEED = 42

def make_data(n: int = 100, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 12, size=n)
    noise = rng.normal(0, 6, size=n)
    scores = np.clip(40 + 5 * hours + noise, 0, 100)
    df = pd.DataFrame(
        {"hours_studied": hours.round(2), "final_score": scores.round(1)}
    )
    return df

def train_and_eval(df: pd.DataFrame):
    X = df[["hours_studied"]].values
    y = df["final_score"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, {"MAE": mae, "RMSE": rmse, "R2": r2}

def save_plot(model: LinearRegression, df: pd.DataFrame, out_path: str = PLOT_PNG):
    # scatter of training data + regression line
    X = df[["hours_studied"]].values
    y = df["final_score"].values
    plt.figure()
    plt.scatter(X, y, label="Data")
    x_line = np.linspace(0, 12, 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, label="Regression line")
    plt.title("Student Score Predictor (100 People)")
    plt.xlabel("Hours Studied")
    plt.ylabel("Final Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train a Student Score Predictor.")
    parser.add_argument("--data", type=str, default=DATA_CSV,
                        help="Path to CSV (will be created if missing).")
    parser.add_argument("--hours", type=float, default=None,
                        help="If provided, predict score for this number of hours.")
    parser.add_argument("--regen", action="store_true",
                        help="Regenerate the synthetic dataset (100 rows).")
    args = parser.parse_args()

    csv_path = Path(args.data)

    # Create or regenerate data if requested or missing
    if args.regen or not csv_path.exists():
        df = make_data(n=100, seed=RANDOM_SEED)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Wrote dataset: {csv_path.resolve()}")
    else:
        df = pd.read_csv(csv_path)

    # Train & evaluate
    model, metrics = train_and_eval(df)
    print("[METRICS]", {k: round(v, 4) for k, v in metrics.items()})
    print(f"[MODEL] coef={model.coef_[0]:.4f}, intercept={model.intercept_:.4f}")

    # Save plot
    save_plot(model, df, out_path=PLOT_PNG)
    print(f"[PLOT] Saved regression plot to: {Path(PLOT_PNG).resolve()}")

    # Optional single prediction
    if args.hours is not None:
        pred = model.predict(np.array([[args.hours]])).item()
        print(f"[PREDICT] Hours={args.hours:.2f} → Predicted score={pred:.2f}")

if __name__ == "__main__":
    main() 