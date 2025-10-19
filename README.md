#  Student Score Predictor

A small machine learning project that predicts a student's final exam score based on the number of hours studied, using **Linear Regression**.

---

##  Project Structure
```
StudentPrediction/
│
├── StudentPredict.py                 # Main training & prediction script
├── student_scores_100.csv            # Synthetic dataset (auto-generated)
├── student_score_regression_100.png  # Regression plot
└── .venv/                            # (optional) virtual environment folder
```

---

## Requirements

Install these packages once:

```bash
pip install numpy pandas scikit-learn matplotlib
```

or if you use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib
```

---

##  How to Run

### 1. Generate data + train model
```bash
python3 StudentPredict.py --regen
```

This creates a dataset (`student_scores_100.csv`), trains a regression model, evaluates performance, and saves a plot (`student_score_regression_100.png`).

### 2. Predict a score
```bash
python3 StudentPredict.py --hours 6
```

Output example:
```
[METRICS] {'MAE': 4.8577, 'RMSE': 5.8331, 'R2': 0.8368}
[MODEL] coef=5.0239, intercept=39.7923
[PREDICT] Hours=6.00 → Predicted score=69.94
```

---

##  Output Files

| File | Description |
|------|--------------|
| `student_scores_100.csv` | dataset of 100 synthetic students |
| `student_score_regression_100.png` | plot showing regression fit |
| Terminal metrics | model performance (MAE, RMSE, R²) |

---

##  Concepts Learned

- Data generation with NumPy & Pandas  
- Train/Test split  
- Linear Regression model training  
- Model evaluation (MAE, RMSE, R²)  
- Data visualization with Matplotlib  

---

