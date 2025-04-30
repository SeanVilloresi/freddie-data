import pandas as pd
import xgboost as xgb
import joblib
import pickle
import json
import numpy as np
import os
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from pandasgui import show

# Loading Stress Model and Bins
model = xgb.Booster()
model.load_model("ensemble_models/best_model.xgb")

# Load JSON file
with open("calibration_thresholds.json", "r") as f:
    calib_data = json.load(f)

# Extract intervals and probabilities
intervals = [(entry["low"], entry["high"]) for entry in calib_data]
probs = [entry["p"] for entry in calib_data]


def apply_empirical_calibration(y_scores, intervals, probs):
    """
    Vectorized mapping from raw y_scores → calibrated probs
    intervals : list of (low, high)
    probs     : list of empirical P(y=1) per interval
    """
    y_scores = np.asarray(y_scores)
    calibrated = np.zeros_like(y_scores, dtype=float)
    for (low, high), p in zip(intervals, probs):
        mask = (y_scores >= low) & (y_scores <= high)
        calibrated[mask] = p
    return calibrated

# Define the root directory
root_dir = './TrainingData'

start_year = 2022
end_year = 2023

TEST_list = []
LABELS_list = []


SALE_COLUMNS = [
    "Actual Loss Calculation", "Zero Balance Removal UPB", "Net Sales Proceeds", 
    "Delinquent Accrued Interest", "Expenses", "MI Recoveries", "Non MI Recoveries"
]

MISC_COLUMNS_TO_DROP = [
    "MSA", 'Postal Code',  'Monthly Reporting Period', 'Distress Date', 'Default Flag', 
    'Zero Balance Code', 'Unresolved', 'Last Time Current', 'Property State', 
    'Current Loan Delinquency Status', 'Loan Sequence Number'
]

CAT_COLUMNS = []

fold = 0

for year_folder in sorted(os.listdir(root_dir)):
    if not year_folder.startswith("Year"):
        continue
    try:
        year = int(year_folder.replace("Year", ""))
    except ValueError:
        continue
    if not (start_year <= year <= end_year):
        continue

    year_path = os.path.join(root_dir, year_folder)
    if os.path.isdir(year_path):
        for file in os.listdir(year_path):
            if file.endswith('.parquet'):
                file_path = os.path.join(year_path, file)
                df = pd.read_parquet(file_path)
                TEST_list.append(df.drop(columns=['Major Stress']))
                LABELS_list.append(df['Major Stress'])
               
# Combine everything into single DataFrames
TEST = pd.concat(TEST_list, ignore_index=True).drop(columns=SALE_COLUMNS)
LABELS = pd.concat(LABELS_list, ignore_index=True)

TEST = TEST.drop(columns=MISC_COLUMNS_TO_DROP)
categorical_cols = TEST.select_dtypes(include=['object', 'category']).columns.tolist()

# Use below to add columns that we want dummies for but may not current be dtype categorical
# categorical_cols += CAT_COLUMNS

TEST= pd.get_dummies(TEST, columns=categorical_cols, drop_first=True)
TEST = TEST.drop(columns = ['Program Indicator_R'], errors = 'ignore')

training_features = model.feature_names
for col in training_features:
    if col not in TEST.columns:
        TEST[col] = 0

# Now re-order columns to match model input
TEST = TEST[training_features]

LABELS = pd.concat(LABELS_list).reset_index(drop=True)
LABELS.fillna(0, inplace=True)

print(TEST.shape)

dtest = xgb.DMatrix(TEST)
preds = model.predict(dtest)

ap_score = average_precision_score(LABELS, preds)
brier = brier_score_loss(LABELS, preds)
logloss = log_loss(LABELS, preds)

print("\nTest Metrics:")
print(f"Average Precision Score: {ap_score:.4f}")
print(f"Brier Score: {brier:.6f}")
print(f"Log Loss: {logloss:.6f}")

# Apply empirical calibration
calibrated_preds = apply_empirical_calibration(preds, intervals, probs)

# Calibrated metrics
ap_score_cal = average_precision_score(LABELS, calibrated_preds)
brier_cal = brier_score_loss(LABELS, calibrated_preds)
logloss_cal = log_loss(LABELS, calibrated_preds)

print("\nTest Metrics (Calibrated):")
print(f"Average Precision Score: {ap_score_cal:.4f}")
print(f"Brier Score: {brier_cal:.6f}")
print(f"Log Loss: {logloss_cal:.6f}")

df = pd.DataFrame({
    'y_true': LABELS,
    'y_prob': preds,
    'y_calib_prob': calibrated_preds
})

df['y_calib_prob'] = df['y_calib_prob'] + 1e-10

# # Bin by original (uncalibrated) predicted probabilities
# df['prob_bin'] = pd.qcut(df['y_calib_prob'], q=100, labels=False, duplicates='drop')

# # Group by bin and compute both raw and calibrated stats
# calibration_df = df.groupby('prob_bin').agg(
#     avg_predicted_prob=('y_prob', 'mean'),
#     avg_calibrated_prob=('y_calib_prob', 'mean'),
#     actual_rate=('y_true', 'mean'),
#     count=('y_true', 'count')
# ).reset_index()

# # Optional: round for display
# calibration_df = calibration_df.round(4)
# show(calibration_df)

# Reload Loan Sequence Number from the raw input files
loan_ids = []
for year_folder in sorted(os.listdir(root_dir)):
    if not year_folder.startswith("Year"):
        continue
    try:
        year = int(year_folder.replace("Year", ""))
    except ValueError:
        continue
    if not (start_year <= year <= end_year):
        continue
    year_path = os.path.join(root_dir, year_folder)
    if os.path.isdir(year_path):
        for file in os.listdir(year_path):
            if file.endswith('.parquet'):
                file_path = os.path.join(year_path, file)
                df = pd.read_parquet(file_path, columns=["Loan Sequence Number", "Monthly Reporting Period"])
                loan_ids.append(df)

# Concatenate all loan metadata
loan_metadata = pd.concat(loan_ids, ignore_index=True)

# Build output DataFrame
output_df = loan_metadata.copy()
output_df["Calibrated P(Stress)"] = calibrated_preds

# Write to Parquet
output_df.to_parquet("calibrated_predictions_2023.parquet", index=False)


