import pandas as pd
import xgboost as xgb
import joblib
import pickle
import json
import numpy as np
import os
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, accuracy_score
from pandasgui import show

# Loading Stress Model and Bins
with open('Models/DefaultGivenDistress.pkl', 'rb') as file:
    model = pickle.load(file)


# Define the root directory
root_dir = './ValData'

start_year = 2007
end_year = 2007

TEST_list = []
LABELS_list = []
IDs_list = []


SALE_COLUMNS = ["Actual Loss Calculation", "Zero Balance Removal UPB", "Net Sales Proceeds", 
                "Delinquent Accrued Interest", "Expenses", "MI Recoveries", "Non MI Recoveries"]

MISC_COLUMNS_TO_DROP = ["MSA", 'Postal Code', 'Major Stress', "Monthly Reporting Period",
                        'Distress Date', 'Zero Balance Code', 'Unresolved', 'Last Time Current',
                         'Property State', 'Current Loan Delinquency Status', 'Loan Sequence Number',
                         'Estimated Loan-to-Value (ELTV)']
CAT_COLUMNS = []

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
                #df = df[(df["Major Stress"] == 1)]
                #if year != 2024:
                #    df = df[~(df["Unresolved"] == 1)]
                TEST_list.append(df.drop(columns=['Default Flag']))
                LABELS_list.append(df['Default Flag'])
                IDs_list.append(df[["Loan Sequence Number", "Monthly Reporting Period"]])

               
# Combine everything into single DataFrames
TEST = pd.concat(TEST_list, ignore_index=True).drop(columns=SALE_COLUMNS)
LABELS = pd.concat(LABELS_list, ignore_index=True)
IDs = pd.concat(IDs_list, ignore_index=True)

TEST = TEST.drop(columns=MISC_COLUMNS_TO_DROP)
categorical_cols = TEST.select_dtypes(include=['object', 'category']).columns.tolist()

# Use below to add columns that we want dummies for but may not current be dtype categorical
# categorical_cols += CAT_COLUMNS

TEST= pd.get_dummies(TEST, columns=categorical_cols, drop_first=True)
TEST = TEST.drop(columns = ['Program Indicator_R'], errors = 'ignore')

TEST["Est. Home Val at Origination"] = TEST["Original UPB"] / TEST["Original Combined Loan-to-Value (CLTV)"] * 100
TEST["Est. Current Home Val"] = TEST["Est. Home Val at Origination"] * (TEST["HPI Change %"] + 1)

TEST["Geographic Est. ELTV"] = TEST["Current Actual UPB"] / TEST["Est. Current Home Val"]

def estimate_months_remaining(row):
    L = row['Original UPB']
    B = row['Current Actual UPB']
    annual_rate = row["Original Interest Rate"]
    N_orig = row['Original Loan Term']
    
    r = annual_rate / 100 / 12

    if r == 0:
        P = L / N_orig
    else:
        P = (L * r) / (1 - (1 + r) ** -N_orig)

    if P <= r * B:
        return np.nan  # Loan isn't amortizing

    N_remaining = (np.log(P) - np.log(P - r * B)) / np.log(1 + r)
    return round(N_remaining)

TEST.loc[TEST["Remaining Months to Legal Maturity"].isna(), "Remaining Months to Legal Maturity"] = TEST[TEST["Remaining Months to Legal Maturity"].isna()].apply(lambda row : estimate_months_remaining(row), axis=1)

training_features = model.get_booster().feature_names

for col in training_features:
    if col not in TEST.columns:
        TEST[col] = 0

# Now re-order columns to match model input
TEST = TEST[training_features]

print(TEST.shape)


preds = model.predict_proba(TEST)[:, 1]

#ap_score = average_precision_score(LABELS, preds)
#brier = brier_score_loss(LABELS, preds)
#logloss = log_loss(LABELS, preds)
#accuracy = accuracy_score(LABELS, model.predict(TEST))

#print("\nTest Metrics:")
#print(f"Average Precision Score: {ap_score:.4f}")
#print(f"Brier Score: {brier:.6f}")
#print(f"Log Loss: {logloss:.6f}")
#print(f"Accuracy: {accuracy}")


df = pd.DataFrame({
    'y_true': LABELS,
    'y_prob': preds,
    'Loan Sequence Number': IDs["Loan Sequence Number"],
    'Monthly Reporting Period' : IDs["Monthly Reporting Period"]

})


#print(LABELS.mean())

# Write to Parquet
df.to_parquet("ValPredictions/DefGivenStress2007.parquet", index=False)


