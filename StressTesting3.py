import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# 1) Load LGD model
with open("Models/LGD.pkl", "rb") as f:
    lgd_model = pickle.load(f)

# 2) Set up directories and lists
root_dir = './ValData'

TEST_list = []
metadata_list = []

# Columns to drop after saving metadata
MISC_COLUMNS_TO_DROP = [
    "MSA", 'Postal Code', 'Default Flag', 'Major Stress',
    'Distress Date', 'Unresolved', 'Last Time Current',
    'Property State', 'Current Loan Delinquency Status',
    'Estimated Loan-to-Value (ELTV)', "Loan Sequence Number", "Monthly Reporting Period"
]

# 3) Load and filter data, capturing metadata
for year in range(2007, 2008):
    year_path = os.path.join(root_dir, f"Year{year}")
    if not os.path.isdir(year_path):
        continue

    for file in os.listdir(year_path):
        if not file.endswith('.parquet'):
            continue

        file_path = os.path.join(year_path, file)
        df = pd.read_parquet(file_path)
       
        # df = df[
        #     (df["Default Flag"] == 1) &
        #     (~df["Actual Loss Calculation"].isna()) &
        #     (df["Actual Loss Calculation"] != 0)
        # ]
        if df.empty:
            continue
        # save metadata before dropping
        metadata_list.append(
            df[["Loan Sequence Number", "Monthly Reporting Period", "Major Stress", "Default Flag", "Unresolved", "Actual Loss Calculation", 'Current Actual UPB']].reset_index(drop=True)
        )
        # keep the rest in TEST_list
        TEST_list.append(df.reset_index(drop=True))

# 4) Concatenate TEST and metadata
TEST = pd.concat(TEST_list, ignore_index=True)
metadata_df = pd.concat(metadata_list, ignore_index=True)
print(sum(TEST['Actual Loss Calculation'].value_counts()))

# 5) Drop unwanted columns from TEST
TEST = TEST.drop(columns=MISC_COLUMNS_TO_DROP)

# 6) One-hot encode categoricals
categorical_cols = TEST.select_dtypes(include=['object', 'category']).columns.tolist()
TEST = pd.get_dummies(TEST, columns=categorical_cols, drop_first=True)

# 7) Build additional features
# Estimated home values
TEST["Est. Home Val at Origination"] = (
    TEST["Original UPB"] /
    (TEST["Original Combined Loan-to-Value (CLTV)"] / 100)
)
TEST["Est. Current Home Val"] = (
    TEST["Est. Home Val at Origination"] *
    (TEST["HPI Change %"] + 1)
)
# Geographic ELTV
TEST["Geographic ELTV"] = (
    TEST["Current Actual UPB"] /
    TEST["Est. Current Home Val"]
)

# 8) Estimate months remaining where missing
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
        return np.nan  # loan isn't amortizing

    N_remaining = (np.log(P) - np.log(P - r * B)) / np.log(1 + r)
    return round(N_remaining)

mask = TEST["Remaining Months to Legal Maturity"].isna()
TEST.loc[mask, "Remaining Months to Legal Maturity"] = (
    TEST.loc[mask]
    .apply(estimate_months_remaining, axis=1)
)

# 9) Clean up MI percentage
TEST.loc[TEST["Mortgage Insurance Percentage (MI %)"] == 999, "Mortgage Insurance Percentage (MI %)"] = 0

# 10) Compute MI cap and recoveries
TEST["MI Cap Payout"] = (
    TEST["Mortgage Insurance Percentage (MI %)"] *
    TEST["Original UPB"] / 100
)

# create a one-feature LinearRegression for interest/expenses
lr_model = LinearRegression()
lr_model.coef_ = np.array([0.14490168125835193])
lr_model.intercept_ = 7234.557086138826
lr_model.n_features_in_ = 1

TEST["Estimated Interest and Expenses"] = lr_model.predict(
    TEST[["Est. Current Home Val"]]
)

TEST["Estimated MI Recoveries"] = (
    TEST[["MI Cap Payout", "Current Actual UPB", "Est. Current Home Val", "Estimated Interest and Expenses"]]
    .apply(lambda row: max(
        0,
        min(
            row["MI Cap Payout"],
            row["Current Actual UPB"] - row["Est. Current Home Val"] + row["Estimated Interest and Expenses"]
        )
    ), axis=1)
)

# 11) Prepare features for LGD prediction
FEATURES = TEST[[
    "Mortgage Insurance Percentage (MI %)",
    "MI Cap Payout",
    "Estimated MI Recoveries",
    "Geographic ELTV",
    "Original Combined Loan-to-Value (CLTV)",
    "Est. Current Home Val",
    "Remaining Months to Legal Maturity",
    "HPI Change %",
    "Current Actual UPB",
    "Numeric Delinquency"
]]

# 12) Predict LGD
predicted_lgd = lgd_model.predict(FEATURES)
true_lgd = (-1 * TEST["Actual Loss Calculation"]).clip(lower=0).values

# 13) Build output DataFrame
output_df = metadata_df.copy()
output_df["LGD"] = predicted_lgd

# 13) Build output DataFrame
output_df = metadata_df.copy()
output_df["Predicted LGD"] = predicted_lgd
output_df["True LGD"] = true_lgd

# 14) Save to Parquet
output_df.to_parquet("lgd_predictions_2007.parquet", index=False)
