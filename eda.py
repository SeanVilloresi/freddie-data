import pandas as pd
from pandasgui import show

column_names = [
    "Credit Score",
    "First Payment Date",
    "First Time Homebuyer Flag",
    "Maturity Date",
    "Metropolitan Statistical Area (MSA) Or Metropolitan Division",
    "Mortgage Insurance Percentage (MI %)",
    "Number of Units",
    "Occupancy Status",
    "Original Combined Loan-to-Value (CLTV)",
    "Original Debt-to-Income (DTI) Ratio",
    "Original UPB",
    "Original Loan-to-Value (LTV)",
    "Original Interest Rate",
    "Channel",
    "Prepayment Penalty Mortgage (PPM) Flag",
    "Amortization Type (Formerly Product Type)",
    "Property State",
    "Property Type",
    "Postal Code",
    "Loan Sequence Number",
    "Loan Purpose",
    "Original Loan Term",
    "Number of Borrowers",
    "Seller Name",
    "Servicer Name",
    "Super Conforming Flag",
    "Pre-HARP Loan Sequence Number",
    "Program Indicator",
    "HARP Indicator",
    "Property Valuation Method",
    "Interest Only (I/O) Indicator",
    "Mortgage Insurance Cancellation Indicator"
]
monthly_performance_columns = [
    "Loan Sequence Number",
    "Monthly Reporting Period",
    "Current Actual UPB",
    "Current Loan Delinquency Status",
    "Loan Age",
    "Remaining Months to Legal Maturity",
    "Defect Settlement Date",
    "Modification Flag",
    "Zero Balance Code",
    "Zero Balance Effective Date",
    "Current Interest Rate",
    "Current Deferred UPB",
    "Due Date of Last Paid Installment (DDLPI)",
    "MI Recoveries",
    "Net Sales Proceeds",
    "Non MI Recoveries",
    "Expenses",
    "Legal Costs",
    "Maintenance and Preservation Costs",
    "Taxes and Insurance",
    "Miscellaneous Expenses",
    "Actual Loss Calculation",
    "Modification Cost",
    "Step Modification Flag",
    "Deferred Payment Plan",
    "Estimated Loan-to-Value (ELTV)",
    "Zero Balance Removal UPB",
    "Delinquent Accrued Interest",
    "Delinquency Due to Disaster",
    "Borrower Assistance Status Code",
    "Current Month Modification Cost",
    "Interest Bearing UPB"
]

# df_orig = pd.read_csv('sample_orig_2011.txt', sep='|', header=None)
# df_orig.columns = column_names
# df_month = pd.read_csv('sample_svcg_2011.txt', sep='|', header=None)
# df_month.columns = monthly_performance_columns

# df_month.to_parquet()



# Group by loan, get unique delinquency statuses per loan
# loan_delinquency = df_month.groupby("Loan Sequence Number")["Current Loan Delinquency Status"].unique()

# Flatten the list of all unique statuses across loans
from collections import Counter
# delinquency_counter = Counter()

# for statuses in loan_delinquency:
#     delinquency_counter.update(statuses)


# delinquency_ever_counts = pd.Series({str(k): v for k, v in delinquency_counter.items()}).sort_values(ascending=False)
# delinquency_counts = delinquency_ever_counts[delinquency_ever_counts >= 50]
# print(delinquency_counts)

# Step 1: Find loan IDs that ever had delinquency status of 3
# loans_with_dq3 = df_month["Loan Sequence Number"].unique()
# print(len(loans_with_dq3))

# # Step 2: Filter the full DataFrame to only those loans
# df_dq3_loans = df_month[df_month["Loan Sequence Number"].isin(loans_with_dq3)]

# # Step 3: For those loans, find the unique zero balance codes they ever had
# zero_balance_for_dq3 = df_dq3_loans[df_dq3_loans["Zero Balance Code"].notna()]

# # Step 4: Count the different Zero Balance Codes
# zero_balance_code_counts = zero_balance_for_dq3["Zero Balance Code"].value_counts(dropna=False)

# print(zero_balance_code_counts)F11Q10161782

###################################################################################################################

df_month = pd.read_csv("PSample/sample_svcg_2011.txt", sep='|', header=None)
df_month.columns = monthly_performance_columns
show(df_month)
# show(df_month)

# sample = pd.read_parquet("DefaultData/Defaults2012.parquet")
# print(sample['Original Debt-to-Income (DTI) Ratio'].value_counts())

# show(sample)

# orig = pd.read_csv('PSample/sample_svcg_2011.txt', sep='|', header=None)

# orig.columns = monthly_performance_columns

# show(orig[orig['Loan Sequence Number'] == 'F11Q10002482'])


