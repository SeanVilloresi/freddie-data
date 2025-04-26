import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm


def ReadPTXT(year):
    Pcols = [
        "Loan Sequence Number", "Monthly Reporting Period", "Current Actual UPB", "Current Loan Delinquency Status",
        "Loan Age", "Remaining Months to Legal Maturity", "Defect Settlement Date", "Modification Flag", "Zero Balance Code", 
        "Zero Balance Effective Date", "Current Interest Rate", "Current Deferred UPB", "Due Date of Last Paid Installment (DDLPI)",
        "MI Recoveries", "Net Sales Proceeds", "Non MI Recoveries", "Expenses", "Legal Costs", "Maintenance and Preservation Costs",
        "Taxes and Insurance", "Miscellaneous Expenses", "Actual Loss Calculation", "Modification Cost", "Step Modification Flag",
        "Deferred Payment Plan", "Estimated Loan-to-Value (ELTV)", "Zero Balance Removal UPB", "Delinquent Accrued Interest",
        "Delinquency Due to Disaster", "Borrower Assistance Status Code", "Current Month Modification Cost", "Interest Bearing UPB"
        ]
      
    df = pd.read_csv(f'PSample/sample_svcg_{year}.txt', sep='|', header=None, 
                     dtype={i : str for i in [3, 7, 23, 24, 28,29]})
    df.columns = Pcols

    return df

def ReadOTXT(year):

    Ocols = [
        "Credit Score", "First Payment Date", "First Time Homebuyer Flag", "Maturity Date", 
        "Metropolitan Statistical Area (MSA) Or Metropolitan Division", "Mortgage Insurance Percentage (MI %)", "Number of Units", 
        "Occupancy Status", "Original Combined Loan-to-Value (CLTV)", "Original Debt-to-Income (DTI) Ratio", 
        "Original UPB", "Original Loan-to-Value (LTV)", "Original Interest Rate", "Channel", "Prepayment Penalty Mortgage (PPM) Flag", 
        "Amortization Type (Formerly Product Type)", "Property State", "Property Type", "Postal Code", "Loan Sequence Number", 
        "Loan Purpose", "Original Loan Term", "Number of Borrowers", "Seller Name", "Servicer Name", "Super Conforming Flag", 
        "Pre-HARP Loan Sequence Number", "Program Indicator", "HARP Indicator", "Property Valuation Method", 
        "Interest Only (I/O) Indicator", "Mortgage Insurance Cancellation Indicator"
        ]
    
    df = pd.read_csv(f'OSample/sample_orig_{year}.txt', sep='|', header=None)
    df.columns = Ocols
    return df

def WriteDistressDataset(year):
    P = ReadPTXT(year)
    P = P.rename(columns={'Monthly Reporting Period':'Distress Date'})

    # Create Stress Flag
    P.loc[(~P["Current Loan Delinquency Status"].isin(('0','1','2','3','4','5'))) | (P["Zero Balance Code"].isin((2.0, 3.0, 9.0))),"Major Stress"] = 1

    # Find Invalid Defaults
    ZB1 = set(P[P["Zero Balance Code"] == 1.0]["Loan Sequence Number"].unique())
    P.loc[(P["Loan Sequence Number"].isin(ZB1)),"Ends In Payoff"] = 1
    MostRecentCurrency = (P[P["Current Loan Delinquency Status"] == '0'].groupby("Loan Sequence Number")["Distress Date"]
                          .max().reset_index().rename(columns={"Distress Date": "Most Recent Currency"}))
    P = P.merge(MostRecentCurrency, on="Loan Sequence Number", how="left")
    UnresolvedDelinquency = set(P[(P["Distress Date"] == 202409) & (P["Current Loan Delinquency Status"] != "0")]["Loan Sequence Number"].dropna().unique())
    P.loc[(P["Loan Sequence Number"].isin(UnresolvedDelinquency)),"Unresolved Delinquency"] = 1
    P.loc[(P["Ends In Payoff"] == 1) | (P["Distress Date"] < P["Most Recent Currency"]) | (P["Unresolved Delinquency"] == 1), "Invalid Default"] = 1

    # Get Loss Data
    P.loc[(P["Major Stress"]==1) & (~(P["Invalid Default"] == 1)), "Default Flag"] = 1
    
    LossData = P[P["Default Flag"]==1][["Loan Sequence Number","Distress Date", "Default Flag",
                                           "Actual Loss Calculation", "Zero Balance Removal UPB", "Net Sales Proceeds", 
                                           "Delinquent Accrued Interest", "Expenses", "MI Recoveries", "Non MI Recoveries"]]
    LossData = LossData.sort_values(by=['Loan Sequence Number', 'Distress Date'])
    LossData = LossData.groupby("Loan Sequence Number").first().reset_index()

    # Create Distress Dataset
    P = P.drop(columns=["Actual Loss Calculation", "Zero Balance Removal UPB", "Net Sales Proceeds", "Delinquent Accrued Interest", 
                        "Expenses", "MI Recoveries", "Non MI Recoveries"])
    P = P.merge(LossData, on=["Loan Sequence Number", "Distress Date", "Default Flag"], how="left")
    
    Distress = P[(P["Major Stress"] == 1)]
    DistressColumns = ["Loan Sequence Number", "Distress Date", "Major Stress", "Default Flag",
                       "Zero Balance Code", "Unresolved Delinquency", "Most Recent Currency", 
                       "Actual Loss Calculation", "Zero Balance Removal UPB", "Net Sales Proceeds", "Delinquent Accrued Interest", 
                       "Expenses", "MI Recoveries", "Non MI Recoveries"
                       ]
    
    Distress[DistressColumns].to_parquet(f"DefaultData/Defaults{year}.parquet", engine="pyarrow", index=False)
    print(f"Wrote DefaultData/Defaults{year}.parquet!")

def WriteTrainingData(year):
    
    PColsToDrop = [
        "Defect Settlement Date", "Zero Balance Code", "Zero Balance Effective Date", "Due Date of Last Paid Installment (DDLPI)",
        "MI Recoveries", "Net Sales Proceeds", "Non MI Recoveries", "Expenses", "Legal Costs", "Maintenance and Preservation Costs",
        "Taxes and Insurance", "Miscellaneous Expenses", "Actual Loss Calculation", "Modification Cost", 
        "Zero Balance Removal UPB", "Delinquent Accrued Interest", "Current Month Modification Cost"
        ]
    
    OColsToDrop = [
        "First Payment Date", "Maturity Date", "Channel", "Seller Name", "Servicer Name", "Super Conforming Flag", 
        "Pre-HARP Loan Sequence Number", "HARP Indicator"
        ]
    
    P = ReadPTXT(year).drop(columns = PColsToDrop)
    O = ReadOTXT(year).drop(columns = OColsToDrop)
    D = pd.read_parquet(f"DefaultData/Defaults{year}.parquet", engine="pyarrow")

    print("Data Loaded!")

    # Maximum Number of Months Each Loan Has Been PReviously Delinquent
    P["Numeric Delinquency"] = pd.to_numeric(P["Current Loan Delinquency Status"], errors="coerce")
    P = P.sort_values(["Loan Sequence Number", "Monthly Reporting Period"])
    P["MaxPriorDelinquency"] = (P.groupby("Loan Sequence Number")["Numeric Delinquency"].expanding().max().shift().fillna(0).reset_index(level=0, drop=True))

    P = P[P["Current Loan Delinquency Status"].isin(('0','1','2','3','4','5'))]
    P = P[P["Monthly Reporting Period"].between(201101, 201812)]
    P = P[P["Monthly Reporting Period"] % 100 == 1]


    MERGED = O.merge(P, on="Loan Sequence Number", how="inner")

    print("Merged Origination & Performance!")

    DistressedLoans = set(D['Loan Sequence Number'].unique())
    DistressDatePairs = set(zip(D['Loan Sequence Number'], D['Distress Date']))

    def add_months(yyyymm, months):
        year = yyyymm // 100
        month = yyyymm % 100
        NewDate = pd.Timestamp(year=year, month=month, day=1) + relativedelta(months=months)
        return NewDate.year * 100 + NewDate.month

    def find_first_match(row):
        
        ID = row['Loan Sequence Number']
        if ID not in DistressedLoans:
            return pd.NA
        
        Date = row['Monthly Reporting Period']
        for i in range(1, 13):
            NextDate = add_months(Date, i)
            if (ID, NextDate) in DistressDatePairs:
                return NextDate
        return pd.NA

    tqdm.pandas(desc="Finding Dates Where Loans Are Within 12 Months Of Distress.")
    MERGED["Distress Date"] = MERGED.progress_apply(find_first_match, axis=1)
    MERGED = MERGED.merge(D, on=["Loan Sequence Number", "Distress Date"], how='left')
    MERGED = MERGED[MERGED['Credit Score'] < 998]
    

    print("Merged with Distress Data!")

    for PerformanceYear in range(2011, 2018 + 1):
        YearDF = MERGED[MERGED['Monthly Reporting Period'] // 100 == PerformanceYear]
        FileName = f'TrainingData/Year{PerformanceYear}/TrainingData{PerformanceYear}Orig{year}.parquet'
        YearDF.to_parquet(FileName, index=False, compression="snappy")

        print(f"Wrote {FileName}!")









