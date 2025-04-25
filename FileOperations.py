import pandas as pd

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

    Ocols = ["Credit Score", "First Payment Date", "First Time Homebuyer Flag", "Maturity Date", 
                        "Metropolitan Statistical Area (MSA) Or Metropolitan Division", "Mortgage Insurance Percentage (MI %)",
                        "Number of Units", "Occupancy Status", "Original Combined Loan-to-Value (CLTV)",
                        "Original Debt-to-Income (DTI) Ratio", "Original UPB", "Original Loan-to-Value (LTV)", 
                        "Original Interest Rate", "Channel", "Prepayment Penalty Mortgage (PPM) Flag", 
                        "Amortization Type (Formerly Product Type)", "Property State", "Property Type", "Postal Code", 
                        "Loan Sequence Number", "Loan Purpose", "Original Loan Term", "Number of Borrowers", "Seller Name", 
                        "Servicer Name", "Super Conforming Flag", "Pre-HARP Loan Sequence Number", "Program Indicator", 
                        "HARP Indicator", "Property Valuation Method", "Interest Only (I/O) Indicator", 
                        "Mortgage Insurance Cancellation Indicator"]
    
    df = pd.read_csv(f'OriginationSample/sample_orig_{year}.txt', sep='|', header=None)
    df.columns = Ocols
    return df

def WriteDistressDataset(year):
    P = ReadPTXT(year)

    # Create Stress Flags
    P.loc[(~P["Current Loan Delinquency Status"].isin(('0','1','2','3','4','5'))),"Major Stress"] = 1

    # Create ZB Flags
    P.loc[(P["Zero Balance Code"].isin((2.0, 3.0, 9.0))),"Zero Balance Default"] = 1
    ZB1 = set(P[P["Zero Balance Code"] == 1.0]["Loan Sequence Number"].dropna().unique())
    P.loc[(P["Loan Sequence Number"].isin(ZB1)),"Ends In Payoff"] = 1

    # Find Invalid Defaults
    MostRecentCurrency = (P[P["Current Loan Delinquency Status"] == '0'].groupby("Loan Sequence Number")["Monthly Reporting Period"]
                          .max().reset_index().rename(columns={"Monthly Reporting Period": "Most Recent Currency"}))
    P = P.merge(MostRecentCurrency, on="Loan Sequence Number", how="left")
    UnresolvedDelinquency = set(P[(P["Monthly Reporting Period"] == 202409) & (P["Current Loan Delinquency Status"] != "0")]["Loan Sequence Number"].dropna().unique())
    P.loc[(P["Loan Sequence Number"].isin(UnresolvedDelinquency)),"Unresolved Delinquency"] = 1
    P.loc[(P["Ends In Payoff"] == 1) | (P["Monthly Reporting Period"] < P["Most Recent Currency"]) | (P["Unresolved Delinquency"] == 1), "Invalid Default"] = 1

    # Get Default Date for Defautled Mortgages
    P.loc[((P["Major Stress"]==1) | (P["Zero Balance Default"]==1)) & (~(P["Invalid Default"] == 1)), "Default Flag"] = 1
    
    DefaultDate = P[P["Default Flag"]==1][["Loan Sequence Number","Monthly Reporting Period"]]
    DefaultDate = DefaultDate.sort_values(by=['Loan Sequence Number', 'Monthly Reporting Period'])
    DefaultDate = DefaultDate.rename(columns={'Monthly Reporting Period':'Default Date'})
    DefaultDate = DefaultDate.groupby("Loan Sequence Number").first().reset_index()

    # Create Distress Dataset
    P = P.merge(DefaultDate, on="Loan Sequence Number", how="left")
    Distress = P[(P["Major Stress"] == 1) | (P["Zero Balance Default"] == 1)]
    DistressColumns = ["Loan Sequence Number", "Monthly Reporting Period", "Major Stress", "Default Flag",
                       "Zero Balance Code", "Unresolved Delinquency", "Actual Loss Calculation", "Zero Balance Removal UPB", 
                       "Net Sales Proceeds", "Delinquent Accrued Interest", "Expenses", "MI Recoveries", "Non MI Recoveries", "Most Recent Currency"]
    
    Distress[DistressColumns].to_parquet(f"DefaultData/Defaults{year}.parquet", engine="pyarrow", index=False)


WriteDistressDataset(2011)






