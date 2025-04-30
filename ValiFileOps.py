import pandas as pd
import numpy as np
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
      
    df = pd.read_csv(f'PVal/historical_data_time_{year}Q1.txt', sep='|', header=None, 
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
    
    df = pd.read_csv(f'historical_data_{year}Q1.txt ', sep='|', header=None, dtype={i : str for i in [27]})
    df.columns = Ocols
    df = df.rename(columns={"Metropolitan Statistical Area (MSA) Or Metropolitan Division" : "MSA"})

    return df

def WriteDistressDataset(year):
    P = ReadPTXT(year)
    P = P.rename(columns={'Monthly Reporting Period':'Distress Date'})

    # Create Flags for Stress and Default
    MajorStress = (~P["Current Loan Delinquency Status"].isin(('0', '1', '2', '3', '4', '5'))) | (P["Zero Balance Code"].isin((2.0, 3.0, 9.0)))
    
    EndsInPayoff = (P["Loan Sequence Number"].isin(set(P[P["Zero Balance Code"] == 1.0]["Loan Sequence Number"].unique())))

    MostRecentCurrency = P["Loan Sequence Number"].map(P[P["Current Loan Delinquency Status"] == '0'].groupby("Loan Sequence Number")["Distress Date"].max())

    MostRecentCurrency = MostRecentCurrency.fillna(-1)

    Recovered = (P["Distress Date"] < MostRecentCurrency)

    Unresolved = (P["Loan Sequence Number"].isin(set(P[(P["Distress Date"] == 202409)]["Loan Sequence Number"].unique())))

    InvalidDefault = (EndsInPayoff | Recovered | Unresolved)

    DefaultFlag = (MajorStress & (~InvalidDefault))

    # Assign Response Variables
    P.loc[:, "Major Stress"] = MajorStress.astype(int)
    P.loc[:, "Default Flag"] = DefaultFlag.astype(int)
    P.loc[:, "Unresolved"] = Unresolved.astype(int)
    P.loc[:, "Last Time Current"] = MostRecentCurrency


    # Get Loss Data
    LossData = P[P["Default Flag"]==1][["Loan Sequence Number","Distress Date", "Default Flag",
                                        "Actual Loss Calculation", "Zero Balance Removal UPB", "Net Sales Proceeds", 
                                        "Delinquent Accrued Interest", "Expenses", "MI Recoveries", "Non MI Recoveries"]]
    LossData = LossData.sort_values(by=['Loan Sequence Number', 'Distress Date'])
    LossData = LossData.groupby("Loan Sequence Number").first().reset_index()
    LossData = LossData.drop(columns=["Distress Date"])

    # Create Distress Dataset
    D = P[(P["Major Stress"] == 1)]
    D = D.drop(columns=["Actual Loss Calculation", "Zero Balance Removal UPB", "Net Sales Proceeds", "Delinquent Accrued Interest", 
                        "Expenses", "MI Recoveries", "Non MI Recoveries"])
    
    D = D.merge(LossData, on=["Loan Sequence Number", "Default Flag"], how="left")
    
    DistressColumns = ["Loan Sequence Number", "Distress Date", "Major Stress", "Default Flag",
                       "Zero Balance Code", "Unresolved", "Last Time Current", 
                       "Actual Loss Calculation", "Zero Balance Removal UPB", "Net Sales Proceeds", "Delinquent Accrued Interest", 
                       "Expenses", "MI Recoveries", "Non MI Recoveries"
                       ]
    
    D[DistressColumns].to_parquet(f"DefaultData/Defaults{year}.parquet", engine="pyarrow", index=False)
    print(f"Wrote DefaultValData/Defaults{year}.parquet!")

def WriteTrainingData(year):
    
    PColsToDrop = [
        "Defect Settlement Date", "Zero Balance Code", "Zero Balance Effective Date", "Due Date of Last Paid Installment (DDLPI)",
        "MI Recoveries", "Net Sales Proceeds", "Non MI Recoveries", "Expenses", "Legal Costs", "Maintenance and Preservation Costs",
        "Taxes and Insurance", "Miscellaneous Expenses", "Actual Loss Calculation", "Modification Cost", 
        "Zero Balance Removal UPB", "Delinquent Accrued Interest", "Current Month Modification Cost", "Step Modification Flag",
        
        ]
    
    OColsToDrop = [
        "Maturity Date", "Channel", "Seller Name", "Servicer Name", "Super Conforming Flag", 
        "Pre-HARP Loan Sequence Number", "HARP Indicator", "Interest Only (I/O) Indicator", "Amortization Type (Formerly Product Type)",
        "Prepayment Penalty Mortgage (PPM) Flag"
        ]
    
    P = ReadPTXT(year).drop(columns = PColsToDrop)
    O = ReadOTXT(year).drop(columns = OColsToDrop)
    D = pd.read_parquet(f"DefaultValData/Defaults{year}.parquet", engine="pyarrow")

    print("Data Loaded!")

    # Maximum Number of Months Each Loan Has Been PReviously Delinquent
    P["Numeric Delinquency"] = pd.to_numeric(P["Current Loan Delinquency Status"], errors="coerce")
    P = P.sort_values(["Loan Sequence Number", "Monthly Reporting Period"])
    P["MaxPriorDelinquency"] = (P.groupby("Loan Sequence Number")["Numeric Delinquency"].expanding().max().shift().fillna(0).reset_index(level=0, drop=True))

    print("Computed Maximum Prior Delinquency!")
    
    P["Modification Flag"] = P["Modification Flag"].notna().astype(int)
    P["Deferred Payment Plan"] = P["Deferred Payment Plan"].notna().astype(int)
    P["Delinquency Due to Disaster"] = P["Delinquency Due to Disaster"].notna().astype(int)

    # Map each loan to its starting month
    FirstMonth = P.groupby('Loan Sequence Number')["Monthly Reporting Period"].min() % 100
    P['Start Month'] = P['Loan Sequence Number'].map(FirstMonth)

    O = O[~O["Property State"].isin(("AS", "GU", "MP", "PR", "VI", "UM"))]
    P = P[P["Current Loan Delinquency Status"].isin(('0','1','2','3','4','5'))]
    P = P[P["Monthly Reporting Period"].between(200001, 200801) | P["Monthly Reporting Period"].between(202212, 202409)]
    P = P[P["Monthly Reporting Period"] % 100 == P['Start Month']]
    P = P.drop(columns=["Start Month"])

    print("Filtered Performance!")

    MERGED = O.merge(P, on="Loan Sequence Number", how="inner")

    print("Merged Origination & Performance!")

    def ExtractMonthFromID(ID):
        # Assumes the string always starts with 'F' + 2 digits + 'Q' + 1 digit
        fiscal_year = int(ID[1:3]) + 2000 
        quarter = int(ID[4])
        return year + (quarter * 3)

    MERGED["Origination Month"] = MERGED["Loan Sequence Number"].apply(lambda ID : ExtractMonthFromID(ID))

    #################################################################################### Macro Merging
    MSA = pd.read_csv("MacroData/MSAMacros.csv")
    State = pd.read_csv("MacroData/StateMacros.csv")

    # Origination Month
    # Use first reporting month as Origination Month
    def shift_back_one_month(ym):
        year = ym // 100
        month = ym % 100
        month -= 1
        if month == 0:
            month = 12
            year -= 1
        return year * 100 + month

    MERGED["First Payment Date"] = MERGED["First Payment Date"].astype(int)  # ensure correct type
    MERGED["Origination Month"] = MERGED["First Payment Date"].apply(shift_back_one_month)
    MERGED["Prior Year Month"] = (MERGED["Monthly Reporting Period"] // 100 - 1) * 100 + (MERGED["Monthly Reporting Period"] % 100)

    # Define merge targets
    macro_merges = [
        ("Current MSA Unemployment Rate", "Current MSA HPI (Seasonally Adjusted)", "MSA", "Monthly Reporting Period", MSA),
        ("MSA Unemployment Rate at Origination", "MSA HPI (Seasonally Adjusted) at Origination", "MSA", "Origination Month", MSA),
        ("Current State Unemployment Rate", "Current State HPI (Seasonally Adjusted)", "Property State", "Monthly Reporting Period", State),
        ("State Unemployment Rate at Origination", "State HPI (Seasonally Adjusted) at Origination", "Property State", "Origination Month", State),
        ("MSA Unemployment Rate 1Y Ago", None, "MSA", "Prior Year Month", MSA),
        ("State Unemployment Rate 1Y Ago", None, "Property State", "Prior Year Month", State)
    ]

    # Perform macro merges
    for ur_col, hpi_col, geo_col, date_col, df_macro in macro_merges:
        cols = [geo_col, "YearMonth", "UnemploymentRate"]
        rename_dict = {"UnemploymentRate": ur_col}
        if hpi_col:
            cols.append("HPI (Seasonally Adjusted)")
            rename_dict["HPI (Seasonally Adjusted)"] = hpi_col
        MERGED = MERGED.merge(
            df_macro[cols].rename(columns=rename_dict),
            left_on=[geo_col, date_col],
            right_on=[geo_col, "YearMonth"],
            how="left"
        ).drop(columns=["YearMonth"])

    # Fill MSA first, fallback to State
    MERGED["Current HPI"] = MERGED["Current MSA HPI (Seasonally Adjusted)"].fillna(MERGED["Current State HPI (Seasonally Adjusted)"])
    MERGED["Origination HPI"] = MERGED["MSA HPI (Seasonally Adjusted) at Origination"].fillna(MERGED["State HPI (Seasonally Adjusted) at Origination"])
    print(MERGED["Current MSA Unemployment Rate"].fillna(MERGED["Current State Unemployment Rate"]).dtype)
    MERGED["Current Unemployment Rate"] = MERGED["Current MSA Unemployment Rate"].fillna(MERGED["Current State Unemployment Rate"]).astype(float)
    MERGED["Unemployment Rate at Origination"] = MERGED["MSA Unemployment Rate at Origination"].fillna(MERGED["State Unemployment Rate at Origination"]).astype(float)
    MERGED["Unemployment Rate 1Y Ago"] = MERGED["MSA Unemployment Rate 1Y Ago"].fillna(MERGED["State Unemployment Rate 1Y Ago"]).astype(float)

    # Final feature engineering
    MERGED["HPI Change %"] = (MERGED["Current HPI"] / MERGED["Origination HPI"]) - 1
    MERGED["Unemployment Rate Change 1Y"] = MERGED["Current Unemployment Rate"] - MERGED["Unemployment Rate 1Y Ago"]

    # Drop temp columns
    MERGED = MERGED.drop(columns=["Origination Month", "Prior Year Month",  "First Payment Date", "Current HPI", "Origination HPI", "Current MSA HPI (Seasonally Adjusted)",
                                  "Current State HPI (Seasonally Adjusted)", "State HPI (Seasonally Adjusted) at Origination", "MSA HPI (Seasonally Adjusted) at Origination",
                                  "State Unemployment Rate 1Y Ago", "Unemployment Rate 1Y Ago", "MSA Unemployment Rate 1Y Ago", "Current State Unemployment Rate",
                                  "Current MSA Unemployment Rate", "State Unemployment Rate at Origination", "MSA Unemployment Rate at Origination"],)
    print("Merged with Macro Data and created HPI % Change and UR Change 1Y!")
    ############################################################################### Macro Merging Ended

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
    print(MERGED.columns)

    for PerformanceYear in [2007, 2008, 2023, 2024]:
        YearDF = MERGED[MERGED['Monthly Reporting Period'] // 100 == PerformanceYear]
        FileName = f'ValData/Year{PerformanceYear}/ValData{PerformanceYear}Orig{year}.parquet'
        YearDF.to_parquet(FileName, index=False, compression="snappy")

        print(f"Wrote {FileName}!")


def WriteMacroFiles():
    MetropolitanUnemployment = pd.read_csv("MacroData/msa_unemployment_rates.csv")
    StateUnemployment = pd.read_csv("MacroData/state_unemployment_rates.csv")

    MetropolitanHPI = pd.read_excel("MacroData/hpi_metro.xls")
    StateHPI = pd.read_excel("MacroData/hpi_state.xls")

    MetropolitanUnemployment = MetropolitanUnemployment.rename(columns={"MSA_FIPS_Code" : "MSA"})
    MetropolitanHPI = MetropolitanHPI.rename(columns={"cbsa" : "MSA"})
    StateUnemployment = StateUnemployment.rename(columns={"State" : 'Property State'})
    StateHPI = StateHPI.rename(columns={"state" : 'Property State'})
    MetropolitanUnemployment = MetropolitanUnemployment.replace('-', np.nan)



    def CleanHPIFile(DF, GeographyName):

        def AddQuarter(row):
            year = row['yr']
            quarter = row['qtr']
    
            if quarter == 4:
                year += 1
                quarter = 1
            else:
                quarter += 1

            return pd.Series({'Adjusted Year': year, 'Adjusted Quarter': quarter})
        
        PrevDF = DF.apply(AddQuarter, axis=1)
        DF = pd.concat([DF, PrevDF], axis=1)
        DF = DF.loc[DF.index.repeat(3)].reset_index(drop=True)


        DF['Month IDX'] = DF.groupby([GeographyName, 'Adjusted Year', 'Adjusted Quarter']).cumcount()
        
        quarter_to_months = {
            1: [1, 2, 3],
            2: [4, 5, 6],
            3: [7, 8, 9],
            4: [10, 11, 12]
        }

        DF['Month'] = DF.apply(lambda row: quarter_to_months[row['Adjusted Quarter']][row['Month IDX']], axis=1)

        DF = DF.drop(columns=['Month IDX'])  
        DF['YearMonth'] = DF.apply(lambda row : row["Adjusted Year"] * 100 + row["Month"], axis = 1)

        DF = DF.rename(columns = {"index_sa" : "HPI (Seasonally Adjusted)"})

        return DF[[GeographyName, "YearMonth", "HPI (Seasonally Adjusted)"]]


    MetropolitanHPI = CleanHPIFile(MetropolitanHPI, "MSA")
    StateHPI = CleanHPIFile(StateHPI, "Property State")

    MetropolitanUnemployment = MetropolitanUnemployment[MetropolitanUnemployment["YearMonth"] > 199900]
    MetropolitanUnemployment = MetropolitanUnemployment[~(MetropolitanUnemployment["YearMonth"] % 100 == 13)]
    StateUnemployment = StateUnemployment[StateUnemployment["YearMonth"] > 199900]

    MetropolitanUnemployment.merge(MetropolitanHPI, on = ["MSA", "YearMonth"], how = "left").to_csv("MacroData/MSAMacros.csv", index=False)
    StateUnemployment.merge(StateHPI, on = ["Property State", "YearMonth"], how = "inner").to_csv("MacroData/StateMacros.csv", index=False)


for year in range(start, end+1):
    WriteDistressDataset(year)
    WriteTrainingData(year)

#WriteMacroFiles()







