import pandas as pd

def ReadPerformanceTXT(year):
    performance_cols = [
        "Loan Sequence Number", "Monthly Reporting Period", "Current Actual UPB", "Current Loan Delinquency Status",
        "Loan Age", "Remaining Months to Legal Maturity", "Defect Settlement Date", "Modification Flag", "Zero Balance Code", 
        "Zero Balance Effective Date", "Current Interest Rate", "Current Deferred UPB", "Due Date of Last Paid Installment (DDLPI)",
        "MI Recoveries", "Net Sales Proceeds", "Non MI Recoveries", "Expenses", "Legal Costs", "Maintenance and Preservation Costs",
        "Taxes and Insurance", "Miscellaneous Expenses", "Actual Loss Calculation", "Modification Cost", "Step Modification Flag",
        "Deferred Payment Plan", "Estimated Loan-to-Value (ELTV)", "Zero Balance Removal UPB", "Delinquent Accrued Interest",
        "Delinquency Due to Disaster", "Borrower Assistance Status Code", "Current Month Modification Cost", "Interest Bearing UPB"
        ]
      
    df = pd.read_csv(f'PerformanceSample/sample_svcg_{year}.txt', sep='|', header=None)
    df.columns = performance_cols
    return df

def ReadOriginationTXT(year):

    origination_cols = ["Credit Score", "First Payment Date", "First Time Homebuyer Flag", "Maturity Date", 
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
    df.columns = origination_cols
    return df
