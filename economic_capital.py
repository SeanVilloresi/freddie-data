import pandas as pd

year = 2007

Distress = pd.read_parquet(f"ValPredictions/calibrated_predictions_{year}.parquet")
DefaultGivenDistress = pd.read_parquet(f"ValPredictions/DefGivenStress{year}.parquet")
LossGivenDefault = pd.read_parquet(f"ValPredictions/lgd_predictions_{year}.parquet")

print(Distress.columns)
print(DefaultGivenDistress.columns)
print(LossGivenDefault.columns)