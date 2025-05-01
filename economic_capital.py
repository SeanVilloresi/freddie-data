import pandas as pd
from pandasgui import show
import numpy as np
from scipy.stats import norm

year = 2023

Distress = pd.read_parquet(f"ValPredictions/calibrated_predictions_{year}.parquet")
DefaultGivenDistress = pd.read_parquet(f"ValPredictions/DefGivenStress{year}.parquet")
LossGivenDefault = pd.read_parquet(f"ValPredictions/lgd_predictions_{year}.parquet")

#Renaming
Distress = Distress.rename(columns={
    "y_true": "Actual Stress"
})


DefaultGivenDistress = DefaultGivenDistress.rename(columns={
    "y_true": "Actual Default",
    "y_prob": "P(Default)"
})

# merge that shit
merged_df = (
    Distress
    .merge(
        DefaultGivenDistress,
        on=["Loan Sequence Number", "Monthly Reporting Period"],
        how="left"
    )
    .merge(
        LossGivenDefault,
        on=["Loan Sequence Number", "Monthly Reporting Period"],
        how="left"
    )
)

# Remove Loans with unknown outcomes or loss
mask_stress = ~((merged_df["Major Stress"] == 1) & (merged_df["Unresolved"] == 1))
mask_default_loss = ~((merged_df["Default Flag"] == 1) & 
                      ((merged_df["Actual Loss Calculation"].isna()) | 
                       (merged_df["Actual Loss Calculation"] == 0)))

merged_df = merged_df[mask_stress & mask_default_loss].reset_index(drop=True).drop(columns=['Actual Loss Calculation', 'Unresolved', 'Major Stress', 'Default Flag'])

def compute_economic_capital(S, D, LGD, alpha=0.999, rho=0.15):
    """
    Compute per‐loan economic capital K_i(α) according to:
    
        K_i(α) = LGD_i * ( Φ( (Φ⁻¹(S_i * D_i) + √ρ·Φ⁻¹(α)) / √(1−ρ) ) − S_i·D_i )
    
    where
      S = P(Major Stress),
      D = P(Default | Major Stress),
      LGD = Loss‐Given‐Default,
      α = confidence level (e.g. 0.999),
      ρ = asset correlation (e.g. 0.15).
    
    Parameters
    ----------
    S     : array‐like of shape (n,)
    D     : array‐like of shape (n,)
    LGD   : array‐like of shape (n,)
    alpha : float, tail probability (e.g. 0.999)
    rho   : float, asset‐correlation (e.g. 0.15)
    
    Returns
    -------
    K     : np.ndarray of shape (n,), economic capital proportions
    """
    S = np.asarray(S, dtype=float)
    D = np.asarray(D, dtype=float)
    LGD = np.asarray(LGD, dtype=float)
    
    SD = S * D
    # inverse CDF of SD
    inv_sd = norm.ppf(SD)
    # inverse CDF of alpha
    inv_alpha = norm.ppf(alpha)
  
    arg = (inv_sd + np.sqrt(rho) * inv_alpha) / np.sqrt(1 - rho)
    # compute K
    K = (norm.cdf(arg) - SD)
    return K, LGD * K 


merged_df['K'], merged_df['Economic Capital']  = compute_economic_capital(
    S=merged_df['Calibrated P(Stress)'],
    D=merged_df['P(Default)'],
    LGD=merged_df['Predicted LGD'],
    alpha=0.99,
    rho=0.15
)

#show(merged_df)

print("True Loss: ", sum(merged_df['True LGD'].fillna(0)))
print("Capital on Hand: ", sum(merged_df['Economic Capital']))




