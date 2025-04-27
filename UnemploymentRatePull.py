import pandas as pd

# Step 1: Read ladata file
data_df = pd.read_csv('ladata60.txt', sep='\s+', engine='python', header=None)
data_df.columns = ['series_id', 'year', 'period', 'value', 'footnote_codes']

# Step 2: Filter to series ending in '3' (unemployment rates only)
data_df = data_df[data_df['series_id'].str.endswith('3')]

# Step 3: Extract MSA FIPS from series_id (positions 9-13, Python 0-indexed)
data_df['MSA_FIPS_Code'] = data_df['series_id'].str[7:12]

# Step 4: Keep only monthly periods
monthly_data_df = data_df[data_df['period'].str.startswith('M')].copy()

# Step 5: Build YearMonth column
monthly_data_df['month'] = monthly_data_df['period'].str[1:].astype(int)
monthly_data_df['year'] = monthly_data_df['year'].astype(int)
monthly_data_df['YearMonth'] = monthly_data_df['year'] * 100 + monthly_data_df['month']

# Step 6: Final output
final_df = monthly_data_df[['MSA_FIPS_Code', 'YearMonth', 'value']]
final_df = final_df.rename(columns={'value': 'UnemploymentRate'})

print(final_df[final_df['MSA_FIPS_Code'] == '24140'])


# Step 7: Save or view
print(final_df.head())
final_df.to_csv('msa_unemployment_rates_clean.csv', index=False)
