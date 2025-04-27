import pandas as pd

# Step 1: Read the state unemployment data
state_df = pd.read_csv('allstates.txt', sep='\s+', engine='python', header=None)
state_df.columns = ['series_id', 'year', 'period', 'value', 'footnote_codes']

# Step 2: Filter to unemployment rates only (series_id ending with '3')
state_df = state_df[state_df['series_id'].str.endswith('3')]

# Step 3: Extract State FIPS from series_id
state_df['State_FIPS_Code'] = state_df['series_id'].str[5:7]

# Step 4: Keep only monthly periods (M01 to M12)
state_df = state_df[state_df['period'].str.startswith('M')].copy()
state_df['month'] = state_df['period'].str[1:].astype(int)
state_df['year'] = state_df['year'].astype(int)
state_df['YearMonth'] = state_df['year'] * 100 + state_df['month']

# Step 5: Create State FIPS → State Abbreviation mapping
state_fips_to_abbrev = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO',
    '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI',
    '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA', '20': 'KS', '21': 'KY',
    '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN',
    '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
    '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH',
    '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC', '46': 'SD',
    '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA',
    '54': 'WV', '55': 'WI', '56': 'WY'
}

# Step 6: Map FIPS to State Abbreviation
state_df['State'] = state_df['State_FIPS_Code'].map(state_fips_to_abbrev)

# Step 7: Final selection
final_state_unemp = state_df[['State', 'YearMonth', 'value']]
final_state_unemp = final_state_unemp.rename(columns={'value': 'UnemploymentRate'})

# Step 8: Output
print(final_state_unemp.head())

# Optional: Save to file
final_state_unemp.to_csv('state_unemployment_rates_clean.csv', index=False)
