#!/usr/bin/env python3
"""
Generate example parameter values for low, medium, and high emission levels.
"""

import pandas as pd
import numpy as np

data_path = 'data/tourism_5000_rows.csv'
df = pd.read_csv(data_path)

total_emissions = df['total_trip_emissions_kgCO2']
percentile_33 = total_emissions.quantile(0.33)
percentile_67 = total_emissions.quantile(0.67)

def classify_emission(value):
    if value <= percentile_33:
        return 'low'
    elif value <= percentile_67:
        return 'medium'
    else:
        return 'high'

df['emission_level'] = df['total_trip_emissions_kgCO2'].apply(classify_emission)

print('=' * 80)
print('EMISSION LEVEL PARAMETER VALUES')
print('=' * 80)

key_params = [
    'trip_days', 'distance_km', 'occupancy', 'congestion_factor', 
    'terrain_factor', 'nightly_kwh_est', 'grid_ef_kgCO2_per_kWh', 
    'diesel_gen_l_per_night', 'season', 'transport_mode', 'hotel_class'
]

for level in ['low', 'medium', 'high']:
    subset = df[df['emission_level'] == level]
    
    # Get median value
    median_val = subset['total_trip_emissions_kgCO2'].median()
    median_idx = (subset['total_trip_emissions_kgCO2'] - median_val).abs().idxmin()
    median_row = df.loc[median_idx]
    
    print(f'\n{level.upper()} EMISSION')
    print(f'Total Emissions: {median_row["total_trip_emissions_kgCO2"]:.2f} kgCO2')
    print('-' * 80)
    
    for param in key_params:
        if param in df.columns:
            val = median_row[param]
            if isinstance(val, (int, np.integer)):
                formatted = str(int(val))
            elif isinstance(val, float):
                formatted = f'{val:.2f}'
            else:
                formatted = str(val)
            print(f'  {param:.<35} {formatted}')

# Also show parameter ranges
print('\n\n' + '=' * 80)
print('PARAMETER RANGES BY EMISSION LEVEL')
print('=' * 80)

numeric_params = ['trip_days', 'distance_km', 'occupancy', 'nightly_kwh_est', 
                  'grid_ef_kgCO2_per_kWh', 'diesel_gen_l_per_night']

for level in ['low', 'medium', 'high']:
    subset = df[df['emission_level'] == level]
    print(f'\n{level.upper()} EMISSION:')
    print('-' * 80)
    
    for param in numeric_params:
        if param in df.columns:
            min_val = subset[param].min()
            max_val = subset[param].max()
            mean_val = subset[param].mean()
            print(f'  {param:.<35} Min: {min_val:7.2f}  |  Max: {max_val:7.2f}  |  Mean: {mean_val:7.2f}')
