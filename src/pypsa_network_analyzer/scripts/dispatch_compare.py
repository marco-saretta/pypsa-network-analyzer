# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:31:01 2025

@author: FEGU
"""
import pandas as pd
import matplotlib.pyplot as plt

start_date = '2022-01-01 00:00:00'
end_date = '2022-12-31 23:00:00'

country_code = 'ES'  

#%%% Read indivudial country
#file_name = f'generation_{country_code}_hourly_data.csv'
file_path = rf"data\generation\generation_hourly_data\generation_{country_code}_hourly_data.csv"

# read csv

generation = pd.read_csv(
    file_path,  # replace with your file path
    quotechar='"',             # handle quoted strings
    sep=",",                   # standard comma separator
)


# Convert Date to datetime and set it as index
generation.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
generation["Date"] = pd.to_datetime(generation["Date"])
generation.set_index("Date", inplace=True)

# --- Filter by date range ---
mask = (generation.index >= start_date) & (generation.index <= end_date)
generation_filtered = generation.loc[mask]

# If the index is timezone-aware, remove the timezone
if generation_filtered.index.tz is not None:
    generation_filtered.index = generation_filtered.index.tz_convert(None)


#%%% Load generation from pypsa WY2022


file_path_pypsa = r"results\hindcast_dynamic_wy_2022\summary_results\generators_dispatch.csv"

generation_pypsa = pd.read_csv(
    file_path_pypsa,  # replace with your file path
    quotechar='"',             # handle quoted strings
    sep=",",                   # standard comma separator
)

generation_pypsa.set_index('snapshot', inplace=True)


filtered_df = generation_pypsa[
    [col for col in generation_pypsa.columns
     if col.startswith(f"{country_code} ") and "load" not in col.lower()]
]

# Extract the part after the first space in each column name
technologies = generation_pypsa.columns.str.split(" ", n=1).str[1]

# Get unique technologies
unique_technologies = technologies.dropna().unique()


# your mapping
mapping = {
    'biomass': 'Biomass',
    'CCGT': 'Fossil Gas',
    'OCGT': 'Fossil Gas',
    'coal': 'Fossil Hard coal',
    'lignite': 'Fossil Hard coal',
    'oil': 'Fossil Oil',
    'geothermal': 'Geothermal',
    'ror': 'Hydro Run-of-river and poundage',
    '0 offwind-ac': 'Wind Offshore',
    '0 offwind-dc': 'Wind Offshore',
    '0 onwind': 'Wind Onshore',
    '0 solar': 'Solar',
    '0 solar-hsat': 'Solar',
    'load': None  # ignore
}

# make mapping lowercase for case-insensitive match
mapping_lower = {k.lower(): v for k, v in mapping.items()}

# 1) remove country code
raw_tech = filtered_df.columns.str.split(" ", n=1).str[1].str.strip()

# 2) map to standardized names
mapped = raw_tech.str.lower().map(mapping_lower)

# 3) keep only columns that mapped successfully (not None/NaN)
keep_mask = mapped.notna()
df_kept = filtered_df.loc[:, keep_mask]

# 4) rename columns using mapped names
df_kept.columns = mapped[keep_mask].values  # <- key fix: use .values so lengths match

# 5) combine columns with the same name by summing
filtered_df_pypsa = df_kept.groupby(df_kept.columns, axis=1).sum()


# Filter the common columns
generation_filtered_common = generation_filtered[filtered_df_pypsa.columns]


# Plot

# Make copies to avoid modifying originals
df1_plot = generation_filtered_common.copy()
df2_plot = filtered_df_pypsa.copy()

# Ensure index is DatetimeIndex
df1_plot.index = pd.to_datetime(df1_plot.index)
df2_plot.index = pd.to_datetime(df2_plot.index)

# Remove timezone if present
if df1_plot.index.tz is not None:
    df1_plot.index = df1_plot.index.tz_convert(None)
if df2_plot.index.tz is not None:
    df2_plot.index = df2_plot.index.tz_convert(None)

# Convert to Python datetime objects for matplotlib
df1_plot.index = df1_plot.index.to_pydatetime()
df2_plot.index = df2_plot.index.to_pydatetime()



common_cols = df1_plot.columns

fig, axes = plt.subplots(len(common_cols), 1, figsize=(12, 4 * len(common_cols)), sharex=True)
if len(common_cols) == 1:
    axes = [axes]

for ax, col in zip(axes, common_cols):
    ax.plot(df1_plot.index, df1_plot[col], label='generation_filtered')
    ax.plot(df2_plot.index, df2_plot[col], label='filtered_df_pypsa', linestyle='--')
    ax.set_title(col)
    ax.set_ylabel('Power [MW]')
    ax.legend()

plt.xlabel('Snapshot')
plt.tight_layout()
plt.show()