#%%
import pypsa
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

# %%
file_dir = Path(__file__).parent.parent.parent.parent.resolve()
network_2024_path = f"{file_dir}/data/network_files_sarah3_rc/hindcast-dyn-rolling-wy2024.nc"
n= pypsa.Network(network_2024_path)

# %%
prices = n.statistics.prices(groupby_time=False).T
prices
# %%
# find max price
max_price = prices.max().max()
print("Max price:", max_price)
# %%

threshold = 1000
row_caps = prices.mask(prices >= threshold).max(axis=1)
df_prices = prices.where(prices < threshold, row_caps, axis=0)
df_prices
# %%
# Resample to weekly and take mean
weekly_prices = df_prices.resample("W").mean()
weekly_prices
# Choose only DE column
weekly_prices_de = weekly_prices["DE"]
weekly_prices_de.plot(figsize=(10, 5))
plt.title("Weekly Average Prices for DE")
plt.xlabel("Time")
plt.ylabel("Price (EUR/MWh)")
plt.grid()
plt.tight_layout()
plt.show()
# %%
# Check highest price in df_prices
max_price = df_prices.max().max()
print("Max price in df_prices:", max_price)

# %%

years = range(2020, 2025)

yearly_totals = {}
load_shedding_hours = []

for year in years:
    
    print(f"Processing {year}...")
    
    # Load network
    network_path = f"{file_dir}/data/network_files_sarah3_rc/hindcast-dyn-rolling-wy{year}.nc"
    n = pypsa.Network(network_path)
    
    # Get generation statistics
    generation = n.statistics.supply(groupby=["country", "carrier"], groupby_time=False)
    generation = generation.T
    
    # Extract load shedding
    try:
        df_ls = generation.xs('Load shedding', level='carrier', axis=1)
        df_ls = df_ls.dropna(axis=1, how="all")
    except KeyError:
        # No load shedding carrier in this year
        yearly_totals[year] = 0
        continue
    
    # ---- Total load shed for the year ----
    yearly_total = df_ls.sum().sum()
    yearly_totals[year] = yearly_total
    
    # ---- Collect hours with any load shedding ----
    # Identify rows where not all values are NaN
    mask = df_ls.notna().any(axis=1)
    hours_with_ls = df_ls.index[mask]
    
    # Store timestamps with year info
    load_shedding_hours.extend(hours_with_ls)

# Convert hours list to dataframe
df_hours_with_ls = pd.DataFrame(index=pd.DatetimeIndex(load_shedding_hours).unique().sort_values())
df_hours_with_ls["load_shedding_occurred"] = True

# ---- Print yearly totals ----
print("\nTotal Load Shedding per Year:")
for year, total in yearly_totals.items():
    print(f"{year}: {total:,.2f} MWh")
# %%
# Plot germany
df_ls_de = df_ls.xs('DE', level='country', axis=1)
df_ls_de.plot(figsize=(10, 5))
plt.title("Load Shedding in DE")
plt.xlabel("Time")
plt.ylabel("Load Shedding (MW)")
plt.grid()
plt.tight_layout()
plt.show()  
# %%
cap = n.statistics.installed_capacity()

# Select only Generator component
gen_cap = cap.loc["Generator"]

# Define renewables
renewables = [
    "Solar",
    "Offshore Wind (AC)",
    "Onshore Wind",
    "Run of River",
    "Load shedding",
    # "biomass",
    # "geothermal"
]
#%%

years = range(2020, 2025)

ls_list = []
prices_list = []
load_list = []
all_countries = set()

for year in years:
    print(f"Processing {year}...")
    
    # Load network
    network_path = f"{file_dir}/data/network_files_sarah3_rc/hindcast-dyn-rolling-wy{year}.nc"
    n = pypsa.Network(network_path)
    
    # Get generation statistics
    generation = n.statistics.supply(groupby=["country", "carrier"], groupby_time=False)
    generation = generation.T
    
    # Extract load shedding
    try:
        df_ls = generation.xs('Load shedding', level='carrier', axis=1)
        df_ls = df_ls.dropna(axis=1, how="all")
    except KeyError:
        # No load shedding carrier in this year -> skip
        continue

    # Simplify columns to countries
    if isinstance(df_ls.columns, pd.MultiIndex):
        if 'country' in df_ls.columns.names:
            ls = df_ls.copy()
            ls.columns = ls.columns.get_level_values('country')
        else:
            ls = df_ls.copy()
            ls.columns = ls.columns.get_level_values(-1)
    else:
        ls = df_ls

    prices = n.statistics.prices(groupby_time=False).T
    load = n.loads_t.p

    # Track countries present in this year
    all_countries.update(ls.columns)

    # Store for later concatenation
    ls_list.append(ls)
    prices_list.append(prices)
    load_list.append(load)

# Make sure all yearly dfs have same columns (fill missing with 0 or NaN)
all_countries = sorted(all_countries)

ls_aligned = []
for df in ls_list:
    # reindex columns to full set of countries; fill missing with 
    df_aligned = df.reindex(columns=all_countries, fill_value=0)
    ls_aligned.append(df_aligned)

prices_filtered = []
for df in prices_list:
    df_aligned = df.reindex(columns=all_countries, fill_value=np.nan)
    prices_filtered.append(df_aligned)

load_filtered = []
for df in load_list:
    df_aligned = df.reindex(columns=all_countries, fill_value=np.nan)
    load_filtered.append(df_aligned)

# Concatenate along time (index is snapshots)
ls = pd.concat(ls_aligned).sort_index()
# Replace all NaN with 0 (if any remain)
ls = ls.fillna(0)

prices = pd.concat(prices_filtered).sort_index()

load = pd.concat(load_filtered).sort_index()

# ls now has:
# - index: full datetime index across all years
# - columns: all countries, consistent across years

# %%


# %%
# Plot all columns (countries) together
ls.plot(figsize=(12, 6))
plt.title("Load Shedding Over Time (All Countries)")
plt.xlabel("Time")
plt.ylabel("Load Shedding (MW)")
plt.grid()
plt.legend(title="Country")
plt.tight_layout()
plt.show()
# %%
# Now all countries except Norway
countries_to_plot = [col for col in ls.columns if col != 'NO']
ls[countries_to_plot].plot(figsize=(12, 6))
plt.title("Load Shedding Over Time (Excluding Norway)")
plt.xlabel("Time")
plt.ylabel("Load Shedding (MW)")
plt.grid()
plt.legend(title="Country")
plt.tight_layout()
plt.show()

# %%
# Now only plot countries with less than 0.00003 (max)
threshold = 0.00002
countries_below_threshold = [col for col in ls.columns if ls[col].max() < threshold]
ls[countries_below_threshold].plot(figsize=(12, 6))
plt.title("Load Shedding Over Time (Countries Below Threshold)")
plt.xlabel("Time")
plt.ylabel("Load Shedding (MW)")
plt.grid()  
# %%
threshold = 1000  # EUR/MWh

for country in ls.columns:
    df_ls_country = ls[country]
    prices_country = prices[country]

    # Skip countries that never exceed the price threshold
    if not (prices_country > threshold).any():
        continue

    if df_ls_country.empty:
        continue  # skip countries without LS (if relevant)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_ls = 'tab:red'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Load Shedding (MW)', color=color_ls)
    ax1.plot(df_ls_country.index, df_ls_country.values,
             color=color_ls, label=f'Load Shedding ({country})')
    ax1.tick_params(axis='y', labelcolor=color_ls)
    ax1.grid()

    ax2 = ax1.twinx()
    color_price = 'tab:blue'
    ax2.set_ylabel('Price (EUR/MWh)', color=color_price)
    ax2.plot(prices_country.index, prices_country.values,
             color=color_price, label=f'Price ({country})')
    ax2.tick_params(axis='y', labelcolor=color_price)

    plt.title(f"Load Shedding and Price in {country} (Only Countries with Price > {threshold} EUR/MWh at any time)")
    fig.tight_layout()
    plt.show()

# %%
# Calculate the ratio of load shedding to total load for Norway
if 'NO' in ls.columns and 'NO' in load.columns:
    ratio_no = ls['NO'] / load['NO']
    ratio_no.plot(figsize=(12, 6))
    plt.title("Ratio of Load Shedding to Total Load in Norway")
    plt.xlabel("Time")
    plt.ylabel("Load Shedding / Total Load")
    plt.grid()
    plt.tight_layout()
    plt.show()
# %%


# 
# %%
