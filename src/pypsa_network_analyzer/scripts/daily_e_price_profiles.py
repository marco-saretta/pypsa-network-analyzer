# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 09:26:16 2025

@author: FEGU
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 09:56:29 2025

@author: FEGU
"""
#%%
import pypsa
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
#%%


# Fonts for plot
title_font = 24
other_font = 18
ticks_font = 14


'''   Entso-e data   '''
# Load file
file_dir = Path(__file__).parent.parent.resolve()
external_file = f"{file_dir}/data/prices/electricity_prices.csv"
e_prices = pd.read_csv(external_file)
e_prices.rename(columns={"Unnamed: 0": "snapshot"}, inplace=True)
e_prices.set_index("snapshot", inplace=True)
# After reading CSV
e_prices.index = pd.to_datetime(e_prices.index)
e_prices.index = e_prices.index.tz_localize(None)


#%%





'''   Model data   '''
#simulation = "hindcast-dyn-spec-cap"
#simulation = "hindcast-dyn"
#simulation = "hindcast_dyn_old"
#simulation = "hindcast-std"
#simulation="hindcast-dyn-2022"
#simulation="hindcast-dyn-spec-cap"
#simulation="hindcast-dyn-spec-cap-rolling" 
#simulation = "hindcast-dyn-rolling"
simulation="hindcast-dyn-spec-cap-irena"


external_file = f"{file_dir}/simulations/{simulation}/results_concat/combined_electricity_prices.csv"
e_prices_pypsa = pd.read_csv(external_file)
e_prices_pypsa.set_index("snapshot", inplace=True)
e_prices_pypsa.index = pd.to_datetime(e_prices_pypsa.index)



#%%


'''   Input for run and plot   '''
# Year, time and country to plot
years = [2020, 2021, 2022, 2023, 2024]


countries = ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI',
       'FR', 'GB', 'GBI', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'ME',
       'MK', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SE', 'SI', 'SK', 'XK', 'EUR', 'DE_LU']


time_filter_dict = ["year", "winter", "spring", "summer", "fall"]




''' Calculate load average data for countries with multiple bidding zone '''
external_file_load = f"{file_dir}/data/load/demand.csv"
load = pd.read_csv(external_file_load)
load.rename(columns={"Unnamed: 0": "snapshot"}, inplace=True)
load.set_index("snapshot", inplace=True)
# After reading CSV
load.index = pd.to_datetime(load.index)
load.index = load.index.tz_localize(None)


#%%
# Read in the load data from pypsa model results for comparison for the years 2020-2024
# We need to concat 4 files here for each year
load_list = []
for year in years:
    external_file_load_pypsa = f"{file_dir}/simulations/{simulation}/weather_year_{year}/results/summary/electric_load.csv"
    load_year = pd.read_csv(external_file_load_pypsa)
    load_year.set_index("snapshot", inplace=True)
    load_year.index = pd.to_datetime(load_year.index)
    load_list.append(load_year)
load_pypsa = pd.concat(load_list)
load_pypsa

#%%
# filter load and e_prices to only include timestamps that are present in e_prices_pypsa  
index = e_prices_pypsa.index
load = load.loc[index]
e_prices = e_prices.loc[index]



#%%

# =========================
# Country/region to bidding zones mapping
# =========================
country_mapping = {
    "DK": ["DK_1", "DK_2"],
    "SE": ["SE_1", "SE_2", "SE_3", "SE_4"],
    "NO": ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5"],
    "IT": ["IT_CNOR", "IT_CSUD", "IT_NORD", "IT_SARD", "IT_SICI"],
    "EUR": [
        "AT", "BE", "BG", "CH", "CZ", "DE_LU", "DK_1", "DK_2", "EE", "ES", "FI",
        "FR", "GR", "HR", "HU", "IE_SEM", "IT_CNOR", "IT_CSUD", "IT_NORD",
        "IT_SARD", "IT_SICI", "LV", "LT", "MK", "NL", "NO_1", "NO_2", "NO_3",
        "NO_4", "NO_5", "PL", "PT", "RO", "RS", "SE_1", "SE_2", "SE_3", "SE_4",
        "SI", "SK", "UA_IPS"
    ],
}

# =========================
# Load-weighted price function
# =========================
def load_weighted_price(price_df, load_df, bidding_zones):
    """
    Compute load-weighted electricity price for a list of bidding zones.
    
    Parameters
    ----------
    price_df : pd.DataFrame
        Zone-level electricity prices (columns = bidding zones)
    load_df : pd.DataFrame
        Zone-level electricity load (columns = bidding zones)
    bidding_zones : list of str
        Zones to aggregate

    Returns
    -------
    pd.Series
        Load-weighted price
    """
    prices = price_df[bidding_zones]
    loads = load_df[bidding_zones].fillna(0)

    weighted_price = (prices * loads).sum(axis=1) / loads.sum(axis=1)
    return weighted_price

# =========================
# Country/region-level prices function
# =========================
def country_level_prices(price_df, load_df, country_mapping):
    """
    Returns a DataFrame with country- or region-level load-weighted prices.

    Parameters
    ----------
    price_df : pd.DataFrame
        Zone-level electricity prices (columns = bidding zones)
    load_df : pd.DataFrame
        Zone-level electricity load (columns = bidding zones)
    country_mapping : dict
        Mapping {region_name: [bidding_zone1, bidding_zone2, ...]}

    Returns
    -------
    pd.DataFrame
        Country/region-level load-weighted prices
    """
    # Ensure datetime alignment
    assert price_df.index.equals(load_df.index), "Indexes do not match"

    country_price_df = pd.DataFrame(index=price_df.index)

    for region, bidding_zones in country_mapping.items():
        missing_zones = set(bidding_zones) - set(price_df.columns)
        if missing_zones:
            raise ValueError(f"Missing zones for {region}: {missing_zones}")

        country_price_df[region] = load_weighted_price(
            price_df=price_df,
            load_df=load_df,
            bidding_zones=bidding_zones
        )

    return country_price_df

# =========================
# Example usage
# =========================
# Assuming e_prices and load are already loaded DataFrames
# with matching datetime index and columns = bidding zones

# Ensure datetime index is naive
for df in [e_prices, load]:
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
        
# Ensure only overlapping time
start = max(load.index.min(), e_prices.index.min())
end   = min(load.index.max(), e_prices.index.max())

common_index = load.loc[start:end].index

load_common = load.loc[common_index]
e_prices_common = e_prices.loc[common_index]

# Compute country-level prices
country_prices = country_level_prices(
    price_df=e_prices_common,
    load_df=load_common,
    country_mapping=country_mapping
)

#%%

e_prices = e_prices.join(country_prices, how="outer")







'''   Loop for both   '''


# Loop over both data sources
price_sources = {
    
    "historical": e_prices,
    "pypsa": e_prices_pypsa
}

# Create plots
for source_name, e_prices_source in price_sources.items():

    for country in countries:

        if country not in e_prices_source.columns:
            print(f"Skipping {country} in {source_name}: not found")
            continue

        e_prices_country = e_prices_source[country]

        if source_name == 'pypsa':
            # Create directories
            plot_dir = (
                file_dir / "data" / "prices" / "electricity_prices_plots" / simulation / source_name / country)
        elif source_name == 'historical':
            # Create directories
            plot_dir = (
                file_dir / "data" / "prices" / "electricity_prices_plots" / source_name / country)
            
            
        plot_dir.mkdir(parents=True, exist_ok=True)

        for year in years:
            for time_filter in time_filter_dict:

                if time_filter == "year":
                    start_date = f"{year}-01-01 00:00:00"
                    end_date = (
                        pd.Timestamp(start_date)
                        + pd.DateOffset(years=1)
                        - pd.Timedelta(hours=1)
                    )
                elif time_filter == "fall":
                    start_date = f"{year}-09-01 00:00:00"
                    end_date = pd.Timestamp(start_date) + pd.DateOffset(months=3) - pd.Timedelta(hours=1)
                elif time_filter == "spring":
                    start_date = f"{year}-03-01 00:00:00"
                    end_date = pd.Timestamp(start_date) + pd.DateOffset(months=3) - pd.Timedelta(hours=1)
                elif time_filter == "summer":
                    start_date = f"{year}-06-01 00:00:00"
                    end_date = pd.Timestamp(start_date) + pd.DateOffset(months=3) - pd.Timedelta(hours=1)
                elif time_filter == "winter":
                    start_date = f"{year-1}-12-01 00:00:00"
                    end_date = pd.Timestamp(start_date) + pd.DateOffset(months=3) - pd.Timedelta(hours=1)

                start_date_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
                end_date_str = end_date.strftime("%Y-%m-%d")

                # Filter prices
                e_prices_filtered = e_prices_country.loc[start_date:end_date]

                if e_prices_filtered.empty:
                    continue

                # Prepare DataFrame
                df = e_prices_filtered.to_frame(name="price")
                df["hour"] = df.index.hour

                # Percentiles
                percentiles = np.arange(10, 100, 10)

                hourly_stats = df.groupby("hour")["price"].agg(
                    **{f"p{p}": lambda x, q=p/100: x.quantile(q) for p in percentiles},
                    median="median",
                    mean="mean"
                )

                plt.figure(figsize=(10, 6))

                # Gradient fill
                for i in range(len(percentiles)//2):
                    lower = f"p{percentiles[i]}"
                    upper = f"p{percentiles[-(i+1)]}"
                    alpha = 0.1 + 0.2 * i
                    plt.fill_between(
                        hourly_stats.index,
                        hourly_stats[lower],
                        hourly_stats[upper],
                        color="blue",
                        alpha=alpha
                    )

                # Median & mean
                plt.plot(hourly_stats.index, hourly_stats["median"],
                         color="red", linewidth=2, label="Median")

                plt.plot(hourly_stats.index, hourly_stats["mean"],
                         linestyle="--", color="orange", linewidth=1.5, label="Mean")

                # Legend
                gradient_handles = [
                    Patch(facecolor="blue", alpha=0.1 + 0.2 * i)
                    for i in range(len(percentiles)//2)
                ]
                gradient_labels = [
                    f"{percentiles[i]}–{percentiles[-(i+1)]} percentile"
                    for i in range(len(percentiles)//2)
                ]

                line_handles = [
                    plt.Line2D([], [], color="red", linewidth=2),
                    plt.Line2D([], [], color="orange", linestyle="--", linewidth=1.5)
                ]
                line_labels = ["Median", "Mean"]

                plt.legend(
                    handles=gradient_handles + line_handles,
                    labels=gradient_labels + line_labels,
                    fontsize=ticks_font,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.3),
                    ncol=3
                )

                plt.xticks([0, 12, 23], fontsize=other_font)
                plt.yticks(fontsize=other_font)
                plt.xlabel("Hour of day", fontsize=other_font)
                plt.ylabel("Electricity price [€/MWh]", fontsize=other_font)
                plt.title(
                    f"Daily Electricity Price Profile – {country} ({source_name})\n"
                    f"{start_date_str} to {end_date_str}",
                    fontsize=title_font
                )
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # Save
                plot_path = plot_dir / f"{country}_{source_name}_{start_date_str}_to_{end_date_str}.pdf"
                plt.savefig(plot_path)
                plt.close()
                
