# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 10:26:27 2026

@author: FEGU
"""

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# --- Config and paths ---

file_dir = Path(__file__).parent.parent.resolve()


"""   Model data   """
# simulation = "hindcast-dyn-spec-cap"
# simulation = "hindcast-dyn"
# simulation = "hindcast_dyn_old"
# simulation = "hindcast-std"
# simulation="hindcast-dyn-2022"
# simulation="hindcast-dyn-spec-cap"
# simulation="hindcast-dyn-spec-cap-rolling"
simulation = "hindcast-dyn-rolling"
# simulation="hindcast-dyn-spec-cap-irena"

weather_year_dict = {
    "weather_year_2020": 2020,
    "weather_year_2021": 2021,
    "weather_year_2022": 2022,
    "weather_year_2023": 2023,
    "weather_year_2024": 2024,
}

time_filter_dict = ["year", "winter", "spring", "summer", "fall"]

# --- Load PyPSA data ---
generator_dispatch_pypsa = []

for weather_year in weather_year_dict.keys():
    file_path = f"{file_dir}/simulations/{simulation}/{weather_year}/results/summary/total_generators_dispatch_with_hydro_phs.csv"
    df = pd.read_csv(file_path)
    df["weather_year"] = weather_year
    generator_dispatch_pypsa.append(df)

generator_dispatch_pypsa = pd.concat(generator_dispatch_pypsa, ignore_index=True)
generator_dispatch_pypsa.set_index("snapshot", inplace=True)
generator_dispatch_pypsa.index = pd.to_datetime(generator_dispatch_pypsa.index)

technologies = generator_dispatch_pypsa.columns.str.split(" ", n=1).str[1]
valid_cols = technologies.notna()
generator_dispatch_pypsa_sum = (
    generator_dispatch_pypsa.loc[:, valid_cols].groupby(technologies[valid_cols], axis=1).sum()
)

# Convert MWh to GWh
generator_dispatch_pypsa_sum = generator_dispatch_pypsa_sum / 1e6

# --- Load ENTSOE data ---

folder = file_dir / "data" / "generation" / "generation_hourly_data"
dfs = []
for filename in os.listdir(folder):
    if filename.startswith("generation_") and filename.endswith("_hourly_data.csv"):
        file_path = folder / filename
        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        dfs.append(df)

generator_dispatch_entsoe_combine = pd.concat(dfs).groupby(level=0).sum()
generator_dispatch_entsoe_combine.index.name = "time"
generator_dispatch_entsoe_combine = generator_dispatch_entsoe_combine.loc[
    :, ~generator_dispatch_entsoe_combine.columns.str.contains(",")
]

# Convert MWh to TWh
generator_dispatch_entsoe_combine = generator_dispatch_entsoe_combine / 1e6
# --- Mapping ---

mapping = {
    "0 offwind-ac": "Wind Offshore",
    "0 offwind-dc": "Wind Offshore",
    "0 onwind": "Wind Onshore",
    "0 solar": "Solar",
    "0 solar-hsat": "Solar",
    "CCGT": "Fossil Gas",
    "OCGT": "Fossil Gas",
    "PHS": "Hydro Pumped Storage Net",
    "biomass": "Biomass",
    "coal": "Fossil Hard coal",
    "geothermal": "Geothermal",
    "hydro": "Hydro Run-of-river and poundage",
    "lignite": "Fossil Brown coal/Lignite",
    "nuclear": "Nuclear",
    "oil": "Fossil Oil",
    "ror": "Hydro Run-of-river and poundage",
}

mapped = generator_dispatch_pypsa_sum.columns.map(mapping)
keep_mask = mapped.notna()
df_kept = generator_dispatch_pypsa_sum.loc[:, keep_mask]
df_kept.columns = mapped[keep_mask].values
generator_dispatch_pypsa_sum = df_kept.groupby(df_kept.columns, axis=1).sum()

# Filter ENTSOE columns to keep only common tech
common_tech_cols = generator_dispatch_pypsa_sum.columns.intersection(generator_dispatch_entsoe_combine.columns)
generator_dispatch_entsoe_combine = generator_dispatch_entsoe_combine.loc[:, common_tech_cols]

# --- Time window function ---


def get_time_window(year: int, period: str, tz=None):
    if period == "year":
        start = pd.Timestamp(f"{year}-01-01 00:00:00")
        end = start + pd.DateOffset(years=1) - pd.Timedelta(hours=1)
    elif period == "fall":
        start = pd.Timestamp(f"{year}-09-01 00:00:00")
        end = start + pd.DateOffset(months=3) - pd.Timedelta(hours=1)
    elif period == "spring":
        start = pd.Timestamp(f"{year}-03-01 00:00:00")
        end = start + pd.DateOffset(months=3) - pd.Timedelta(hours=1)
    elif period == "summer":
        start = pd.Timestamp(f"{year}-06-01 00:00:00")
        end = start + pd.DateOffset(months=3) - pd.Timedelta(hours=1)
    elif period == "winter":
        start = pd.Timestamp(f"{year - 1}-12-01 00:00:00")
        end = start + pd.DateOffset(months=3) - pd.Timedelta(hours=1)
    else:
        raise ValueError(f"Unknown period: {period}")
    if tz is not None:
        start = start.tz_localize(tz)
        end = end.tz_localize(tz)
    return start, end


# --- Create directories once ---

pypsa_plot_dir = file_dir / "simulations" / simulation / "aggregated_dispatch_comparison"
pypsa_plot_dir.mkdir(parents=True, exist_ok=True)

historical_plot_dir = file_dir / "data" / "generation" / "aggregated_dispatch_comparison"
historical_plot_dir.mkdir(parents=True, exist_ok=True)

# --- Compute dispatch totals ---

generator_sources = {
    "historical": generator_dispatch_entsoe_combine,
    "pypsa": generator_dispatch_pypsa_sum,
}

dispatch_totals = {}

for source_name, dispatch_df in generator_sources.items():
    dispatch_totals[source_name] = {}
    if source_name == "pypsa":
        for weather_year in sorted(weather_year_dict.values()):
            start, end = get_time_window(weather_year, "year")
            period_data = dispatch_df.loc[start:end]
            dispatch_totals[source_name][weather_year] = period_data.sum()
    elif source_name == "historical":
        for weather_year in sorted(weather_year_dict.values()):
            dispatch_totals[source_name][weather_year] = {}
            for time_filter in time_filter_dict:
                start, end = get_time_window(weather_year, time_filter, tz="UTC")
                period_data = dispatch_df.loc[start:end]
                dispatch_totals[source_name][weather_year][time_filter] = period_data.sum()


dispatch_totals = {}

for source_name, dispatch_df in generator_sources.items():
    dispatch_totals[source_name] = {}

    if source_name == "pypsa":
        for weather_year in sorted(weather_year_dict.values()):
            dispatch_totals[source_name][weather_year] = {}
            for time_filter in time_filter_dict:
                start, end = get_time_window(weather_year, time_filter)
                period_data = dispatch_df.loc[start:end]
                dispatch_totals[source_name][weather_year][time_filter] = period_data.sum()

    elif source_name == "historical":
        for weather_year in sorted(weather_year_dict.values()):
            dispatch_totals[source_name][weather_year] = {}
            for time_filter in time_filter_dict:
                start, end = get_time_window(weather_year, time_filter, tz="UTC")
                period_data = dispatch_df.loc[start:end]
                dispatch_totals[source_name][weather_year][time_filter] = period_data.sum()


# --- Plot total yearly dispatch per technology with 5 panels (one per year) ---

years = sorted(weather_year_dict.values())
techs = generator_dispatch_pypsa_sum.columns.tolist()
n_years = len(years)
n_techs = len(techs)

fig, axes = plt.subplots(nrows=1, ncols=n_years, figsize=(20, 5), sharey=True)
width = 0.35
x = np.arange(n_techs)

for i, year in enumerate(years):
    ax = axes[i]

    pypsa_data = dispatch_totals["pypsa"].get(year, {}).get("year", pd.Series(dtype=float)).reindex(techs).fillna(0)
    hist_data = dispatch_totals["historical"].get(year, {}).get("year", pd.Series(dtype=float)).reindex(techs).fillna(0)

    bars1 = ax.bar(x - width / 2, pypsa_data.values, width, label="PyPSA", color="tab:blue")
    bars2 = ax.bar(x + width / 2, hist_data.values, width, label="Historical", color="tab:orange")

    ax.set_title(str(year))
    ax.set_xticks(x)
    ax.set_xticklabels(techs, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    if i == 0:
        ax.set_ylabel("Total Dispatch (TWh)")

# fig.legend(['PyPSA', 'Historical'], loc='lower center', ncol=2, fontsize=12)
# plt.suptitle('Yearly Generation Dispatch by Technology', fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.95])


fig.legend(["PyPSA", "Historical"], loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, fontsize=12)
plt.suptitle("Yearly Generation Dispatch by Technology", fontsize=16)
plt.tight_layout()

# --- Save figure to both plot directories ---

pypsa_fig_path = pypsa_plot_dir / "yearly_dispatch_comparison.png"
historical_fig_path = historical_plot_dir / "yearly_dispatch_comparison.png"

fig.savefig(pypsa_fig_path, dpi=300)
fig.savefig(historical_fig_path, dpi=300)

plt.show()


# Plotting for the different seasons in each year

seasons = ["winter", "spring", "summer", "fall"]
years = sorted(weather_year_dict.values())
techs = generator_dispatch_pypsa_sum.columns.tolist()
n_techs = len(techs)
width = 0.35
x = np.arange(n_techs)

for year in years:
    fig, axes = plt.subplots(nrows=1, ncols=len(seasons), figsize=(20, 5), sharey=True)
    fig.suptitle(f"Seasonal Generation Dispatch Comparison for {year}", fontsize=16)

    for i, season in enumerate(seasons):
        ax = axes[i]

        pypsa_data = dispatch_totals["pypsa"].get(year, {}).get(season, pd.Series(dtype=float)).reindex(techs).fillna(0)

        hist_data = (
            dispatch_totals["historical"].get(year, {}).get(season, pd.Series(dtype=float)).reindex(techs).fillna(0)
        )

        bars1 = ax.bar(x - width / 2, pypsa_data.values, width, label="PyPSA", color="tab:blue")
        bars2 = ax.bar(x + width / 2, hist_data.values, width, label="Historical", color="tab:orange")

        ax.set_title(season.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(techs, rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        # Set y-axis lower limit to zero:
        ax.set_ylim(bottom=0)

        if i == 0:
            ax.set_ylabel("Total Dispatch (TWh)")

    fig.legend(["PyPSA", "Historical"], loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save and show as before...

    # Save figure to plot directories
    pypsa_fig_path = pypsa_plot_dir / f"seasonal_dispatch_comparison_{year}.png"
    historical_fig_path = historical_plot_dir / f"seasonal_dispatch_comparison_{year}.png"

    fig.savefig(pypsa_fig_path, dpi=300)
    fig.savefig(historical_fig_path, dpi=300)

    plt.show()
