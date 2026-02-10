# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 09:26:16 2025

@author: FEGU
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 09:56:29 2025

@author: Lalka
"""
# %%
import pypsa
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
# %%


# Fonts for plot
title_font = 24
other_font = 18
ticks_font = 14


"""   Entso-e data   """
# Load file
file_dir = Path(__file__).parent.parent.resolve()
external_file = f"{file_dir}/data/prices/electricity_prices.csv"
e_prices = pd.read_csv(external_file)
e_prices.rename(columns={"Unnamed: 0": "snapshot"}, inplace=True)
e_prices.set_index("snapshot", inplace=True)
# After reading CSV
e_prices.index = pd.to_datetime(e_prices.index)
e_prices.index = e_prices.index.tz_localize(None)


# %%

# =========================
# Country/region to bidding zones mapping
# =========================
country_mapping = {
    "DK": ["DK_1", "DK_2"],
    "SE": ["SE_1", "SE_2", "SE_3", "SE_4"],
    "NO": ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5"],
    "IT": ["IT_CNOR", "IT_CSUD", "IT_NORD", "IT_SARD", "IT_SICI"],
    "DE": ["DE_LU"],
    # "EUR": [
    #     "AT", "BE", "BG", "CH", "CZ", "DE_LU", "DK_1", "DK_2", "EE", "ES", "FI",
    #     "FR", "GR", "HR", "HU", "IE_SEM", "IT_CNOR", "IT_CSUD", "IT_NORD",
    #     "IT_SARD", "IT_SICI", "LV", "LT", "MK", "NL", "NO_1", "NO_2", "NO_3",
    #     "NO_4", "NO_5", "PL", "PT", "RO", "RS", "SE_1", "SE_2", "SE_3", "SE_4",
    #     "SI", "SK", "UA_IPS"
    # ],
}


# %%

"""   Model data   """
# simulation = "hindcast-dyn-spec-cap"
# simulation = "hindcast-dyn"
# simulation = "hindcast_dyn_old"
# simulation = "hindcast-std"
# simulation="hindcast-dyn-2022"
# simulation="hindcast-dyn-spec-cap"
# simulation="hindcast-dyn-spec-cap-rolling"
# simulation = "hindcast-dyn-rolling"
simulation = "hindcast-dyn-spec-cap-irena"


external_file = f"{file_dir}/simulations/{simulation}/results_concat/combined_electricity_prices.csv"
e_prices_pypsa = pd.read_csv(external_file)
e_prices_pypsa.set_index("snapshot", inplace=True)
e_prices_pypsa.index = pd.to_datetime(e_prices_pypsa.index)


# %%


"""   Input for run and plot   """
# Year, time and country to plot
years = [2020, 2021, 2022, 2023, 2024]


countries = [
    "AL",
    "AT",
    "BA",
    "BE",
    "BG",
    "CH",
    "CZ",
    "DE",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "GB",
    "GBI",
    "GR",
    "HR",
    "HU",
    "IE",
    "IT",
    "LT",
    "LU",
    "LV",
    "ME",
    "MK",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "RS",
    "SE",
    "SI",
    "SK",
    "XK",
    "EUR",
    "DE_LU",
]


time_filter_dict = ["year", "winter", "spring", "summer", "fall"]


""" Calculate load average data for countries with multiple bidding zone """
external_file_load = f"{file_dir}/data/load/demand.csv"
load = pd.read_csv(external_file_load)
load.rename(columns={"Unnamed: 0": "snapshot"}, inplace=True)
load.set_index("snapshot", inplace=True)
# After reading CSV
load.index = pd.to_datetime(load.index)
load.index = load.index.tz_localize(None)


# %%
# Read in the load data from pypsa model results for comparison for the years 2020-2024
# We need to concat 4 files here for each year
load_list = []
for year in years:
    external_file_load_pypsa = (
        f"{file_dir}/simulations/{simulation}/weather_year_{year}/results/summary/electric_load.csv"
    )
    load_year = pd.read_csv(external_file_load_pypsa)
    load_year.set_index("snapshot", inplace=True)
    load_year.index = pd.to_datetime(load_year.index)
    load_list.append(load_year)
load_pypsa = pd.concat(load_list)
load_pypsa

# %%
# filter load and e_prices to only include timestamps that are present in e_prices_pypsa
index = e_prices_pypsa.index
load = load.loc[index]
e_prices = e_prices.loc[index]
# %%


# %%


def aggregate_real_to_countries(
    price_real: pd.DataFrame,
    load_real: pd.DataFrame,
    country_mapping: dict,
):
    """
    Aggregate zone-level real price & load data to country level
    using load-weighted prices.
    """

    for country, zones in country_mapping.items():
        zones = [z for z in zones if z in price_real.columns and z in load_real.columns]
        if not zones:
            continue

        price_c = price_real[zones]
        load_c = load_real[zones]

        weighted_price = (price_c * load_c).sum(axis=1) / load_c.sum(axis=1)
        total_load = load_c.sum(axis=1)

        # Add the new entry to the original dataframes and sort columns alphabetically
        price_real[country] = weighted_price
        load_real[country] = total_load
        price_real = price_real.reindex(sorted(price_real.columns), axis=1)
        load_real = load_real.reindex(sorted(load_real.columns), axis=1)

    return price_real, load_real


# %%


def harmonize_and_compare_prices(
    price_model: pd.DataFrame,
    load_model: pd.DataFrame,
    price_real: pd.DataFrame,
    load_real: pd.DataFrame,
    country_mapping: dict,
):
    """
    Per-timestamp harmonized load-weighted prices for modeled and real data.

    Workflow:
    1) Aggregate real data to country level
    2) Align timestamps
    3) Find common country columns across all 4 datasets
    4) Per-timestamp masking: zone contributes only if all 4 values exist
    5) Compute load-weighted prices
    """

    # -------------------------------------------------
    # 1) Aggregate real data to countries FIRST
    # -------------------------------------------------
    price_real_c, load_real_c = aggregate_real_to_countries(price_real, load_real, country_mapping)

    # -------------------------------------------------
    # 2) Align timestamps
    # -------------------------------------------------
    common_index = (
        price_model.index.intersection(load_model.index)
        .intersection(price_real_c.index)
        .intersection(load_real_c.index)
    )

    pm = price_model.loc[common_index]
    lm = load_model.loc[common_index]
    pr = price_real_c.loc[common_index]
    lr = load_real_c.loc[common_index]

    # -------------------------------------------------
    # 3) Find common country columns
    # -------------------------------------------------
    common_countries = set(pm.columns) & set(lm.columns) & set(pr.columns) & set(lr.columns)
    common_countries = sorted(common_countries)

    print("Common countries used in comparison:")
    print(common_countries)

    pm = pm[common_countries]
    lm = lm[common_countries]
    pr = pr[common_countries]
    lr = lr[common_countries]

    # -------------------------------------------------
    # 4) Validity mask (per timestamp & country)
    # -------------------------------------------------
    valid = pm.notna() & lm.notna() & pr.notna() & lr.notna()

    # -------------------------------------------------
    # 5) Load-weighted prices (masked)
    # -------------------------------------------------
    european_price = pd.DataFrame(index=common_index)

    european_price["price_model"] = (pm * lm * valid).sum(axis=1) / (lm * valid).sum(axis=1)
    european_price["price_real"] = (pr * lr * valid).sum(axis=1) / (lr * valid).sum(axis=1)

    european_price["load_model"] = (lm * valid).sum(axis=1)
    european_price["load_real"] = (lr * valid).sum(axis=1)

    return european_price


# %%

# Call the function to harmonize and compare prices
european_price = harmonize_and_compare_prices(
    price_model=e_prices_pypsa,
    load_model=load_pypsa,
    price_real=e_prices,
    load_real=load,
    country_mapping=country_mapping,
)

# %%

# Daily resampling
daily_price = european_price[["price_model", "price_real"]].resample("D").mean()

# Add helpers
daily_price["year"] = daily_price.index.year
daily_price["doy"] = daily_price.index.dayofyear


# %%
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Plot: all years in one figure
# =========================


def plot_european_daily_prices_all_years(
    european_price: pd.DataFrame,
    years: list,
    title: str = "European Daily Electricity Prices",
    cmap_name: str = "viridis",
):
    """
    Plot daily European electricity prices:
    - All years in one plot
    - One color per year
    - Model: solid line
    - Historical: dashed line
    - Two legends:
        * Line style (model vs historical)
        * Color (year)
    """

    # -------------------------------------------------
    # 1) Daily resampling
    # -------------------------------------------------
    daily = european_price[["price_model", "price_real"]].resample("D").mean()
    daily["year"] = daily.index.year
    daily["doy"] = daily.index.dayofyear

    # -------------------------------------------------
    # 2) Colors per year
    # -------------------------------------------------
    years_sorted = sorted(years)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.1, 0.9, len(years_sorted)))
    color_map = dict(zip(years_sorted, colors))

    # -------------------------------------------------
    # 3) Plot
    # -------------------------------------------------
    plt.figure(figsize=(14, 7))

    for year in years_sorted:
        data_y = daily[daily["year"] == year]
        if data_y.empty:
            continue

        color = color_map[year]

        # Model (solid)
        plt.plot(
            data_y["doy"],
            data_y["price_model"],
            color=color,
            linewidth=2,
        )

        # Historical (dashed)
        plt.plot(
            data_y["doy"],
            data_y["price_real"],
            color=color,
            linewidth=2,
            linestyle="--",
        )

    # -------------------------------------------------
    # 4) Legend 1: line style (model vs historical)
    # -------------------------------------------------
    style_handles = [
        plt.Line2D([0], [0], color="black", lw=2, linestyle="-"),
        plt.Line2D([0], [0], color="black", lw=2, linestyle="--"),
    ]
    style_labels = ["Model", "Historical"]

    style_legend = plt.legend(
        style_handles,
        style_labels,
        fontsize=14,
        loc="upper left",
        title="Data source",
    )

    # IMPORTANT: keep first legend
    plt.gca().add_artist(style_legend)

    # -------------------------------------------------
    # 5) Legend 2: year → color
    # -------------------------------------------------
    year_handles = [plt.Line2D([0], [0], color=color_map[year], lw=3) for year in years_sorted]

    year_labels = [str(year) for year in years_sorted]

    plt.legend(
        year_handles,
        year_labels,
        fontsize=12,
        loc="upper right",
        title="Year",
        ncol=1,
    )

    # -------------------------------------------------
    # 6) Labels & styling
    # -------------------------------------------------
    plt.xlabel("Day of year", fontsize=16)
    plt.ylabel("Electricity price [€/MWh]", fontsize=16)
    plt.title(title, fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


# %%
plot_european_daily_prices_all_years(
    european_price=european_price,
    years=years,
    title="European Daily Electricity Prices (Model vs Historical)",
)

# %%
