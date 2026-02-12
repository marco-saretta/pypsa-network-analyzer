# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 13:31:45 2025

@author: FEGU
"""

import pypsa
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


file_dir = Path(__file__).parent.parent.resolve()


simulation = "hindcast-dyn-spec-cap"
simulation = "hindcast-dyn"


weather_year_dict = {
    "weather_year_2020": 2020,
    "weather_year_2021": 2021,
    "weather_year_2022": 2022,
    "weather_year_2023": 2023,
    "weather_year_2024": 2024,
}


net = str(file_dir / "simulations" / simulation / "weather_year_2020" / "networks" / "base_s_39_elec_Ept.nc")


network = pypsa.Network(net)

color_dict = network.carriers["color"].to_dict()


weather_year = "weather_year_2020"


external_file = (
    f"{file_dir}/simulations/{simulation}/{weather_year}/results/summary/installed_capacities_by_carrier_MW.csv"
)
capacities = pd.read_csv(external_file)
capacities = capacities.set_index("carrier")
capacities = capacities * 1e-3


# Map colors to each carrier

# Map colors for all carriers
bar_colors = [color_dict.get(carrier, "#333333") for carrier in capacities.index]


plt.figure(figsize=(8, 4))
plt.bar(capacities.index, capacities["p_nom_opt"], color=bar_colors)
plt.xticks(rotation=45)
plt.ylabel("Installed Capacity [GW]")
plt.title("Installed Capacity by Technology for year 2020")
plt.tight_layout()
plt.show()


# Choose which carriers to include
include_carriers = [
    "offwind-ac",
    "offwind-dc",
    "offwind-float",
    "onwind",
    "solar",
    "solar-hsat",
]  # <-- add any names you want

# Filter capacities
filtered_capacities = capacities[capacities.index.isin(include_carriers)]

# Map colors for all carriers
bar_colors = [color_dict.get(carrier, "#333333") for carrier in filtered_capacities.index]


plt.figure(figsize=(8, 4))
plt.bar(filtered_capacities.index, filtered_capacities["p_nom_opt"], color=bar_colors)
plt.xticks(rotation=45)
plt.ylabel("Installed Capacity [GW]")
plt.title("Installed Renewable Capacity by Technology for year 2020")
plt.tight_layout()
plt.show()

# %%%


weather_year_dict = {
    "weather_year_2020": 2020,
    "weather_year_2021": 2021,
    "weather_year_2022": 2022,
    "weather_year_2023": 2023,
    "weather_year_2024": 2024,
}

for weather_year in weather_year_dict.keys():
    external_file = (
        f"{file_dir}/simulations/{simulation}/{weather_year}/results/summary/installed_capacities_by_carrier_MW.csv"
    )
    capacities = pd.read_csv(external_file)
    capacities = capacities.set_index("carrier")
    capacities = capacities * 1e-3

    # Choose which carriers to include
    include_carriers = [
        "offwind-ac",
        "offwind-dc",
        "offwind-float",
        "onwind",
        "solar",
        "solar-hsat",
    ]  # <-- add any names you want

    # Filter capacities
    filtered_capacities = capacities[capacities.index.isin(include_carriers)]

    # Map colors for all carriers
    bar_colors = [color_dict.get(carrier, "#333333") for carrier in filtered_capacities.index]

    plt.figure(figsize=(8, 4))
    plt.bar(filtered_capacities.index, filtered_capacities["p_nom_opt"], color=bar_colors)
    plt.xticks(rotation=45)
    plt.ylabel("Installed Capacity [GW]")
    plt.title(f"Installed Renewable Capacity by Technology for year {weather_year_dict[weather_year]}")
    plt.tight_layout()
    plt.show()


# %%%  Multi year plot


all_years_data = []

plot_dir = file_dir / "simulations" / simulation / "Capacities_aggregated_EUR"
plot_dir.mkdir(exist_ok=True)

for weather_year in weather_year_dict.keys():
    # Load CSV
    external_file = (
        f"{file_dir}/simulations/{simulation}/{weather_year}/results/summary/installed_capacities_by_carrier_MW.csv"
    )
    capacities = pd.read_csv(external_file).set_index("carrier")
    capacities = capacities * 1e-3

    # Select renewable carriers
    include_carriers = ["offwind-ac", "offwind-dc", "offwind-float", "onwind", "solar", "solar-hsat"]

    filtered = capacities.loc[capacities.index.isin(include_carriers), "p_nom_opt"]

    # Store results: each iteration becomes a column
    year_num = weather_year_dict[weather_year]  # e.g. "weather_year_2020" → 2020
    all_years_data.append(filtered.rename(f"{year_num}"))

# Combine all years into one table
df_all_years = pd.concat(all_years_data, axis=1)

# Replace missing values with zero (likely no capacity exists for some carriers)
df_all_years = df_all_years.fillna(0)

# Map colors
bar_colors = [color_dict.get(carrier, "#333333") for carrier in df_all_years.index]

# ---- Plot all years in one figure ----

plt.figure(figsize=(10, 6))
df_all_years.T.plot(kind="bar", figsize=(10, 6), color=bar_colors)
plt.xticks(rotation=0)
plt.ylabel("Installed Capacity [GW]")
plt.title(f"Installed Renewable Capacity by Technology Across Weather Years. {simulation}")
plt.tight_layout()
plot_path = plot_dir / "Capacities_aggregated_EUR.pdf"
plt.savefig(plot_path)
plt.close()
