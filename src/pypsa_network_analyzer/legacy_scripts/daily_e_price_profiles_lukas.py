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
file_dir = Path(__file__).parent.parent.parent.parent.resolve()
external_file_entso_e = f"{file_dir}/data/benchmark/electricity_prices.csv"
# Read in electricity prices and parse dates = True, first column as index
e_prices = pd.read_csv(
    external_file_entso_e,
    parse_dates=True,
    index_col=0
)

# Remove timezone
e_prices.index = e_prices.index.tz_localize(None)

# Rename index properly
e_prices.index.name = "snapshot"

# %%

# =========================
# Country/region to bidding zones mapping
# =========================
country_mapping = {
    "DK": ["DK_1", "DK_2"],
    "SE": ["SE_1", "SE_2", "SE_3", "SE_4"],
    "NO": ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5"],
    "IT": ["IT_CNOR", "IT_CSUD", "IT_NORD", "IT_SARD", "IT_SICI"],
}


# %%

"""   Model data   """
simulations = [
    "hindcast-dyn",
    "hindcast-dyn-rolling",
    "hindcast-std",
]


# ------------------------------------------
# Load model price data for all simulations
# ------------------------------------------
price_models = {}

for simulation in simulations:
    external_file = (
        f"{file_dir}/results_concat/{simulation}/electricity_prices/"
        f"combined_electricity_prices.csv"
    )

    df = pd.read_csv(external_file)
    df.set_index("snapshot", inplace=True)
    df.index = pd.to_datetime(df.index)

    price_models[simulation] = df


# %%


"""   Input for run and plot   """
# Year, time and country to plot
years = [2020, 2021, 2022, 2023, 2024]



""" Calculate load average data for countries with multiple bidding zone """
external_file_load = f"{file_dir}/data/benchmark/demand.csv"
load = pd.read_csv(
    external_file_load,
    parse_dates=True,
    index_col=0
)

# Remove timezone
load.index = load.index.tz_localize(None)

# Rename index properly
load.index.name = "snapshot"


# %%
# Read in the load data from pypsa model results for comparison for the years 2020-2024
# We need to concat 4 files here for each year
load_list = []
for year in years:
    external_file_load_pypsa = (
        f"{file_dir}/results/{simulation}-wy{year}/summary/electric_load.csv"
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

    # First delete any existing country-level columns, which are in the mapping
    for country in country_mapping.keys():
        if country in price_real.columns:
            price_real.drop(columns=country, inplace=True)
            print(f"Dropped existing country column from price_real: {country}")
        if country in load_real.columns:
            load_real.drop(columns=country, inplace=True)
            print(f"Dropped existing country column from load_real: {country}")
        

    # For Luxemburg copy the DE column in prices and add with LU as name 
    price_real["LU"] = price_real["DE"]

    for country, zones in country_mapping.items():
        zones = [z for z in zones if z in price_real.columns and z in load_real.columns]
        if not zones:
            continue

        price_c = price_real[zones]
        load_c = load_real[zones]

        print(f"Aggregating {country} from zones: {zones}")

        # Make sure that when load is missing for up to 5 timesteps
        # we take the average of the last known load value and the next known load value,
        #  to avoid losing too much data
        # Fill gaps up to 5 timesteps using interpolation
        # Treat zeros as missing
        load_c = load_c.replace(0, np.nan)
        load_c = load_c.interpolate(method="time", limit=5)
        price_c = price_c.interpolate(method="time", limit=5)

        weighted_price = (price_c * load_c).sum(axis=1) / load_c.sum(axis=1)
        total_load = load_c.sum(axis=1)

        # Add the new entry to the original dataframes and sort columns alphabetically
        price_real[country] = weighted_price
        load_real[country] = total_load
        price_real = price_real.reindex(sorted(price_real.columns), axis=1)
        load_real = load_real.reindex(sorted(load_real.columns), axis=1)


    return price_real, load_real



#%%
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

# ------------------------------------------
# Harmonize each simulation 
# ------------------------------------------
european_prices = {}

for simulation in simulations:
    european_prices[simulation] = harmonize_and_compare_prices(
        price_model=price_models[simulation],
        load_model=load_pypsa,
        price_real=e_prices,
        load_real=load,
        country_mapping=country_mapping,
    )


# %%
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_weekly_european_prices_all_simulations(
    european_prices: dict,
    title: str = "European Weekly Electricity Prices (2020–2024)",
    x_length: float = 8,
    sim_color: dict | None = None,
):
    """
    Plot weekly European electricity prices styled consistently
    with ResultsPlotter.plot_prices.
    """

    # --- Style identical to class ---
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    plt.rcParams["font.family"] = "Arial"

    phi = 1.618

    # Same Okabe–Ito defaults as ResultsPlotter
    default_sim_color = {
        "benchmark": "#56b4e9",
        "hindcast-dyn": "#009e73",
        "hindcast-dyn-rolling": "#d55e00",
        "hindcast-std": "#e69f00",
    }

    if sim_color is None:
        sim_color = default_sim_color

    fig, ax = plt.subplots(figsize=(x_length, x_length / phi))

    # --- Weekly resample ---
    weekly_data = {}
    for sim, df in european_prices.items():
        weekly_data[sim] = df[["price_model", "price_real"]].resample("W").mean()

    # Historical
    hist = next(iter(weekly_data.values()))["price_real"]

    # --- Plot benchmark (same color logic as class) ---
    ax.plot(
        hist.index,
        hist,
        label="Benchmark",
        color=sim_color["benchmark"],
        linewidth=1.8,
    )

    # --- Plot simulations ---
    for sim, df in weekly_data.items():
        ax.plot(
            df.index,
            df["price_model"],
            label=sim,
            color=sim_color.get(sim, None),
        )

    # --- Formatting identical to class ---
    ax.legend(frameon=True)
    ax.set_title(title, loc="left", fontsize=14, pad=20)
    ax.set_xlim(left=hist.index.min(), right=hist.index.max())
    ax.set_ylabel("EUR/MWh")
    ax.grid(True, linestyle="dashed", alpha=0.5)
    plt.tight_layout()
    # Save as pdf in figures_paper folder
    fig_dir = Path(__file__).parent.parent.parent.parent.resolve() / "figures_paper"
    fig_path = fig_dir / "price_EU.pdf"
    plt.savefig(fig_path, format="pdf")
    plt.show()


# %%
plot_weekly_european_prices_all_simulations(
    european_prices=european_prices,
)

# %%
