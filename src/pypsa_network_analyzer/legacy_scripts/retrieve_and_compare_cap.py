# %%
import pandas as pd
from entsoe import EntsoePandasClient
from pathlib import Path
import pypsa
#%%

# API token 
API_TOKEN = '0b8225f3-1ca7-4abc-9f42-2b90e0a910ea'

client = EntsoePandasClient(api_key=API_TOKEN)

start = pd.Timestamp("2022-01-01", tz="Europe/Brussels")
end   = pd.Timestamp("2022-12-31 23:59", tz="Europe/Brussels")

country_codes = [
    "AL","AT","BA","BE","BG","CH","CZ","DE","DK","EE","ES","FI","FR","GB",
    "GR","HR","HU","IE","IT","LT","LU","LV","ME","MK","NL","NO","PL","PT",
    "RO","RS","SE","SI","SK","XK"
]

results = []

for code in country_codes:
    print(f"Querying {code}...")
    try:
        df = client.query_installed_generation_capacity(
            country_code=code,
            start=start,
            end=end
        )

        # If Series → convert to DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame(name="capacity_mw")

        # If already DataFrame → keep as is
        df["country"] = code
        results.append(df)

    except Exception as e:
        print(f"Skipping {code}: {e}")

if results:
    all_data = pd.concat(results)
else:
    print("No data retrieved.")

#%%
# ---- Mapping ENTSO-E -> PyPSA carriers ----
carrier_mapping = {
    "Biomass": "biomass",
    "Fossil Brown coal/Lignite": "lignite",
    "Fossil Coal-derived gas": "coal",
    "Fossil Gas": "Combined-Cycle Gas",
    "Fossil Hard coal": "coal",
    "Fossil Oil": "oil",
    "Fossil Oil shale": "oil",
    "Fossil Peat": "coal",
    "Geothermal": "geothermal",
    "Hydro Pumped Storage": "Pumped Hydro Storage",
    "Hydro Run-of-river and poundage": "Run of River",
    "Hydro Water Reservoir": "Reservoir & Dam",
    "Marine": "other",
    "Nuclear": "nuclear",
    "Other": "other",
    "Other renewable": "other",
    "Solar": "Solar",
    "Waste": "biomass",
    "Wind Offshore": "Offshore Wind (AC)",
    "Wind Onshore": "Onshore Wind",
}

# %%
# Remove timestamp index (reset it)
df = all_data.reset_index(drop=True)

# Set country as index
df = df.set_index("country")

# Apply carrier mapping
df = df.rename(columns=carrier_mapping)

# 4Group columns that now share the same name and sum them
df_grouped_entso_e = df.T.groupby(level=0).sum().T
# Add EU as sum of all countries as a new row
df_grouped_entso_e.loc["EU"] = df_grouped_entso_e.sum()

# Check for dublicated countries and only keep the first one (if any)
if df_grouped_entso_e.index.duplicated().any():
    print("Warning: Duplicated countries found. Keeping only the first occurrence.")
    df_grouped_entso_e = df_grouped_entso_e[~df_grouped_entso_e.index.duplicated(keep="first")]


# %%
file_dir = Path(__file__).parent.parent.parent.parent.resolve()
network_2022_path = f"{file_dir}/data/network_files_sarah3_rc/hindcast-dyn-rolling-wy2022.nc"
n_2022 = pypsa.Network(network_2022_path)
# %%
# ---- PyPSA Europe ----
cap_2022 = n_2022.statistics.installed_capacity(groupby=["country", "carrier"])
# %%
cap_country_carrier = cap_2022.groupby(["country", "carrier"]).sum()
# %%
df_grouped_pypsa = cap_country_carrier.unstack("carrier")

df_grouped_pypsa = df_grouped_pypsa.fillna(0)
# Drop columns AC, DC, Load shedding
df_grouped_pypsa = df_grouped_pypsa.drop(columns=["AC", "DC", "Load shedding"], errors="ignore")
df_grouped_pypsa.loc["EU"] = df_grouped_pypsa.sum()
df_grouped_pypsa


# %%

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["legend.fontsize"] = 18
    plt.rcParams["legend.title_fontsize"] = 18


setup_style()


plot_label_mapping = {
    "biomass": "Biomass",
    "lignite": "Lignite",
    "coal": "Coal",
    "Combined-Cycle Gas": "Combined-\nCycle Gas",
    "oil": "Oil",
    "geothermal": "Geothermal",
    "Pumped Hydro Storage": "Pumped \nHydro Storage",
    "Run of River": "Run-of-River",
    "Reservoir & Dam": "Reservoir &\nDam",
    "nuclear": "Nuclear",
    "other": "Other",
    "Solar": "Solar",
    "Offshore Wind (AC)": "Offshore Wind",
    "Onshore Wind": "Onshore Wind",
}


def plot_capacity_comparison(
    country,
    entsoe_df=df_grouped_entso_e,
    pypsa_df=df_grouped_pypsa,
    carrier_order=None,
    ylim=None,
):
    if country not in entsoe_df.index:
        raise ValueError(f"{country} not found in ENTSO-E dataframe.")
    if country not in pypsa_df.index:
        raise ValueError(f"{country} not found in PyPSA dataframe.")

    entsoe = entsoe_df.loc[country]
    pypsa = pypsa_df.loc[country]

    carriers = sorted(set(entsoe.index) | set(pypsa.index))
    if carrier_order is not None:
        carriers = [c for c in carrier_order if c in carriers] + [
            c for c in carriers if c not in carrier_order
        ]

    entsoe = entsoe.reindex(carriers, fill_value=0)
    pypsa = pypsa.reindex(carriers, fill_value=0)

    comparison = pd.DataFrame({"ENTSO-E": entsoe, "PyPSA": pypsa})
    comparison = comparison[(comparison != 0).any(axis=1)]
    comparison = comparison / 1000
    comparison.index = [plot_label_mapping.get(c, c) for c in comparison.index]

    colors = ["#56b4e9", "#fb6a4a"]

    fig, ax = plt.subplots(figsize=(10, 5))
    comparison.plot(kind="bar", ax=ax, color=colors, width=0.5, edgecolor="none")

    ax.set_ylabel("Installed capacity (GW)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", linestyle=":", linewidth=0.8)
    ax.grid(axis="x", linestyle=":", linewidth=0.8)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.legend(title="", loc="upper right", frameon=False)

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    path = file_dir / "figures_paper" / "capacity_comparison_plots"
    path.mkdir(exist_ok=True)
    plt.savefig(path / f"{country}_capacity_comparison.png", dpi=300)

    if ylim is None:
        ylim = ax.get_ylim()

    plt.show()
    return ylim


def plot_capacity_all_countries_by_carrier(
    carrier,
    entsoe_df=df_grouped_entso_e,
    pypsa_df=df_grouped_pypsa,
    country_order=None,
):
    entsoe_df = entsoe_df.copy().drop(index="EU", errors="ignore")
    pypsa_df = pypsa_df.copy().drop(index="EU", errors="ignore")

    if carrier not in entsoe_df.columns and carrier not in pypsa_df.columns:
        raise ValueError(f"{carrier} not found in either dataframe.")

    all_countries = sorted(
        set(entsoe_df.index[entsoe_df.get(carrier, 0) != 0])
        | set(pypsa_df.index[pypsa_df.get(carrier, 0) != 0])
    )

    if country_order is not None:
        all_countries = [c for c in country_order if c in all_countries] + [
            c for c in all_countries if c not in country_order
        ]

    entsoe_series = entsoe_df.get(carrier, pd.Series(0, index=entsoe_df.index)).reindex(all_countries)
    pypsa_series = pypsa_df.get(carrier, pd.Series(0, index=pypsa_df.index)).reindex(all_countries)

    comparison = pd.DataFrame(
        {"PyPSA": pypsa_series, "ENTSO-E": entsoe_series}, index=all_countries
    )
    comparison = comparison[(comparison != 0).any(axis=1)]

    colors = ["#6baed6", "#fb6a4a"]

    n = len(comparison.index)
    fig_width = max(8, n * 0.65)

    fig, ax = plt.subplots(figsize=(fig_width, 5))
    comparison.plot(kind="bar", ax=ax, color=colors, width=0.5, edgecolor="none")

    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", linestyle=":", linewidth=0.8)
    ax.grid(axis="x", linestyle=":", linewidth=0.8)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.legend(title="", loc="upper right", frameon=False)

    plt.tight_layout()

    path = file_dir / "figures_paper" / "capacity_comparison_plots"
    path.mkdir(exist_ok=True)
    plt.savefig(path / f"{carrier}_capacity_comparison.png", dpi=300)
    plt.show()


# %%
carrier_order = [
    "Solar",
    "Onshore Wind",
    "Offshore Wind (AC)",
    "Combined-Cycle Gas",
    "Open-Cycle Gas",
    "oil",
    "coal",
    "lignite",
    "nuclear",
    "biomass",
    "geothermal",
    "Pumped Hydro Storage",
    "Run of River",
    "Reservoir & Dam",
    "other",
]

ylim = plot_capacity_comparison("DE", carrier_order=carrier_order)
plot_capacity_comparison("ES", carrier_order=carrier_order, ylim=ylim)
plot_capacity_comparison("NO", carrier_order=carrier_order)
plot_capacity_comparison("FR", carrier_order=carrier_order)
plot_capacity_comparison("IT", carrier_order=carrier_order)
plot_capacity_comparison("DK", carrier_order=carrier_order)
plot_capacity_comparison("EU", carrier_order=carrier_order)

# %%
plot_capacity_all_countries_by_carrier("Combined-Cycle Gas")
plot_capacity_all_countries_by_carrier("oil")
plot_capacity_all_countries_by_carrier("lignite")
plot_capacity_all_countries_by_carrier("coal")
plot_capacity_all_countries_by_carrier("nuclear")
plot_capacity_all_countries_by_carrier("biomass")
plot_capacity_all_countries_by_carrier("Solar")
plot_capacity_all_countries_by_carrier("Offshore Wind (AC)")
plot_capacity_all_countries_by_carrier("Onshore Wind")
plot_capacity_all_countries_by_carrier("Pumped Hydro Storage")
plot_capacity_all_countries_by_carrier("Run of River")
plot_capacity_all_countries_by_carrier("Reservoir & Dam")
# %%
