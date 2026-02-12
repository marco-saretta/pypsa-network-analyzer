import pypsa
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


file_dir = Path(__file__).parent.parent.resolve()


simulation = "hindcast-dyn-spec-cap"

scenarios_dict = {
    f"{simulation}_wy_{year}": str(
        file_dir / "simulations" / simulation / f"weather_year_{year}" / "networks" / "base_s_39_elec_Ept.nc"
    )
    for year in range(2020, 2025)
}


plot_dir = file_dir / "simulations" / simulation / "capacity_estimation_plots"
plot_dir.mkdir(exist_ok=True)


countries = [
    "AT",
    "BA",
    "BG",
    "CH",
    "CY",
    "CZ",
    "EE",
    "ES",
    "FI",
    "GE",
    "GR",
    "HU",
    "LV",
    "MD",
    "MK",
    "NL",
    "PL",
    "PT",
    "RS",
    "SI",
    "SK",
    "XK",
]

# --- File paths pattern for generation CSVs ---
generation_file_pattern = str(
    file_dir / "data" / "generation" / "generation_hourly_data" / "generation_{country}_hourly_data.csv"
)


# --- Loop over countries ---
all_countries_installed = {}

for country_code in countries:
    print(f"Processing {country_code}...")

    try:
        # --- Read generation data ---
        generation_file_path = generation_file_pattern.format(country=country_code)
        generation = pd.read_csv(generation_file_path, quotechar='"', sep=",")
        generation.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
        generation["Date"] = pd.to_datetime(generation["Date"])
        generation.set_index("Date", inplace=True)

        installed_cal = {}
        installed_pypsa = {}

        # --- Loop over years / scenarios ---
        for year_key, filepath in scenarios_dict.items():
            year = int(year_key[-4:])
            n = pypsa.Network(str(filepath))  # convert Path to string

            start_date = f"{year}-01-01 00:00:00"
            end_date = (pd.Timestamp(start_date) + pd.DateOffset(years=1) - pd.Timedelta(hours=1)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # Filter generation data
            generation_filtered = generation.loc[start_date:end_date]
            if generation_filtered.index.tz is not None:
                generation_filtered.index = generation_filtered.index.tz_convert(None)

            # Total solar generation
            gen_solar_total = generation_filtered["Solar"].sum()

            # --- Find solar generator dynamically ---
            gen_names = [g for g in n.generators.index if country_code in g and "solar" in g.lower()]
            if not gen_names:
                print(f"Skipping {country_code} for year {year} (no solar generator found)")
                continue

            solar_gen = gen_names[0]

            p_max_pu_series = n.generators_t.p_max_pu[solar_gen].loc[start_date:end_date]
            p_max_pu_solar_mean_year = p_max_pu_series.mean()

            if p_max_pu_solar_mean_year == 0:
                print(f"Skipping {country_code} for year {year} (p_max_pu zero)")
                continue

            # Installed capacity for the year
            installed_capacity = gen_solar_total / (len(generation_filtered) * p_max_pu_solar_mean_year)

            installed_cal[year] = installed_capacity
            installed_pypsa[year] = n.generators.p_nom_opt[gen_names[0]] + n.generators.p_nom_opt[gen_names[1]]

        # --- Align years for plotting ---
        years = sorted(installed_pypsa.keys() & installed_cal.keys())
        df_installed = pd.DataFrame(
            {"PyPSA": [installed_pypsa[y] for y in years], "gen/CF": [installed_cal[y] for y in years]}, index=years
        )

        # --- Plot ---
        plt.figure(figsize=(10, 5))
        df_installed.plot(kind="bar", color=["steelblue", "orange"], edgecolor="k")
        plt.title(f"Installed Capacity Comparison ({country_code} solar + solar hsat) by Year")
        plt.xlabel("Year")
        plt.ylabel("MW Installed Capacity [-]")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save
        plot_path = plot_dir / f"{country_code}.pdf"
        plt.savefig(plot_path)
        plt.close()

        # Store results
        all_countries_installed[country_code] = {"installed_pypsa": installed_pypsa, "installed_cal": installed_cal}

    except Exception as e:
        print(f"Skipping {country_code} due to error: {e}")
        continue


# ---- Aggregate over all countries ----
print("Computing aggregated installed capacity for ALL countries...")

aggregated_installed_cal = {}
aggregated_installed_pypsa = {}

# Determine available years across countries
all_years = sorted({int(k[-4:]) for k in scenarios_dict.keys()})

for year in all_years:
    total_gen_solar = 0
    p_max_pu_list = []
    pypsa_p_nom_opt_total = 0
    hour_count = None

    for country_code in countries:
        # Skip countries that failed earlier
        if country_code not in all_countries_installed:
            continue

        try:
            generation_file_path = generation_file_pattern.format(country=country_code)
            generation = pd.read_csv(generation_file_path, quotechar='"', sep=",")
            generation.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
            generation["Date"] = pd.to_datetime(generation["Date"])
            generation.set_index("Date", inplace=True)

            # Filter dates
            start_date = f"{year}-01-01 00:00:00"
            end_date = (pd.Timestamp(start_date) + pd.DateOffset(years=1) - pd.Timedelta(hours=1)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            generation_filtered = generation.loc[start_date:end_date]
            if generation_filtered.index.tz is not None:
                generation_filtered.index = generation_filtered.index.tz_convert(None)

            if hour_count is None:
                hour_count = len(generation_filtered)

            # Solar generation sum
            total_gen_solar += generation_filtered["Solar"].sum()

            # --- Load PyPSA network ---
            year_key = f"{simulation}_wy_{year}"
            filepath = scenarios_dict[year_key]
            n = pypsa.Network(filepath)

            # Find solar + solar-hsat
            gen_names = [g for g in n.generators.index if country_code in g and "solar" in g.lower()]

            if not gen_names:
                continue

            # p_max_pu mean for each solar generator
            for g in gen_names:
                p_max_pu_list.append(n.generators_t.p_max_pu[g].loc[start_date:end_date].mean())

                # Installed capacity from PyPSA (sum if solar + hsat)
                pypsa_p_nom_opt_total += n.generators.p_nom_opt[g]

        except Exception:
            continue

    # --- Final aggregation for this year ---
    if hour_count is None or len(p_max_pu_list) == 0:
        print(f"Skipping aggregated year {year} (missing data)")
        continue

    mean_p_max_pu_all = sum(p_max_pu_list) / len(p_max_pu_list)

    installed_capacity_total = total_gen_solar / (hour_count * mean_p_max_pu_all)

    aggregated_installed_cal[year] = installed_capacity_total
    aggregated_installed_pypsa[year] = pypsa_p_nom_opt_total


# ---- Plot aggregated results ----
df_agg = pd.DataFrame(
    {
        "PyPSA": [aggregated_installed_pypsa[y] for y in all_years if y in aggregated_installed_pypsa],
        "gen/CF": [aggregated_installed_cal[y] for y in all_years if y in aggregated_installed_cal],
    },
    index=[y for y in all_years if y in aggregated_installed_cal],
)

plt.figure(figsize=(10, 5))
df_agg.plot(kind="bar", color=["steelblue", "orange"], edgecolor="k")
plt.title(f"Installed Capacity Comparison (aggregated) by Year")
plt.xlabel("Year")
plt.ylabel("MW Installed Capacity [-]")
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

plot_path = plot_dir / "ALL_COUNTRIES.pdf"
plt.savefig(plot_path)
plt.close()

print(f"Saved aggregated plot: {plot_path}")
