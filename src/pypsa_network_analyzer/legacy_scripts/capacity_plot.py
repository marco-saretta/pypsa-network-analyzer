# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 11:03:10 2025

@author: FEGU
"""

""" FEGU functions """
from pathlib import Path
import pandas as pd

file_dir = Path(__file__).parent.parent.resolve()
import pypsa


def collect_capacities(simulation: str, weather_year_dict: dict, weather_year: float, capacities_list: list):
    # FEGU change
    # --- Installed capacities merging ---
    external_file = f"{file_dir}/simulations/{simulation}/{weather_year}/results/summary/installed_capacities.csv"

    yearly_capacities = pd.read_csv(external_file)

    # Expecting columns: ["generator", "capacity"]
    # Rename "capacity" → <weather_year_int>
    year = weather_year_dict[weather_year]
    yearly_capacities = yearly_capacities.set_index(["generator", "bus"]).rename(columns={"p_nom_opt": year})

    capacities_list.append(yearly_capacities)

    return capacities_list


def plot_capacities_for_myears_all_generators(capacities_list, simulation: str, logger: None):
    import matplotlib.pyplot as plt

    if capacities_list:
        capacities_df = pd.concat(capacities_list, axis=1)

        output_path = file_dir / "simulations" / simulation / "merged_installed_capacities.csv"

        capacities_df.to_csv(output_path)
        logger.info(f"Merged installed capacities saved to {output_path}")

        # --------------- CREATE PLOTS PER BUS ------------------- #
        plot_dir = file_dir / "simulations" / simulation / "capacity_plots"
        plot_dir.mkdir(exist_ok=True)

        # unique bus names
        buses = capacities_df.index.get_level_values("bus").unique()

        for bus in buses:
            df_bus = capacities_df.xs(bus, level="bus")

            plt.figure(figsize=(10, 6))
            df_bus.plot(kind="bar")  # each generator = row, each year = column
            plt.title(f"Installed Capacity by Generator for Bus {bus}")
            plt.xlabel("Generator")
            plt.ylabel("Capacity")
            plt.tight_layout()

            # Save
            plot_path = plot_dir / f"{bus}.pdf"
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"Saved plot: {plot_path}")


def plot_capacities_for_myears(capacities_list, simulation: str, logger: None):
    import matplotlib.pyplot as plt

    if capacities_list:
        capacities_df = pd.concat(capacities_list, axis=1)

        output_path = file_dir / "simulations" / simulation / "merged_installed_capacities.csv"

        capacities_df.to_csv(output_path)
        logger.info(f"Merged installed capacities saved to {output_path}")

        # --------------- CREATE PLOTS PER BUS ------------------- #
        plot_dir = file_dir / "simulations" / simulation / "capacity_plots"
        plot_dir.mkdir(exist_ok=True)

        # unique bus names
        buses = capacities_df.index.get_level_values("bus").unique()

        import re

        def has_token_zero(name):
            # split on spaces, underscores, and hyphens
            tokens = re.split(r"[ _\-]+", str(name))
            return "0" in tokens

        for bus in buses:
            df_bus = capacities_df.xs(bus, level="bus")

            # Filter generators that contain token "0"
            df_bus = df_bus[df_bus.index.map(has_token_zero)]

            # If no generators left, skip
            if df_bus.empty:
                continue

            plt.figure(figsize=(10, 6))
            df_bus.plot(kind="bar")  # each generator = row, each year = column
            plt.title(f"Installed Capacity by Generator for Bus {bus}")
            plt.xlabel("Generator")
            plt.ylabel("Capacity")
            plt.tight_layout()

            # Save
            plot_path = plot_dir / f"{bus}.pdf"
            plt.savefig(plot_path)
            plt.close()

            logger.info(f"Saved plot: {plot_path}")


def collect_generator_info(simulation: str, weather_year_dict: dict, weather_year: float, gen_cap_info_list: list):
    # FEGU change
    # --- Installed capacities merging ---
    external_file = (
        f"{file_dir}/simulations/{simulation}/{weather_year}/results/summary/generator_capacity_collected_info.csv"
    )

    gen_cap_info = pd.read_csv(external_file)

    cols = gen_cap_info.columns
    year = weather_year_dict[weather_year]
    gen_cap_info = gen_cap_info.set_index("generator").rename(
        columns={col: f"{col}_{year}" for col in cols if col not in ["generator", "bus"]}
    )

    gen_cap_info_list.append(gen_cap_info)

    return gen_cap_info_list


def generator_cap_check(gen_cap_info_list, simulation: str, logger: None):

    output_path = file_dir / "simulations" / simulation / "generator_extendable.csv"

    inconsistencies_path = file_dir / "simulations" / simulation / "inconsistencies.csv"

    if gen_cap_info_list:
        gen_cap_info_df = pd.concat(gen_cap_info_list, axis=1)

        # --- Identify columns with year suffix ---
        col_groups = {}
        for col in gen_cap_info_df.columns:
            if "_" in col:
                base, year = col.rsplit("_", 1)
                if year.isdigit():
                    col_groups.setdefault(base, []).append(col)

        # --- Check consistency across years (rounded to nearest 10) ---
        inconsistencies_list = []

        for base, columns in col_groups.items():
            subset = (gen_cap_info_df[columns] / 10).round(0) * 10  # round to nearest 10

            # Rows where values differ across years
            unequal_mask = subset.nunique(axis=1) > 1

            if unequal_mask.any():
                inconsistent_rows = subset.loc[unequal_mask].copy()
                inconsistent_rows.insert(0, "variable", base)
                inconsistent_rows.insert(1, "generator", inconsistent_rows.index)
                inconsistencies_list.append(inconsistent_rows)

        # --- Combine all inconsistencies ---
        if inconsistencies_list:
            inconsistencies_df = pd.concat(inconsistencies_list, axis=0)
        else:
            # Create a CSV with one row indicating no inconsistencies
            inconsistencies_df = pd.DataFrame({"Info": ["No inconsistencies found"]})

        # --- Save inconsistencies CSV ---
        inconsistencies_df.to_csv(inconsistencies_path, index=False)

        # --- Save extendable cutout ---
        extendable_cols = [c for c in gen_cap_info_df.columns if "p_nom_extendable" in c]
        cutout_extendable = gen_cap_info_df[gen_cap_info_df[extendable_cols].any(axis=1)]
        cutout_extendable.to_csv(output_path)

        return
