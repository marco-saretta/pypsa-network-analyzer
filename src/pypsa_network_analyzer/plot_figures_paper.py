import hydra
from omegaconf import DictConfig
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import seaborn as sns
import pypsa
import numpy as np
import re

# ---------------------------------------------------------------------------
# Capacity-comparison constants
# ---------------------------------------------------------------------------

CAPACITY_BENCHMARK_YEAR = 2022
CAPACITY_NETWORK_STEM   = "hindcast-dyn-rolling-wy2022"

CAPACITY_PLOT_LABEL_MAPPING: dict[str, str] = {
    "biomass":              "Biomass",
    "lignite":              "Lignite",
    "coal":                 "Coal",
    "Combined-Cycle Gas":   "Combined-\nCycle Gas",
    "oil":                  "Oil",
    "geothermal":           "Geothermal",
    "Pumped Hydro Storage": "Pumped \nHydro Storage",
    "Run of River":         "Run-of-River",
    "Reservoir & Dam":      "Reservoir &\nDam",
    "nuclear":              "Nuclear",
    "other":                "Other",
    "Solar":                "Solar",
    "Offshore Wind (AC)":   "Offshore Wind",
    "Onshore Wind":         "Onshore Wind",
}

CAPACITY_CARRIER_ORDER: list[str] = [
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

# ENTSO-E, PyPSA  (per-country bars)
CAPACITY_COLORS_COUNTRY = ["#56b4e9", "#fb6a4a"]
# PyPSA, ENTSO-E  (cross-country bars)
CAPACITY_COLORS_CARRIER = ["#6baed6", "#fb6a4a"]

class ResultsPlotter:
    """Class to handle loading and plotting of simulation results."""

    def __init__(self, cfg: DictConfig):
        # Config and directories
        self.cfg = cfg
        self.root_dir = Path(cfg.paths.root)
        self.data_dir = Path(cfg.paths.data)
        self.results_dir = Path(cfg.paths.results)
        self.results_concat_dir = Path(cfg.paths.results_concat)
        self.figures_dir = Path(cfg.paths.figures)
        self.figures_dir.mkdir(exist_ok=True)

        # Configuration
        self.sim_labels = list(cfg.config_results_concat.keys())
        self.error_list = ["mae", "rmse", "smape"]
        self.benchmark_name = "electricity_prices"
        self.benchmark_name_unfiltered = "electricity_prices_unfiltered"
        self.load_shed_name = "load_shedding_times"
        self.export_format = cfg.plot_export_format
        self.error_units = {"mae": "EUR/MWh", "rmse": "EUR/MWh", "smape": "%"}

        # Config labels
        self.error_max_values = {"mae": 250, "rmse": 300, "smape": 125}
        self.error_axis_labels = {"mae": "[EUR/MWh]", "rmse": "[EUR/MWh]", "smape": "[%]"}

        self.setup_style()
        self.load_scores()
        self.load_prices()
        self.load_prices_unfiltered()
        self.load_timeseries_load_shedding()
        self.load_capacity_data()

    def setup_style(self):
        plt.style.use("seaborn-v0_8-whitegrid")
        mpl.rcParams["axes.spines.right"] = False
        mpl.rcParams["axes.spines.top"] = False
        # sns.set_theme(style="whitegrid",context="paper",font_scale=1.1)
        self.phi = 1.618

        # Colorblind-friendly palette from Okabe-Ito
        self.nature_orange = "#e69f00"
        self.nature_sky_blue = "#56b4e9"
        self.nature_bluish_green = "#009e73"
        self.nature_yellow = "#f0e442"
        self.nature_blue = "#0072b2"
        self.nature_vermillion = "#d55e00"
        self.nature_reddish_purple = "#cc79a7"

        # Set global Matplotlib font to Arial using plt.rcParams
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["axes.titleweight"] = "bold"
        # plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams["axes.labelsize"] = 16
        plt.rcParams["xtick.labelsize"] = 16
        plt.rcParams["ytick.labelsize"] = 16
        plt.rcParams["legend.fontsize"] = 16
        # Legent titles should be same size as axis labels
        plt.rcParams["legend.title_fontsize"] = 16

        self.sim_color = {
            "benchmark": self.nature_sky_blue,
            "hindcast-dyn": self.nature_bluish_green,
            "hindcast-dyn-rolling": self.nature_vermillion,
            "hindcast-std": self.nature_orange,
        }

        # Fixed display order and human-readable labels
        self.sim_plot_order = [
            "hindcast-std",
            "hindcast-dyn",
            "hindcast-dyn-rolling",
        ]
        self.sim_display_names = {
            "hindcast-std": "Hindcast-\nstatic",
            "hindcast-dyn": "Hindcast-\ndynamic",
            "hindcast-dyn-rolling": "Hindcast-dynamic-\nrolling horizon",
            "benchmark": "Historical",
        }


    def load_scores(self):
        """Load all error scores (hourly, daily, weekly) into a nested dictionary structure."""
        self.scores_dict = {}

        temporal_resolutions = {
            "hourly": "",
            "daily": "_daily",
            "weekly": "_weekly",
        }

        for sim_label in self.sim_labels:
            self.scores_dict[sim_label] = {}
            for error in self.error_list:
                self.scores_dict[sim_label][error] = {}
                for temporal_res, suffix in temporal_resolutions.items():
                    error_file = f"scores_{error}{suffix}.csv"
                    file_path = (
                        self.results_concat_dir
                        / sim_label
                        / self.benchmark_name
                        / "scores"
                        / error_file
                    )

                    if not file_path.exists():
                        print(f"Warning: Missing file (skipping): {file_path}")
                        self.scores_dict[sim_label][error][temporal_res] = None
                        continue

                    df = pd.read_csv(file_path, index_col=0)
                    self.scores_dict[sim_label][error][temporal_res] = df

    def load_data(self):
        """Load all error scores into a dictionary structure."""
        self.scores_dict = {}

        for sim_label in self.sim_labels:
            self.scores_dict[sim_label] = {}
            for error in self.error_list:
                error_file = f"scores_{error}.csv"
                file_path = self.results_concat_dir / sim_label / self.benchmark_name / "scores" / error_file

                if not file_path.exists():
                    raise FileNotFoundError(f"Missing file: {file_path}")

                df = pd.read_csv(file_path, index_col=0)
                self.scores_dict[sim_label][error] = df

    def load_prices(self, sim_labels=None, interpolate=True):
        """
        Load and clean benchmark + simulation prices.
        Stores result in self.prices_dict dict.
        """
        if sim_labels is None:
            sim_labels = self.sim_labels

        self.prices_dict = {}

        # Load simulation
        for sim_label in sim_labels:
            file_dir = self.results_concat_dir / sim_label / self.benchmark_name / f"combined_{self.benchmark_name}.csv"

            df_sim = pd.read_csv(file_dir, index_col=0, parse_dates=True)
            df_sim = df_sim[df_sim.index.year.isin(self.cfg.years_list)]

            self.prices_dict[sim_label] = df_sim

        # Load benchmark
        benchmark_path = self.data_dir / "benchmark" / "electricity_prices.csv"
        df_bench = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)

        if df_bench.index.tz is None:
            df_bench.index = df_bench.index.tz_localize("UTC")
        else:
            df_bench.index = df_bench.index.tz_convert("UTC")

        if interpolate:
            df_bench = df_bench.interpolate().ffill().bfill()

        df_bench = df_bench[df_bench.index.year.isin(self.cfg.years_list)]

        self.prices_dict["benchmark"] = df_bench

    def load_prices_unfiltered(self, sim_labels=None, interpolate=True):
        """
        Load and clean benchmark + simulation prices.
        Stores result in self.prices_unfiltered_dict dict.
        """
        if sim_labels is None:
            sim_labels = self.sim_labels

        self.prices_unfiltered_dict = {}

        # Load simulation
        for sim_label in sim_labels:
            file_dir = (
                self.results_concat_dir
                / sim_label
                / self.benchmark_name_unfiltered
                / f"combined_{self.benchmark_name_unfiltered}.csv"
            )

            df_sim = pd.read_csv(file_dir, index_col=0, parse_dates=True)
            df_sim = df_sim[df_sim.index.year.isin(self.cfg.years_list)]

            self.prices_unfiltered_dict[sim_label] = df_sim

        # Load benchmark
        benchmark_path = self.data_dir / "benchmark" / "electricity_prices.csv"
        df_bench = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)

        if df_bench.index.tz is None:
            df_bench.index = df_bench.index.tz_localize("UTC")
        else:
            df_bench.index = df_bench.index.tz_convert("UTC")

        if interpolate:
            df_bench = df_bench.interpolate().ffill().bfill()

        df_bench = df_bench[df_bench.index.year.isin(self.cfg.years_list)]

        self.prices_unfiltered_dict["benchmark"] = df_bench

    def load_timeseries_load_shedding(self, sim_labels=None):
        """
        Load load shedding timestamps for each simulation.
        Stores result in self.load_shedding_dict.
        """

        if sim_labels is None:
            sim_labels = self.sim_labels

        self.load_shedding_dict = {}

        for sim_label in sim_labels:
            file_path = (
                self.results_concat_dir
                / sim_label
                / self.load_shed_name
                / f"combined_{self.load_shed_name}.csv"
            )

            series = pd.read_csv(
                file_path,
                usecols=["load_shedding_time"],  # optional, ensures only needed column
            )["load_shedding_time"]

            # Ensure datetimes
            series = pd.to_datetime(series, errors="coerce")  # invalid parsing becomes NaT

            # Drop NaT values
            series = series.dropna()

            # Ensure UTC tz
            if series.dt.tz is None:
                series = series.dt.tz_localize("UTC")
            else:
                series = series.dt.tz_convert("UTC")

            # Filter years
            series = series[series.dt.year.isin(self.cfg.years_list)]

            self.load_shedding_dict[sim_label] = series

    def load_capacity_data(self):
            """
            Load ENTSO-E benchmark capacity (2022) and PyPSA capacity for
            hindcast-dyn-rolling-wy2022 into:
                self.entsoe_capacity_df  — (country × carrier), MW
                self.pypsa_capacity_df   — (country × carrier), MW

            Sets attributes to None and prints a warning if files are missing;
            capacity plots are then skipped gracefully.
            """
            entsoe_path = (
                self.data_dir / "benchmark"
                / f"entsoe_installed_capacity_{CAPACITY_BENCHMARK_YEAR}.csv"
            )
            if entsoe_path.exists():
                self.entsoe_capacity_df = pd.read_csv(entsoe_path, index_col=0)
            else:
                print(
                    f"Warning: ENTSO-E capacity file not found — capacity plots will be skipped.\n"
                    f"  Expected: {entsoe_path}\n"
                    f"  Run fetch_and_save_entsoe_capacity() first."
                )
                self.entsoe_capacity_df = None

            pypsa_path = (
                self.results_dir / CAPACITY_NETWORK_STEM / "summary"
                / "installed_capacity_by_country_carrier_MW.csv"
            )
            if pypsa_path.exists():
                self.pypsa_capacity_df = pd.read_csv(pypsa_path, index_col=0)
            else:
                print(
                    f"Warning: PyPSA capacity file not found — capacity plots will be skipped.\n"
                    f"  Expected: {pypsa_path}\n"
                    f"  Run NetworkAnalyzer.extract_pypsa_capacity() first."
                )
                self.pypsa_capacity_df = None

    def plot_error_by_simulation_and_year(self, error_metric, x_length=8, temporal_resolution="hourly"):
        """Create boxplot showing error metric by simulation and year for a given temporal resolution."""
        temporal_suffix_map = {"hourly": "", "daily": "_daily", "weekly": "_weekly"}
        suffix = temporal_suffix_map.get(temporal_resolution, "")

        records = []

        # Respect fixed order, only include labels present in sim_labels
        ordered_sim_labels = [s for s in self.sim_plot_order if s in self.sim_labels]


        for sim_name in ordered_sim_labels:
            csv_path = (
                self.results_concat_dir
                / sim_name
                / self.benchmark_name
                / "scores"
                / f"scores_{error_metric}{suffix}.csv"
            )

            if not csv_path.exists():
                print(f"Warning: Missing file (skipping): {csv_path}")
                continue

            df = pd.read_csv(csv_path, index_col=0)
            df_long = df.reset_index(names="year").melt(
                id_vars="year",
                var_name="country",
                value_name=error_metric,
            )
            # Apply display name here, not the raw key
            df_long["simulation"] = self.sim_display_names.get(sim_name, sim_name)
            records.append(df_long)

        if not records:
            print(f"No data found for {error_metric} ({temporal_resolution}). Skipping plot.")
            return

        long_df = pd.concat(records, ignore_index=True)
        long_df["year"] = long_df["year"].astype(str)
        year_order = sorted(long_df["year"].unique())

        # Use display names in the correct order for x-axis
        sim_order_display = [self.sim_display_names.get(s, s) for s in ordered_sim_labels]


        plt.figure(figsize=(x_length, x_length / self.phi))

        sns.boxplot(
            data=long_df,
            x="simulation",
            y=error_metric,
            hue="year",
            hue_order=year_order,
            order=sim_order_display,
            palette=sns.color_palette(palette="Blues"),
            width=0.8,
            linewidth=0.6,
            showfliers=False,
        )
        sns.despine(right=True, top=True)

        plt.xlabel("")
        plt.xticks(rotation=0, ha="center")
        plt.grid(axis="y", alpha=0.3)
        plt.ylim(bottom=0, top=self.error_max_values[error_metric])
        plt.ylabel(self.error_axis_labels[error_metric])
        #plt.title(f"{error_metric.upper()} — {temporal_resolution.capitalize()}", loc="left", fontsize=13)

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            handles=handles,
            labels=labels,
            title="Year",
            frameon=True,
            loc="upper right",
            ncols=2,
            columnspacing=1.0,
            title_fontproperties={"size": plt.rcParams["legend.title_fontsize"]},
        )
        plt.tight_layout()

        output_path = self.figures_dir / f"{error_metric}_by_simulation_and_year_{temporal_resolution}.{self.export_format}"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")


    def plot_error_by_simulation_and_year_all(self, x_length=8, temporal_resolution="hourly"):
        """Create boxplot showing all error metrics by simulation and year for a given temporal resolution."""
        temporal_suffix_map = {"hourly": "", "daily": "_daily", "weekly": "_weekly"}
        suffix = temporal_suffix_map.get(temporal_resolution, "")

        ordered_sim_labels = [s for s in self.sim_plot_order if s in self.sim_labels]


        long_dfs = {}
        for error_metric in self.error_list:
            records = []
            for sim_name in ordered_sim_labels:
                csv_path = (
                    self.results_concat_dir
                    / sim_name
                    / self.benchmark_name
                    / "scores"
                    / f"scores_{error_metric}{suffix}.csv"
                )

                if not csv_path.exists():
                    print(f"Warning: Missing file (skipping): {csv_path}")
                    continue

                df = pd.read_csv(csv_path, index_col=0)
                df_long = df.reset_index(names="year").melt(
                    id_vars="year",
                    var_name="country",
                    value_name=error_metric,
                )
                df_long["simulation"] = self.sim_display_names.get(sim_name, sim_name)
                records.append(df_long)

            if not records:
                print(f"No data found for {error_metric} ({temporal_resolution}). Skipping.")
                continue

            long_df = pd.concat(records, ignore_index=True)
            long_df["year"] = long_df["year"].astype(str)
            long_dfs[error_metric] = long_df

        if not long_dfs:
            print(f"No data available for temporal_resolution='{temporal_resolution}'. Skipping plot.")
            return

        available_metrics = [m for m in self.error_list if m in long_dfs]
        year_order = sorted(long_dfs[available_metrics[0]]["year"].unique())
        sim_order_display = [self.sim_display_names.get(s, s) for s in ordered_sim_labels]


        fig, axs = plt.subplots(
            nrows=len(available_metrics),
            ncols=1,
            sharex=True,
            sharey=False,
            figsize=(x_length, x_length / self.phi * len(available_metrics)),
        )

        if len(available_metrics) == 1:
            axs = [axs]

        for ax, error_metric in zip(axs, available_metrics):
            long_df = long_dfs[error_metric]


            sns.boxplot(
                ax=ax,
                data=long_df,
                x="simulation",
                y=error_metric,
                hue="year",
                hue_order=year_order,
                order=sim_order_display,
                palette=sns.color_palette(palette="Blues"),
                width=0.8,
                linewidth=0.6,
                showfliers=False,
            )
            sns.despine(ax=ax, right=True, top=True)

            ax.set_xlabel("")
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(bottom=0, top=self.error_max_values[error_metric])
            ax.set_ylabel(f"{error_metric} {self.error_axis_labels[error_metric]}")

            if ax == axs[0]:
                # ax.set_title(f"All Metrics — {temporal_resolution.capitalize()}", loc="left", fontsize=13)
                ax.legend(title="Year", frameon=True)
            else:
                ax.get_legend().remove()

        plt.xticks(rotation=0, ha="center")
        plt.tight_layout()

        output_path = self.figures_dir / f"all_metric_by_simulation_and_year_{temporal_resolution}.{self.export_format}"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")


    def plot_boxplot_per_country(self, x_length=8, temporal_resolution="hourly"):
        """Create grid of boxplots showing error distributions per country for a given temporal resolution."""
        ordered_sim_labels = [s for s in self.sim_plot_order if s in self.sim_labels]


        fig, axes = plt.subplots(
            nrows=len(self.sim_labels),
            ncols=len(self.error_list),
            figsize=(x_length, x_length * self.phi),
            sharex="col",
        )

        for i, sim_label in enumerate(ordered_sim_labels):
            for j, error in enumerate(self.error_list):
                df = self.scores_dict[sim_label][error].get(temporal_resolution)
                ax = axes[i, j]

                if df is None:
                    ax.set_visible(False)
                    continue

                df_long = df.reset_index(names="year").melt(
                    id_vars="year",
                    var_name="country",
                    value_name=error,
                )

                sns.boxplot(
                    data=df_long,
                    y="country",
                    x=error,
                    ax=ax,
                    showfliers=True,
                    width=0.6,
                )

                ax.set_ylabel("")
                ax.grid(axis="x", alpha=0.4)
                ax.grid(axis="y", alpha=0.15)

                if i == 0:
                    continue
                    # ax.set_title(
                    #     f"{error.upper()} ({self.error_units[error]}) — {temporal_resolution.capitalize()}",
                    #     fontsize=11,
                    #     pad=10,
                    # )

                if j == 0:
                    display_name = self.sim_display_names.get(sim_label, sim_label)
                    ax.text(
                        -0.25,
                        0.5,
                        display_name,
                        transform=ax.transAxes,
                        fontsize=14,
                        va="center",
                        ha="right",
                        rotation=90,
                    )

        for j, error in enumerate(self.error_list):
            for i in range(len(self.sim_labels)):
                axes[i, j].set_xlim(left=0, right=self.error_max_values[error])

            top_ax = axes[0, j]
            top_ax.xaxis.tick_top()
            top_ax.tick_params(axis="x", which="both", top=True, labeltop=True, bottom=False, labelbottom=False)

            for i in range(1, len(self.sim_labels)):
                axes[i, j].tick_params(
                    axis="x", which="both", top=False, labeltop=False, bottom=False, labelbottom=False
                )

        plt.tight_layout()

        output_path = self.figures_dir / f"error_distribution_per_country_{temporal_resolution}.{self.export_format}"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")


    def plot_yearly_values_per_country(self, x_length=8, temporal_resolution="hourly"):
        """
        Create grid of scatter plots showing yearly values per country for a given temporal resolution.
        """
        ordered_sim_labels = [s for s in self.sim_plot_order if s in self.sim_labels]


        fig, axes = plt.subplots(
            nrows=len(ordered_sim_labels),
            ncols=len(self.error_list),
            figsize=(x_length, x_length * self.phi),
            sharex="col",
        )

        if len(ordered_sim_labels) == 1 and len(self.error_list) == 1:
            axes = [[axes]]
        elif len(ordered_sim_labels) == 1:
            axes = [axes]
        elif len(self.error_list) == 1:
            axes = [[ax] for ax in axes]

        legend_handles = []
        legend_labels = []

        for i, sim_label in enumerate(ordered_sim_labels):
            for j, error in enumerate(self.error_list):
                df = self.scores_dict[sim_label][error].get(temporal_resolution)
                ax = axes[i][j]

                if df is None:
                    ax.set_visible(False)
                    continue

                df_long = df.reset_index(names="year").melt(
                    id_vars="year",
                    var_name="country",
                    value_name=error,
                )

                df_long["year"] = df_long["year"].astype(str)
                year_order = sorted(df_long["year"].unique())

                palette = sns.color_palette("coolwarm", n_colors=len(year_order))
                year_color_map = dict(zip(year_order, palette))

                for year in year_order:
                    subset = df_long[df_long["year"] == year]

                    sc = ax.scatter(
                        subset[error],
                        subset["country"],
                        color=year_color_map[year],
                        s=40,
                        edgecolor="white",
                        linewidth=0.4,
                        alpha=0.9,
                        label=year if (i == 0 and j == 0) else None,
                        zorder=3,
                    )

                    if i == 0 and j == 0:
                        legend_handles.append(sc)
                        legend_labels.append(year)

                ax.grid(color="gray", linewidth=0.6, alpha=0.7, linestyle="dashed")
                ax.set_xlim(0, self.error_max_values[error])
                ax.invert_yaxis()
                ax.set_ylabel("")

                if i == 0:
                    ax.set_title(
                        f"{error.upper()} ({self.error_units[error]}) — {temporal_resolution.capitalize()}",
                        fontsize=11,
                        pad=10,
                    )

                if j == 0:
                    display_name = self.sim_display_names.get(sim_label, sim_label)
                    ax.text(
                        -0.25,
                        0.5,
                        display_name,
                        transform=ax.transAxes,
                        fontsize=11,
                        va="center",
                        ha="right",
                        rotation=90,
                    )

                if i == 0:
                    ax.xaxis.tick_top()
                    ax.tick_params(
                        axis="x", which="both", top=True, labeltop=True, bottom=False, labelbottom=False
                    )
                else:
                    ax.tick_params(
                        axis="x", which="both", top=False, labeltop=False, bottom=False, labelbottom=False
                    )

        fig.legend(
            legend_handles,
            legend_labels,
            title="Year",
            loc="lower center",
            ncol=len(legend_labels),
            frameon=True,
            bbox_to_anchor=(0.5, 0),
        )

        bottom_space = 0.035
        plt.tight_layout(rect=[0, bottom_space, 1, 1])

        output_path = self.figures_dir / f"error_yearly_values_per_country_{temporal_resolution}.{self.export_format}"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_prices(
        self, x_length=8, resampling_rule="D", countries_list=["DE", "ES", "IT", "FR", "DK", "NO"], rolling_window=None
    ):
        """Plot benchmark vs simulations per country."""

        plot_order = [
            "benchmark",
            "hindcast-std",
            "hindcast-dyn",
            "hindcast-dyn-rolling",
        ]

        for country in countries_list:
            fig, ax = plt.subplots(figsize=(x_length, x_length / self.phi))

            for label in plot_order:
                if label not in self.prices_dict:
                    continue
                df = self.prices_dict[label]
                
                if country not in df.columns:
                    continue

                series = df[country]

                if resampling_rule:
                    series = series.resample(resampling_rule).mean()

                if label == "benchmark":
                    ax.plot(series.index, series, label="Benchmark", color=self.sim_color[label])
                else:
                    ax.plot(
                        series.index,
                        series,
                        label=label,
                        color=self.sim_color[label],
                    )

                ax.legend(frameon=True)
                ax.set_title(f"{country} – Electricity Prices Daily resample", loc="left", fontsize=14, pad=20)
                ax.set_xlim(left=series.index.min(), right=series.index.max())
                # ax.set_ylim(bottom=0)
                ax.set_ylabel("EUR/MWh")
                ax.grid(True, linestyle="dashed", alpha=0.5)

            plt.tight_layout()

            output_path = self.figures_dir / f"price_{country}.{self.export_format}"
            plt.savefig(output_path)
            plt.close()

            print(f"Saved: {output_path}")

    def plot_europe_prices(
        self,
        x_length=8,
        resampling_rule="D",
        sim_labels=None,
        rolling_window=None,
        load_shedding_label="hindcast-dyn-rolling",    ):
        """
        Plot Europe reference price (europe_price_ref) and per-simulation europe_price.
        Generates two plots: filtered and unfiltered electricity prices.
        """

        if sim_labels is None:
            sim_labels = self.sim_labels

        # Loop over both filtered and unfiltered datasets
        datasets = {
            "filtered": self.prices_dict,
            "unfiltered": self.prices_unfiltered_dict,
        }

        for suffix, price_dict in datasets.items():
            # Find a europe_price_ref from the first simulation that contains it
            europe_ref_series = None
            for lab in sim_labels:
                df = price_dict.get(lab)
                if df is None:
                    continue
                if "europe_price_ref" in df.columns:
                    europe_ref_series = df["europe_price_ref"].copy()
                    break

            if europe_ref_series is None:
                print(f"No 'europe_price_ref' found in any simulation ({suffix}). Skipping plot.")
                continue

            # Prepare figure
            fig, ax = plt.subplots(figsize=(x_length, x_length / self.phi))

            # Plot reference first (resample + rolling if requested)
            ref_series = europe_ref_series
            if resampling_rule:
                ref_series = ref_series.resample(resampling_rule).mean()
            if rolling_window:
                ref_series = ref_series.rolling(rolling_window, min_periods=1, center=False).mean()

            ref_color = self.sim_color.get("benchmark", "black")
            ax.plot(ref_series.index, ref_series, label="Europe reference", color=ref_color, linewidth=2.0, zorder=3)

            # Then plot each simulation's europe_price (if present)
            plotted_any = False
            for lab in sim_labels:
                df = price_dict.get(lab)
                if df is None:
                    continue
                if "europe_price" not in df.columns:
                    continue

                s = df["europe_price"].copy()
                # align tz to UTC
                try:
                    if s.index.tz is None:
                        s.index = s.index.tz_localize("UTC")
                    else:
                        s.index = s.index.tz_convert("UTC")
                except Exception:
                    pass

                if resampling_rule:
                    s = s.resample(resampling_rule).mean()
                if rolling_window:
                    s = s.rolling(rolling_window, min_periods=1, center=False).mean()

                color = self.sim_color.get(lab, None)
                ax.plot(s.index, s, label=lab, color=color, zorder=4, linewidth=0.8)
                plotted_any = True

            if not plotted_any:
                print(f"No simulation had 'europe_price' column ({suffix}). Only reference plotted.")

            # -------------------------------------------------
            # Shade load shedding timestamps (light grey)
            # -------------------------------------------------
            if hasattr(self, "load_shedding_dict") and self.load_shedding_dict is not None:
                timestamps = self.load_shedding_dict.get(load_shedding_label)
                if timestamps is not None and len(timestamps) > 0:
                    for ts in timestamps:
                        try:
                            start = pd.to_datetime(ts)
                            if start.tzinfo is None:
                                start = start.tz_localize("UTC")
                            else:
                                start = start.tz_convert("UTC")
                        except Exception:
                            continue

                        ax.axvspan(
                            start,
                            start + pd.Timedelta(days=1),
                            color="lightgrey",
                            alpha=0.05,
                            zorder=0,
                        )

            # Final plot cosmetics
            legend_label_map = {
                "Europe reference": "Historical",
                "hindcast-std": "Hindcast-static",
                "hindcast-dyn": "Hindcast-dynamic",
                "hindcast-dyn-rolling": "Hindcast-dynamic-rolling horizon",
            }

            # Enforce desired legend order
            desired_order = [
                "Europe reference",
                "hindcast-std",
                "hindcast-dyn",
                "hindcast-dyn-rolling",
            ]

            handles, labels = ax.get_legend_handles_labels()
            handle_map = dict(zip(labels, handles))
            ordered_handles = [handle_map[k] for k in desired_order if k in handle_map]
            ordered_labels = [legend_label_map.get(k, k) for k in desired_order if k in handle_map]
            ax.legend(
                ordered_handles,
                ordered_labels,
                frameon=True,
                loc="upper center",
                ncol=2,
                bbox_to_anchor=(0.5, 1.3),
                fancybox=True,
            )

            ax.set_ylim(bottom=0, top=600)

            # Compute x limits
            xmins = [ref_series.index.min()]
            xmaxs = [ref_series.index.max()]
            for lab in sim_labels:
                df = price_dict.get(lab)
                if df is None:
                    continue
                if "europe_price" in df.columns:
                    idx = df["europe_price"].index
                    xmins.append(idx.min())
                    xmaxs.append(idx.max())
            ax.set_xlim(left=min(xmins), right=max(xmaxs))

            # X-axis: show only years, move ticks lower
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(axis="x", pad=15)  # move xticks lower

            ax.set_ylabel("EUR/MWh")
            ax.grid(True, linestyle="dashed", alpha=0.6)

            plt.tight_layout()
            output_path = self.figures_dir / f"price_europe_{suffix}.{self.export_format}"
            plt.savefig(output_path)
            plt.close()
            print(f"Saved: {output_path}")

    # -----------------------------------------------------------------------
    # Capacity comparison plots
    # -----------------------------------------------------------------------

    def plot_capacity_comparison(
        self,
        country: str,
        carrier_order: list[str] | None = None,
        ylim: tuple | None = None,
    ) -> tuple:
        """
        Grouped bar chart: ENTSO-E vs PyPSA installed capacity for one country.
        Returns ylim so it can be shared across sibling plots (e.g. DE and ES).
        """
        entsoe = self.entsoe_capacity_df.loc[country]
        pypsa  = self.pypsa_capacity_df.loc[country]

        carriers = sorted(set(entsoe.index) | set(pypsa.index))
        if carrier_order is not None:
            carriers = [c for c in carrier_order if c in carriers] + [
                c for c in carriers if c not in carrier_order
            ]

        comparison = pd.DataFrame(
            {"ENTSO-E": entsoe.reindex(carriers, fill_value=0),
             "PyPSA":   pypsa.reindex(carriers, fill_value=0)}
        )
        comparison = comparison[(comparison != 0).any(axis=1)] / 1000  # MW → GW
        comparison.index = [CAPACITY_PLOT_LABEL_MAPPING.get(c, c) for c in comparison.index]

        fig, ax = plt.subplots(figsize=(10, 10 / self.phi))
        comparison.plot(kind="bar", ax=ax, color=CAPACITY_COLORS_COUNTRY, width=0.5, edgecolor="none")

        ax.set_ylabel("Installed capacity (GW)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(axis="y", linestyle=":", linewidth=0.8)
        ax.grid(axis="x", linestyle=":", linewidth=0.8)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.legend(title="", loc="upper right", frameon=False)

        if ylim is not None:
            ax.set_ylim(ylim)
        returned_ylim = ax.get_ylim()

        plt.tight_layout()
        output_dir = self.figures_dir / "capacity_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{country}_capacity_comparison.{self.export_format}", dpi=300)
        plt.close(fig)
        print(f"Saved: {output_dir}/{country}_capacity_comparison.{self.export_format}")

        return returned_ylim

    def plot_capacity_all_countries_by_carrier(
        self,
        carrier: str,
        country_order: list[str] | None = None,
    ) -> None:
        """
        Grouped bar chart: PyPSA vs ENTSO-E installed capacity across all
        countries for a single carrier.
        """
        entsoe_df = self.entsoe_capacity_df.copy().drop(index="EU", errors="ignore")
        pypsa_df  = self.pypsa_capacity_df.copy().drop(index="EU", errors="ignore")

        if carrier not in entsoe_df.columns and carrier not in pypsa_df.columns:
            print(f"Warning: '{carrier}' not found in either capacity dataframe — skipping.")
            return

        all_countries = sorted(
            set(entsoe_df.index[entsoe_df.get(carrier, pd.Series(dtype=float)) != 0])
            | set(pypsa_df.index[pypsa_df.get(carrier, pd.Series(dtype=float)) != 0])
        )
        if country_order is not None:
            all_countries = [c for c in country_order if c in all_countries] + [
                c for c in all_countries if c not in country_order
            ]

        comparison = pd.DataFrame(
            {
                "PyPSA":   pypsa_df.get(carrier, pd.Series(0, index=pypsa_df.index)).reindex(all_countries, fill_value=0),
                "ENTSO-E": entsoe_df.get(carrier, pd.Series(0, index=entsoe_df.index)).reindex(all_countries, fill_value=0),
            },
            index=all_countries,
        )
        comparison = comparison[(comparison != 0).any(axis=1)]

        n = len(comparison)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.65), 5))
        comparison.plot(kind="bar", ax=ax, color=CAPACITY_COLORS_CARRIER, width=0.5, edgecolor="none")

        ax.set_ylabel("Installed capacity (MW)")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(axis="y", linestyle=":", linewidth=0.8)
        ax.grid(axis="x", linestyle=":", linewidth=0.8)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.legend(title="", loc="upper right", frameon=False)

        plt.tight_layout()
        output_dir = self.figures_dir / "capacity_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_carrier = carrier.replace("/", "-").replace(" ", "_")
        plt.savefig(output_dir / f"{safe_carrier}_capacity_comparison.{self.export_format}", dpi=300)
        plt.close(fig)
        print(f"Saved: {output_dir}/{safe_carrier}_capacity_comparison.{self.export_format}")

    def plot_all_capacity_comparisons(self) -> None:
        """
        Generate the full set of capacity comparison figures.
        Skips gracefully if either capacity CSV was not found at load time.
        """
        if self.entsoe_capacity_df is None or self.pypsa_capacity_df is None:
            print("Skipping capacity comparison plots — one or both data files missing.")
            return

        print("\n--- Generating capacity comparison plots ---")

        # Per-country: DE and ES share a y-axis, others are independent
        ylim = self.plot_capacity_comparison("DE", carrier_order=CAPACITY_CARRIER_ORDER)
        self.plot_capacity_comparison("ES", carrier_order=CAPACITY_CARRIER_ORDER, ylim=ylim)
        for country in ["NO", "FR", "IT", "DK", "EU"]:
            self.plot_capacity_comparison(country, carrier_order=CAPACITY_CARRIER_ORDER)

        # Cross-country per carrier
        for carrier in [
            "Combined-Cycle Gas",
            "oil",
            "lignite",
            "coal",
            "nuclear",
            "biomass",
            "Solar",
            "Offshore Wind (AC)",
            "Onshore Wind",
            "Pumped Hydro Storage",
            "Run of River",
            "Reservoir & Dam",
        ]:
            self.plot_capacity_all_countries_by_carrier(carrier)



    def plot_price_dispatch_compare(self):
        data_dir = self.data_dir
        figure_dir = self.figures_dir

        # This script applies mapping twice. First to PyPSA carriers and later to ENTSO-E data. This gives the common names.
        # This is the "nice names"
        mapping_pypsa_carriers = {
            "Combined-Cycle Gas": "Gas",
            "Load shedding": None,
            "Offshore Wind (AC)": "Wind Offshore",
            "Offshore Wind (DC)": "Wind Offshore",
            "Onshore Wind": "Wind Onshore",
            "Open-Cycle Gas": "Gas",
            "Run of River": "Run-of-river",
            "Solar": "Solar",
            "biomass": "Biomass",
            "nuclear": "Nuclear",
            "coal": "Coal",
            "lignite": "Lignite",
            "oil": "Oil",
            "solar-hsat": "Solar",
            "Pumped Hydro Storage": "Pumped Hydro Storage",
            "Reservoir & Dam": "Reservoir & Dam",
            "geothermal": "Geothermal"
        }

        start_date = (pd.Timestamp("2022-01-01"),)
        end_date = (pd.Timestamp("2022-12-31 23:00"),)


        countries_include = ["DE","ES"]

        resampling_rule = "W-MON"  # Weekly, starting on Monday

        x_length = 6

        phi = 1.618


        n = pypsa.Network(data_dir / "network_files" / "hindcast-dyn-rolling-wy2022.nc")

        # Color mapping for plotting

        color_dict = n.carriers["color"].to_dict()

        mapping_color = {
            "offwind-ac": "Wind Offshore",
            "offwind-dc": "Wind Offshore",
            "onwind": "Wind Onshore",
            "solar": "Solar",
            "solar-hsat": "Solar",
            "CCGT": "Gas",
            "OCGT": "Gas",
            "PHS": "Pumped Hydro Storage",
            "biomass": "Biomass",
            "coal": "Coal",
            "geothermal": "Geothermal",
            "hydro": "Reservoir & Dam",
            "lignite": "Lignite",
            "nuclear": "Nuclear",
            "oil": "Oil",
            "ror": "Run-of-river",
        }

        color_dict_new = {}

        for original_name, new_name in mapping_color.items():
            # Use the color from the original carrier
            if original_name in color_dict:
                color_dict_new[new_name] = color_dict[original_name]


        color_dict_new['Other'] = 'tab:red'

        for country in countries_include:
            supply_stats = n.statistics.supply(groupby=["bus", "carrier"], groupby_time=False) / 1000
            country_supply = supply_stats.xs(country, level="bus")
            country_generators_and_storage = country_supply.loc[["Generator", "StorageUnit"], :]
            country_dispatch_raw = country_generators_and_storage.T.droplevel(0, axis=1)

            ignored_carriers = [
                col for col, target in mapping_pypsa_carriers.items() if target is None and col in country_dispatch_raw.columns
            ]
            country_dispatch_raw = country_dispatch_raw.drop(columns=ignored_carriers)

            carrier_rename = {
                col: target
                for col, target in mapping_pypsa_carriers.items()
                if target is not None and col in country_dispatch_raw.columns
            }
            country_dispatch = country_dispatch_raw.rename(columns=carrier_rename)
            country_dispatch = country_dispatch.T.groupby(level=0).sum().T  # merg


            if resampling_rule:
                country_dispatch_res = country_dispatch.resample(resampling_rule).sum()
            else:
                country_dispatch_res = country_dispatch

            withdrawal_stats = n.statistics.withdrawal(groupby=["bus", "carrier"], groupby_time=False) / 1000 # Converted to GWh
            country_withdrawal = withdrawal_stats.xs(country, level="bus")
            country_generators_and_storage = country_withdrawal.loc[["Generator", "StorageUnit"], :]
            country_consumption_raw = country_generators_and_storage.T.droplevel(0, axis=1)

            carrier_rename = {
                col: target
                for col, target in mapping_pypsa_carriers.items()
                if target is not None and col in country_consumption_raw.columns
            }
            country_consumption = country_consumption_raw.rename(columns=carrier_rename)
            country_consumption = country_consumption.T.groupby(level=0).sum().T  # merg


            keek_withdrawal = ["Pumped Hydro Storage"]
            # Drop all other columns except the ones in keek_withdrawal
            country_consumption_clean = country_consumption[keek_withdrawal]
            country_consumption_clean


            if resampling_rule:
                country_consumption_res = country_consumption_clean.resample(resampling_rule).sum()
            else:
                country_consumption_res = country_consumption_clean


            # Entso-e data
            entsoe_raw = pd.read_csv(
            data_dir / "generation" / f"generation_{country}_hourly_data.csv",
            index_col=0,
            parse_dates=True,
            )

            entsoe_filtered_raw = entsoe_raw[pd.Timestamp(start_date[0], tz="UTC") : pd.Timestamp(end_date[0], tz="UTC")]

            #Mapping depend on entso-e reporting
            if "Actual Aggregated" in str(entsoe_filtered_raw.columns):
                print("agg")
                keep_cols = [
                    "('Biomass', 'Actual Aggregated')",
                    "('Fossil Brown coal/Lignite', 'Actual Aggregated')",
                    "('Fossil Coal-derived gas', 'Actual Aggregated')",
                    "('Fossil Gas', 'Actual Aggregated')",
                    "('Fossil Hard coal', 'Actual Aggregated')",
                    "('Fossil Oil', 'Actual Aggregated')",
                    "('Geothermal', 'Actual Aggregated')",
                    "('Hydro Pumped Storage', 'Actual Aggregated')",
                    #"('Hydro Pumped Storage', 'Actual Consumption')",
                    "('Hydro Run-of-river and poundage', 'Actual Aggregated')",
                    "('Hydro Water Reservoir', 'Actual Aggregated')",
                    "('Nuclear', 'Actual Aggregated')",
                    "('Other', 'Actual Aggregated')",
                    "('Other renewable', 'Actual Aggregated')",
                    "('Solar', 'Actual Aggregated')",
                    "('Waste', 'Actual Aggregated')",
                    "('Wind Offshore', 'Actual Aggregated')",
                    "('Wind Onshore', 'Actual Aggregated')",
                ]
                


                keep_cols_existing = [c for c in keep_cols if c in entsoe_filtered_raw.columns]

                other_cols = [c for c in entsoe_filtered_raw.columns if "(" not in c]

                final_cols = list(set(keep_cols_existing + other_cols))

                entsoe_filtered = entsoe_filtered_raw[final_cols].copy()
                entsoe_filtered = entsoe_filtered.drop(columns=["Energy storage"], errors="ignore")
            

                mapping_entsoe_pypsa = {
                    "('Biomass', 'Actual Aggregated')": "Biomass",
                    "('Fossil Brown coal/Lignite', 'Actual Aggregated')": "Lignite",
                    "('Fossil Coal-derived gas', 'Actual Aggregated')": "Gas",
                    "('Fossil Gas', 'Actual Aggregated')": "Gas",
                    "('Fossil Hard coal', 'Actual Aggregated')": "Coal",
                    "('Fossil Oil', 'Actual Aggregated')": "Oil",
                    "('Geothermal', 'Actual Aggregated')": "Geothermal",
                    "('Hydro Pumped Storage', 'Actual Aggregated')": "Pumped Hydro Storage",
                    #"('Hydro Pumped Storage', 'Actual Consumption')": ,
                    "('Hydro Run-of-river and poundage', 'Actual Aggregated')": "Run-of-river",
                    "('Hydro Water Reservoir', 'Actual Aggregated')": "Reservoir & Dam",
                    "('Nuclear', 'Actual Aggregated')": "Nuclear",
                    "('Other', 'Actual Aggregated')": "Other",
                    "('Other renewable', 'Actual Aggregated')": "Other",
                    "('Solar', 'Actual Aggregated')": "Solar",
                    "('Waste', 'Actual Aggregated')": "Biomass",
                    "('Wind Offshore', 'Actual Aggregated')": "Wind Offshore",
                    "('Wind Onshore', 'Actual Aggregated')": "Wind Onshore",
                }

                entsoe_clean = (entsoe_filtered.rename(columns=mapping_entsoe_pypsa) /1000).copy() # convert to GWh
                entsoe_clean = entsoe_clean.T.groupby(level=0).sum().T


            else:
                print("else")
                mapping_entsoe_pypsa = {
                "Biomass": "Biomass",
                "Fossil Brown coal/Lignite": "Lignite",
                "Fossil Coal-derived gas": "Gas",
                "Fossil Gas": "Gas",
                "Fossil Hard coal": "Coal",
                "Fossil Oil": "Oil",
                "Fossil Oil shale": "Oil",
                "Fossil Peat": "Lignite",
                "Geothermal": "Geothermal",
                "Hydro Pumped Storage Net": "Pumped Hydro Storage",
                "Hydro Run-of-river and poundage": "Run-of-river",
                "Hydro Water Reservoir": "Reservoir & Dam",
                "Marine": "Other",
                "Nuclear": "Nuclear",
                "Other": "Other",
                "Other renewable": "Other",
                "Solar": "Solar",
                "Waste": "Biomass",
                "Wind Offshore": "Wind Offshore",
                "Wind Onshore": "Wind Onshore",
            }

                
                entsoe_filtered = entsoe_filtered_raw.rename(columns=mapping_entsoe_pypsa) 
                entsoe_filtered = entsoe_filtered.drop(columns=["Energy storage"], errors="ignore")
                entsoe_clean = (entsoe_filtered.T.groupby(level=0).sum().T)/1000 # convert to GWh
                

            #Additional mapping required for contries that report in mixed formes
            mapping_entsoe_pypsa = {
                "Biomass": "Biomass",
                "Fossil Brown coal/Lignite": "Lignite",
                "Fossil Coal-derived gas": "Gas",
                "Fossil Gas": "Gas",
                "Fossil Hard coal": "Coal",
                "Fossil Oil": "Oil",
                "Fossil Oil shale": "Oil",
                "Fossil Peat": "Lignite",
                "Geothermal": "Geothermal",
                "Hydro Pumped Storage Net": "Pumped Hydro Storage",
                "Hydro Pumped Storage": "Pumped Hydro Storage",
                "Hydro Run-of-river and poundage": "Run-of-river",
                "Hydro Water Reservoir": "Reservoir & Dam",
                "Marine": "Other",
                "Nuclear": "Nuclear",
                "Other": "Other",
                "Other renewable": "Other",
                "Solar": "Solar",
                "Waste": "Biomass",
                "Wind Offshore": "Wind Offshore",
                "Wind Onshore": "Wind Onshore",
                }

            mapping_existing = {k: v for k, v in mapping_entsoe_pypsa.items() if k in entsoe_clean.columns}

            entsoe_filtered_v2 = entsoe_clean.rename(columns=mapping_existing).copy()
            entsoe_clean = entsoe_filtered_v2.T.groupby(level=0).sum().T.copy()


            # Test how ENTSO-E reports 
            agg_col = "('Hydro Pumped Storage', 'Actual Aggregated')"
            cons_col = "('Hydro Pumped Storage', 'Actual Consumption')"
            net_col = "Hydro Pumped Storage Net"
            base_col = "Hydro Pumped Storage"

            if agg_col in entsoe_filtered_raw.columns and cons_col in entsoe_filtered_raw.columns and entsoe_filtered_raw["('Hydro Pumped Storage', 'Actual Aggregated')"].sum()>0:
                print("agg")
                # Calculate the net dispatch for Pumped Hydro Storage
                #country_dispatch_res["Pumped Hydro Storage"] = country_dispatch_res["Pumped Hydro Storage"] - country_consumption_res["Pumped Hydro Storage"]
                # Clip all negative values to zero (since we are only interested in the dispatch, not the consumption)
                #country_dispatch_res["Pumped Hydro Storage"] = country_dispatch_res["Pumped Hydro Storage"].clip(lower=0)
                country_dispatch["Pumped Hydro Storage"] = country_dispatch["Pumped Hydro Storage"].clip(lower=0)
                entsoe_clean["Pumped Hydro Storage"]=entsoe_clean["Pumped Hydro Storage"].clip(lower=0)
            elif net_col in entsoe_filtered_raw.columns:
                # Calculate the net dispatch for Pumped Hydro Storage
                #country_dispatch_res["Pumped Hydro Storage"] = country_dispatch_res["Pumped Hydro Storage"] - country_consumption_res["Pumped Hydro Storage"]
                country_dispatch["Pumped Hydro Storage"] = country_dispatch["Pumped Hydro Storage"] - country_consumption["Pumped Hydro Storage"]
                # Clip all negative values to zero (since we are only interested in the dispatch, not the consumption)
                country_dispatch["Pumped Hydro Storage"] = country_dispatch["Pumped Hydro Storage"].clip(lower=0)
                entsoe_clean["Pumped Hydro Storage"]=entsoe_clean["Pumped Hydro Storage"].clip(lower=0)
                print("net")
            elif base_col in entsoe_filtered_raw.columns:
                print("base")
                # Calculate the net dispatch for Pumped Hydro Storage
                #country_dispatch_res["Pumped Hydro Storage"] = country_dispatch_res["Pumped Hydro Storage"] - country_consumption_res["Pumped Hydro Storage"]
                # Clip all negative values to zero (since we are only interested in the dispatch, not the consumption)
                #country_dispatch_res["Pumped Hydro Storage"] = country_dispatch_res["Pumped Hydro Storage"].clip(lower=0)
                country_dispatch["Pumped Hydro Storage"] = country_dispatch["Pumped Hydro Storage"].clip(lower=0)
                entsoe_clean["Pumped Hydro Storage"]=entsoe_clean["Pumped Hydro Storage"].clip(lower=0)

            entsoe_clean = entsoe_clean.tz_localize(None)

            unique_cols = list(dict.fromkeys(entsoe_clean.columns.tolist()))
            clean_data = {col: entsoe_clean[[col]].sum(axis=1) for col in unique_cols}
            entsoe_dedup = pd.DataFrame(clean_data, index=entsoe_clean.index)

            diff = entsoe_dedup.copy()
            for col in diff.columns:
                if col in country_dispatch.columns:
                    diff[col] = entsoe_dedup[col] - country_dispatch[col]
                # else: leave as entsoe value untouched

            diff_res = diff.resample(resampling_rule).sum()


            # Get prices

            prices_dict = {}


            sim_labels = {'hindcast-std',
                        'hindcast-dyn',
                        'hindcast-dyn-rolling'}
            # Load simulation
            for sim_label in sim_labels:
                #file_dir = Path('../results_concat') / sim_label / 'electricity_prices' / "combined_electricity_prices.csv"
                file_dir = self.results_concat_dir / sim_label / 'electricity_prices' / "combined_electricity_prices.csv"

                df_sim = pd.read_csv(file_dir, index_col=0, parse_dates=True)
                df_sim = df_sim[df_sim.index.year.isin([2022])]

                prices_dict[sim_label] = df_sim

            # Load benchmark
            benchmark_path = data_dir / "benchmark" / "electricity_prices.csv"
            df_bench = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)

            if df_bench.index.tz is None:
                df_bench.index = df_bench.index.tz_localize("UTC")
            else:
                df_bench.index = df_bench.index.tz_convert("UTC")

            df_bench = df_bench.interpolate().ffill().bfill()

            df_bench = df_bench[df_bench.index.year.isin([2022])]

            prices_dict["benchmark"] = df_bench
            
            #Plot

          
            # ── Figure ─────────────────────────────────────────────────────────────────────
            phi = 1.618
            fig, axs = plt.subplots(
                ncols=1,
                nrows=3,
                figsize=(x_length, x_length * phi * 0.8),
                sharex=False,
            )

            # ── Global style (mirrors setup_style) ────────────────────────────────────────
            plt.style.use("seaborn-v0_8-whitegrid")
            mpl.rcParams["axes.spines.right"]   = False
            mpl.rcParams["axes.spines.top"]     = False
            plt.rcParams["font.family"]         = "Arial"
            plt.rcParams["axes.titleweight"]    = "bold"
            plt.rcParams["axes.labelsize"]      = 16
            plt.rcParams["xtick.labelsize"]     = 16
            plt.rcParams["ytick.labelsize"]     = 16
            plt.rcParams["legend.fontsize"]     = 16
            plt.rcParams["legend.title_fontsize"] = 16

            # ── ax[0] : Prices ─────────────────────────────────────────────────────────────
            plot_order = [
                "benchmark",
                "hindcast-std",
                "hindcast-dyn",
                "hindcast-dyn-rolling",
            ]

            sim_color = {
                "benchmark":            "#56b4e9",
                "hindcast-dyn":         "#009e73",
                "hindcast-dyn-rolling": "#d55e00",
                "hindcast-std":         "#e69f00",
            }

            for label in plot_order:
                if label not in prices_dict:
                    continue
                df = prices_dict[label]
                if country not in df.columns:
                    continue
                series = df[country]
                if resampling_rule:
                    series = series.resample("D").mean()
                axs[0].plot(series, label=label, color=sim_color.get(label, None))

            axs[0].set_xlim(left=start_date, right=end_date)
            axs[0].set_ylim(bottom=0, top=720)
            axs[0].set_ylabel("Prices [EUR/MWh]")
            axs[0].yaxis.grid(True, linestyle="dashed", linewidth=0.5, alpha=0.4, zorder=0)
            axs[0].xaxis.grid(False)
            axs[0].set_axisbelow(True)
            axs[0].set_xticks([])
            axs[0].tick_params(axis="y", which="both", left=False)

            # ── ax[1] : Dispatch ───────────────────────────────────────────────────────────
            carrier_order = [
                "Lignite",
                "Nuclear",
                "Coal",
                "Gas",
                "Oil",
                "Biomass",
                "Reservoir & Dam",
                "Pumped Hydro Storage",
                "Run-of-river",
                "Wind Offshore",
                "Wind Onshore",
                "Solar",
            ]

            def reorder_columns(df, order):
                existing  = [c for c in order if c in df.columns]
                remaining = [c for c in df.columns if c not in existing]
                return df[existing + remaining]

            country_dispatch_res_twh = reorder_columns(country_dispatch_res, carrier_order) / 1000
            country_dispatch_res_twh.plot.bar(
                stacked=True, width=1.0, ax=axs[1], legend=False,
                color=color_dict_new,
            )
            axs[1].set_ylabel("Weekly dispatch\n[TWh]")
            axs[1].set_xlabel("")
            axs[1].yaxis.grid(True, linestyle="dashed", linewidth=0.5, alpha=0.4, zorder=0)
            axs[1].xaxis.grid(False)
            axs[1].set_axisbelow(True)
            axs[1].set_xticks([])
            axs[1].yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
            axs[1].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
            axs[1].tick_params(axis="y", which="both", left=False)

            # ── ax[2] : Dispatch difference ────────────────────────────────────────────────
            diff_res_plot = diff_res.copy() / 1000 # Convert to TWh
            diff_timestamps = diff_res_plot.index
            diff_res_plot = diff_res_plot.reset_index(drop=True)

            diff_res_plot.plot.bar(
                stacked=True, width=1.0, ax=axs[2], legend=False,
                color=color_dict_new,
            )
            axs[2].set_xlabel("")
            axs[2].set_ylabel("Weekly dispatch\ndifference [TWh]")
            axs[2].yaxis.grid(True, linestyle="dashed", linewidth=0.5, alpha=0.4, zorder=0)
            axs[2].xaxis.grid(False)
            axs[2].set_axisbelow(True)
            axs[2].set_ylim(bottom=-6, top=6)
            axs[2].yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
            axs[2].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
            axs[2].tick_params(axis="y", which="both", left=False)
            axs[2].axhline(0, color="black", linewidth=0.6, zorder=3)

            # Monthly ticks
            tick_positions = []
            tick_labels    = []
            prev_month     = None
            for i, ts in enumerate(diff_timestamps):
                if ts.month != prev_month:
                    tick_positions.append(i)
                    tick_labels.append(
                        ts.strftime("%b %Y") if (ts.month == 1 or prev_month is None)
                        else ts.strftime("%b")
                    )
                    prev_month = ts.month
            axs[2].set_xticks(tick_positions)
            axs[2].set_xticklabels(tick_labels, rotation=45, ha="right")
            axs[2].tick_params(axis="x", length=3)

            # ── Shared spine cleanup ────────────────────────────────────────────────────────
            for ax in axs:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)


            plt.tight_layout()
            # Save the figure
            file_path = figure_dir / f"price_dispatch_comparison_{country}.pdf"
            plt.savefig(file_path, bbox_inches="tight")
        

    def generate_all_plots(self):
        """Generate all plots."""
        print("Generating plots...")

        temporal_resolutions = ["hourly", "daily", "weekly"]

        for temporal_resolution in temporal_resolutions:
            print(f"\n--- Generating plots for temporal resolution: {temporal_resolution} ---")

            self.plot_boxplot_per_country(temporal_resolution=temporal_resolution)
            self.plot_yearly_values_per_country(temporal_resolution=temporal_resolution)
            self.plot_error_by_simulation_and_year_all(x_length=6, temporal_resolution=temporal_resolution)

            for error_metric in self.error_list:
                self.plot_error_by_simulation_and_year(
                    error_metric, x_length=7, temporal_resolution=temporal_resolution
                )

        # Price plots (not temporal-resolution-specific)
        self.plot_prices()
        self.plot_europe_prices()
        self.plot_all_capacity_comparisons()
        self.plot_price_dispatch_compare()

        print("\nAll plots generated successfully!")


@hydra.main(
    version_base=None,
    config_name="default_config",
    config_path="../../configs",
)
def main(cfg: DictConfig):
    """Main entry point for Hydra."""
    plotter = ResultsPlotter(cfg)
    plotter.generate_all_plots()
    return plotter


if __name__ == "__main__":
    main()
