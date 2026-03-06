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


class ResultsPlotter:
    """Class to handle loading and plotting of simulation results."""

    def __init__(self, cfg: DictConfig):
        # Config and directories
        self.cfg = cfg
        self.root_dir = Path(cfg.paths.root)
        self.data_dir = Path(cfg.paths.data)
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

    def plot_DE_ES_plot(
        self,
        x_length: float = 8,
        resampling_rule: str | None = "W",
        countries_list: list[str] = None,
        start_date: pd.Timestamp = "2022-01-01",
        end_date: pd.Timestamp = "2023-01-01",
        network_file="hindcast-dyn-rolling-wy2022.nc",
        ):
        """Plot benchmark vs simulations per country."""

        if countries_list is None:
            countries_list = ["DE", "ES"]

        plot_order = [
            "benchmark",
            "hindcast-std",
            "hindcast-dyn",
            "hindcast-dyn-rolling",
        ]

        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)


        # Mapping PyPSA carriers -> standardised names
        mapping_pypsa_carriers = {
            "Combined-Cycle Gas": "Fossil Gas",
            "Load shedding": None,
            "Offshore Wind (AC)": "Wind Offshore",
            "Offshore Wind (DC)": "Wind Offshore",
            "Onshore Wind": "Wind Onshore",
            "Open-Cycle Gas": "Fossil Gas",
            "Run of River": "Hydro Run-of-river and poundage",
            "Solar": "Solar",
            "biomass": "Biomass",
            "nuclear": "Nuclear",
            "coal": "Fossil Hard coal",
            "lignite": "Fossil Brown coal/Lignite",
            "oil": "Fossil Oil",
            "solar-hsat": "Solar",
            "Pumped Hydro Storage": "Hydro Pumped Storage",
            "Reservoir & Dam": "Hydro Water Reservoir",
        }

        for country in countries_list:
            fig, axs = plt.subplots(
                ncols=1,
                nrows=3,
                figsize=(x_length, x_length * self.phi),
                sharex=True,  # x-axis differs between bar and line plots
            )

            for label in plot_order:
                if label not in self.prices_dict:
                    continue
                df = self.prices_dict[label]

                if df.index.tz is not None:
                    _start = start_date.tz_localize("UTC") if start_date.tzinfo is None else start_date
                    _end   = end_date.tz_localize("UTC")   if end_date.tzinfo is None   else end_date
                else:
                    _start = start_date.tz_localize(None) if start_date.tzinfo is not None else start_date
                    _end   = end_date.tz_localize(None)   if end_date.tzinfo is not None   else end_date

                df = df[(_start <= df.index) & (df.index <= _end)]
                if country not in df.columns:
                    continue

                series = df[country]
                if resampling_rule:
                    series = series.resample(resampling_rule).mean()

                axs[0].plot(series.index, series, label=label, color=self.sim_color.get(label, None))

            legend_label_map = {
                "benchmark":            "Historical",
                "hindcast-std":         "Hindcast-static",
                "hindcast-dyn":         "Hindcast-dynamic",
                "hindcast-dyn-rolling": "Hindcast-dynamic-rolling horizon",
            }
            desired_order = ["benchmark", "hindcast-std", "hindcast-dyn", "hindcast-dyn-rolling"]
            handles, labels = axs[0].get_legend_handles_labels()
            handle_map      = dict(zip(labels, handles))
            ordered_handles = [handle_map[k] for k in desired_order if k in handle_map]
            ordered_labels  = [legend_label_map.get(k, k) for k in desired_order if k in handle_map]

            axs[0].legend(ordered_handles, ordered_labels, frameon=True,
                        loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.3), fancybox=True)
            axs[0].set_xlim(left=start_date, right=end_date)
            axs[0].set_ylim(bottom=0, top=650)
            axs[0].set_ylabel("EUR/MWh")
            axs[0].grid(True, linestyle="dashed", alpha=0.5)

            # Middle plot
            n = pypsa.Network(self.data_dir / "network_files" / network_file)

            supply_stats = n.statistics.supply(groupby=["bus", "carrier"], groupby_time=False) / 1000
            country_supply = supply_stats.xs(country, level="bus")
            country_generators_and_storage = country_supply.loc[["Generator", "StorageUnit"], :]
            country_dispatch_raw = country_generators_and_storage.T.droplevel(0, axis=1)

            ignored_carriers = [col for col, target in mapping_pypsa_carriers.items()
                                if target is None and col in country_dispatch_raw.columns]
            country_dispatch_raw = country_dispatch_raw.drop(columns=ignored_carriers)

            carrier_rename = {col: target for col, target in mapping_pypsa_carriers.items()
                            if target is not None and col in country_dispatch_raw.columns}
            country_dispatch = country_dispatch_raw.rename(columns=carrier_rename)
            country_dispatch = country_dispatch.T.groupby(level=0).sum().T  # merge duplicate cols

            if resampling_rule:
                country_dispatch_res = country_dispatch.resample(resampling_rule).sum()
            else:
                country_dispatch_res = country_dispatch

            country_dispatch_res.plot.bar(stacked=True, width=1.0, ax=axs[1])
            axs[1].set_ylabel(f"PyPSA dispatch resampled {resampling_rule} [GWh]")
            axs[1].grid(False)
            axs[1].yaxis.grid(True, linestyle='-', linewidth=0.8)
            axs[1].xaxis.grid(False)

            # Bottom plot

            entsoe_raw = pd.read_csv(
                self.data_dir / "generation" / f"generation_{country}_hourly_data.csv",
                index_col=0,
                parse_dates=True,
            )

            # Timezone alignment
            if entsoe_raw.index.tz is not None:
                _start = start_date.tz_localize("UTC") if start_date.tzinfo is None else start_date
                _end   = end_date.tz_localize("UTC")   if end_date.tzinfo is None   else end_date
            else:
                _start = start_date.tz_localize(None) if start_date.tzinfo is not None else start_date
                _end   = end_date.tz_localize(None)   if end_date.tzinfo is not None   else end_date

            entsoe_filtered = entsoe_raw[(_start <= entsoe_raw.index) & (entsoe_raw.index <= _end)]

            # Parse tuple-like column names -> keep only 'Actual Aggregated', except Pumped Hydro
            def parse_entsoe_col(col_str):
                """Return (carrier, aggregation_type) from a string like \"('Fossil Gas', 'Actual Aggregated')\"."""
                col_str = col_str.strip().strip("'\"() ")
                parts   = [p.strip().strip("'\" ") for p in col_str.split(",", 1)]
                return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "")

            # Build clean ENTSO-E dataframe with standardised carrier columns
            entsoe_carriers = {}  # carrier_name -> list of series to sum

            for col in entsoe_filtered.columns:
                carrier, agg_type = parse_entsoe_col(col)

                if carrier == "Hydro Pumped Storage":
                    # Keep charge and discharge separated
                    if "Aggregated" in agg_type:
                        label_col = "Hydro Pumped Storage (generation)"
                    elif "Consumption" in agg_type:
                        label_col = "Hydro Pumped Storage (consumption)"
                    else:
                        continue
                else:
                    # For all other carriers keep only Actual Aggregated
                    if "Aggregated" not in agg_type:
                        continue
                    label_col = carrier  # use the standardised ENTSO-E carrier name directly

                entsoe_carriers.setdefault(label_col, []).append(entsoe_filtered[col])

            entsoe_dispatch = pd.DataFrame(
                {col: pd.concat(series_list, axis=1).sum(axis=1)
                for col, series_list in entsoe_carriers.items()}
            )

            # Convert MW -> GWh (hourly data: MWh per hour / 1000)
            entsoe_dispatch = entsoe_dispatch / 1000

            if resampling_rule:
                entsoe_dispatch_res = entsoe_dispatch.resample(resampling_rule).sum()
            else:
                entsoe_dispatch_res = entsoe_dispatch

            # Align columns: map ENTSO-E standard names -> PyPSA standard names for comparison
            # (Pumped Hydro generation maps to the same carrier; consumption kept separate)
            entsoe_to_pypsa_name = {
                "Biomass":                              "Biomass",
                "Fossil Brown coal/Lignite":            "Fossil Brown coal/Lignite",
                "Fossil Coal-derived gas":              "Fossil Hard coal",
                "Fossil Gas":                           "Fossil Gas",
                "Fossil Hard coal":                     "Fossil Hard coal",
                "Fossil Oil":                           "Fossil Oil",
                "Fossil Oil shale":                     "Fossil Oil",
                "Fossil Peat":                          "Fossil Hard coal",
                "Geothermal":                           "Geothermal",
                "Hydro Pumped Storage (generation)":    "Hydro Pumped Storage",
                "Hydro Pumped Storage (consumption)":   "Hydro Pumped Storage (consumption)",
                "Hydro Run-of-river and poundage":      "Hydro Run-of-river and poundage",
                "Hydro Water Reservoir":                "Hydro Water Reservoir",
                "Nuclear":                              "Nuclear",
                "Other":                                "Other",
                "Other renewable":                      "Other",
                "Solar":                                "Solar",
                "Waste":                                "Biomass",
                "Wind Offshore":                        "Wind Offshore",
                "Wind Onshore":                         "Wind Onshore",
            }

            entsoe_dispatch_renamed = entsoe_dispatch_res.rename(columns=entsoe_to_pypsa_name)
            entsoe_dispatch_renamed = entsoe_dispatch_renamed.T.groupby(level=0).sum().T  # merge duplicates

            # Align both dataframes to the same columns and index
            common_carriers = country_dispatch_res.columns.intersection(entsoe_dispatch_renamed.columns)
            pypsa_aligned   = country_dispatch_res[common_carriers].reindex(entsoe_dispatch_renamed.index)
            entsoe_aligned  = entsoe_dispatch_renamed[common_carriers]

            dispatch_diff = entsoe_aligned - pypsa_aligned  # ENTSO-E minus PyPSA

            # Plot diff with same monthly tick logic
            dispatch_diff.plot.bar(stacked=True, width=1.0, ax=axs[2], legend=True)

            monthly_ticks_bottom = {}
            for i, ts in enumerate(dispatch_diff.index):
                month_key = (ts.year, ts.month)
                if month_key not in monthly_ticks_bottom:
                    monthly_ticks_bottom[month_key] = (i, ts.strftime("%b"))

            tick_positions_bottom = [pos for pos, _ in monthly_ticks_bottom.values()]
            tick_labels_bottom    = [lbl for _, lbl in monthly_ticks_bottom.values()]

            #axs[2].set_xticks(tick_positions_bottom)
            #axs[2].set_xticklabels(tick_labels_bottom, rotation=45, ha="right", fontsize=8)
            #axs[2].axhline(0, color="black", linewidth=0.8)
            #axs[2].set_ylabel("ENTSO-E − PyPSA [GWh]")
            #axs[2].set_xlabel("Date")
            #axs[2].grid(False)
            #axs[2].yaxis.grid(True, linestyle="-", linewidth=0.8)

            plt.tight_layout()
            output_path = self.figures_dir / f"that_plot_{country}.{self.export_format}"
            plt.savefig(output_path)
            plt.close(fig)

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
