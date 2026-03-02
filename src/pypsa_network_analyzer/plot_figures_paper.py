import hydra
from omegaconf import DictConfig
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import seaborn as sns


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
        self.error_max_values = {"mae": 250, "rmse": 300, "smape": 150}
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
        plt.rcParams["axes.labelsize"] = 18
        plt.rcParams["xtick.labelsize"] = 18
        plt.rcParams["ytick.labelsize"] = 18
        plt.rcParams["legend.fontsize"] = 16

        self.sim_color = {
            "benchmark": self.nature_sky_blue,
            "hindcast-dyn": self.nature_bluish_green,
            "hindcast-dyn-rolling": self.nature_vermillion,
            "hindcast-std": self.nature_orange,
        }

    def load_scores(self):
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
                self.results_concat_dir / sim_label / self.load_shed_name / f"combined_{self.load_shed_name}.csv"
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

    def plot_error_by_simulation_and_year(self, error_metric, x_length=8):
        """Create boxplot showing MAE by simulation and year."""
        error_metric = error_metric
        records = []

        # Loop over simulation groups
        for sim_name in self.sim_labels:
            csv_path = (
                self.results_concat_dir / sim_name / self.benchmark_name / "scores" / f"scores_{error_metric}.csv"
            )

            if not csv_path.exists():
                raise FileNotFoundError(f"Missing file: {csv_path}")

            # Rows = years, columns = countries
            df = pd.read_csv(csv_path, index_col=0)

            # Long / tidy format
            df_long = df.reset_index(names="year").melt(
                id_vars="year",
                var_name="country",
                value_name=error_metric,
            )

            df_long["simulation"] = sim_name
            records.append(df_long)

        # Concatenate all simulations
        long_df = pd.concat(records, ignore_index=True)

        # Ensure year is treated as categorical and ordered
        long_df["year"] = long_df["year"].astype(str)
        year_order = sorted(long_df["year"].unique())

        # Plot
        plt.figure(figsize=(x_length, x_length / self.phi))

        sns.boxplot(
            data=long_df,
            x="simulation",
            y=error_metric,
            hue="year",
            hue_order=year_order,
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

        plt.legend(frameon=True)
        plt.tight_layout()

        output_path = self.figures_dir / f"{error_metric}_by_simulation_and_year.{self.export_format}"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_error_by_simulation_and_year_all(self, x_length=12):
        """Create boxplot showing all error metrics by simulation and year."""

        # Build long dataframes for each error metric
        long_dfs = {}
        for error_metric in self.error_list:
            records = []
            for sim_name in self.sim_labels:
                csv_path = (
                    self.results_concat_dir / sim_name / self.benchmark_name / "scores" / f"scores_{error_metric}.csv"
                )

                if not csv_path.exists():
                    raise FileNotFoundError(f"Missing file: {csv_path}")

                df = pd.read_csv(csv_path, index_col=0)
                df_long = df.reset_index(names="year").melt(
                    id_vars="year",
                    var_name="country",
                    value_name=error_metric,
                )
                df_long["simulation"] = sim_name
                records.append(df_long)

            long_df = pd.concat(records, ignore_index=True)
            long_df["year"] = long_df["year"].astype(str)
            long_dfs[error_metric] = long_df

        year_order = sorted(long_dfs[self.error_list[0]]["year"].unique())

        fig, axs = plt.subplots(
            nrows=len(self.error_list),
            ncols=1,
            sharex=True,
            sharey=False,
            figsize=(x_length, x_length * self.phi),
            gridspec_kw={"hspace": 0.2},  # better vertical spacing
        )

        # Handle case where there's only one error metric (axs won't be a list)
        if len(self.error_list) == 1:
            axs = [axs]

        for ax, error_metric in zip(axs, self.error_list):
            long_df = long_dfs[error_metric]

            sns.boxplot(
                ax=ax,
                data=long_df,
                x="simulation",
                y=error_metric,
                hue="year",
                hue_order=year_order,
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

            # Only show legend on the first subplot
            if ax == axs[0]:
                ax.legend(frameon=True, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.3), fancybox=True)
            else:
                ax.get_legend().remove()

        plt.xticks(rotation=0, ha="center")
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.12, right=0.98, hspace=0.3)

        output_path = self.figures_dir / f"all_metric_by_simulation_and_year.{self.export_format}"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_boxplot_per_country(self, x_length=8):
        """Create grid of boxplots showing error distributions per country."""

        fig, axes = plt.subplots(
            nrows=len(self.sim_labels),
            ncols=len(self.error_list),
            figsize=(x_length, x_length * self.phi),
            sharex="col",
        )

        for i, sim_label in enumerate(self.sim_labels):
            for j, error in enumerate(self.error_list):
                df = self.scores_dict[sim_label][error]

                df_long = df.reset_index(names="year").melt(
                    id_vars="year",
                    var_name="country",
                    value_name=error,
                )

                ax = axes[i, j]

                sns.boxplot(
                    data=df_long,
                    y="country",
                    x=error,
                    ax=ax,
                    showfliers=True,
                    width=0.6,
                )

                # Remove "country" label from y-axis
                ax.set_ylabel("")

                ax.grid(axis="x", alpha=0.4)
                ax.grid(axis="y", alpha=0.15)

                # Add column titles on top row
                if i == 0:
                    ax.set_title(f"{error.upper()} ({self.error_units[error]})", fontsize=11, pad=10)

                # Add row labels on the left
                if j == 0:
                    ax.text(
                        -0.25,
                        0.5,
                        sim_label,
                        transform=ax.transAxes,
                        fontsize=11,
                        va="center",
                        ha="right",
                        rotation=90,
                    )

        # Configure x-axis for each column AFTER all plots are created
        for j, error in enumerate(self.error_list):
            # Set limits for all rows in this column
            for i in range(len(self.sim_labels)):
                axes[i, j].set_xlim(left=0, right=self.error_max_values[error])

            # Configure only the top subplot in each column
            top_ax = axes[0, j]
            top_ax.xaxis.tick_top()
            top_ax.tick_params(axis="x", which="both", top=True, labeltop=True, bottom=False, labelbottom=False)

            # Hide x-axis labels and ticks for middle and bottom rows
            for i in range(1, len(self.sim_labels)):
                axes[i, j].tick_params(
                    axis="x", which="both", top=False, labeltop=False, bottom=False, labelbottom=False
                )

        plt.tight_layout()

        output_path = self.figures_dir / f"error_distribution_per_country.{self.export_format}"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

    def plot_yearly_values_per_country(self, x_length=8):
        """
        Create grid of scatter plots showing yearly values per country.
        Same layout as boxplot_per_country, but instead of boxplots
        we plot one dot per year (x = error value, y = country).
        """

        fig, axes = plt.subplots(
            nrows=len(self.sim_labels),
            ncols=len(self.error_list),
            figsize=(x_length, x_length * self.phi),
            sharex="col",
        )

        # Ensure axes is 2D even if only 1 sim or 1 metric
        if len(self.sim_labels) == 1 and len(self.error_list) == 1:
            axes = [[axes]]
        elif len(self.sim_labels) == 1:
            axes = [axes]
        elif len(self.error_list) == 1:
            axes = [[ax] for ax in axes]

        legend_handles = []
        legend_labels = []

        for i, sim_label in enumerate(self.sim_labels):
            for j, error in enumerate(self.error_list):
                df = self.scores_dict[sim_label][error]

                # rows = years, columns = countries
                df_long = df.reset_index(names="year").melt(
                    id_vars="year",
                    var_name="country",
                    value_name=error,
                )

                df_long["year"] = df_long["year"].astype(str)
                year_order = sorted(df_long["year"].unique())

                # Same year color logic as other year-based plots
                # Palette options: mako, blues, coolwarm, crest, mako, rocket_r, rocket
                palette = sns.color_palette("coolwarm", n_colors=len(year_order))
                year_color_map = dict(zip(year_order, palette))

                ax = axes[i][j]

                # Plot one dot per (country, year)
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

                    # Collect legend entries only once
                    if i == 0 and j == 0:
                        legend_handles.append(sc)
                        legend_labels.append(year)

                # Clean grid (minimal)
                ax.grid(color="gray", linewidth=0.6, alpha=0.7, linestyle="dashed")

                # Spines
                # ax.spines["left"].set_linewidth(0.8)
                # ax.spines["bottom"].set_linewidth(0.8)

                ax.set_xlim(0, self.error_max_values[error])
                ax.invert_yaxis()
                ax.set_ylabel("")

                # Column titles (top row)
                if i == 0:
                    ax.set_title(
                        f"{error.upper()} ({self.error_units[error]})",
                        fontsize=11,
                        pad=10,
                    )

                # Row labels (left side)
                if j == 0:
                    ax.text(
                        -0.25,
                        0.5,
                        sim_label,
                        transform=ax.transAxes,
                        fontsize=11,
                        va="center",
                        ha="right",
                        rotation=90,
                    )

                # Move x-axis to top like original boxplot layout
                if i == 0:
                    ax.xaxis.tick_top()
                    ax.tick_params(
                        axis="x",
                        which="both",
                        top=True,
                        labeltop=True,
                        bottom=False,
                        labelbottom=False,
                    )
                else:
                    ax.tick_params(
                        axis="x",
                        which="both",
                        top=False,
                        labeltop=False,
                        bottom=False,
                        labelbottom=False,
                    )

        # Add one single legend below entire figure
        fig.legend(
            legend_handles,
            legend_labels,
            title="Year",
            loc="lower center",
            ncol=len(legend_labels),
            frameon=True,
            bbox_to_anchor=(0.5, 0),
        )

        # ---- Add buffer space at bottom for legend ----
        bottom_space = 0.035  # increase if needed
        plt.tight_layout(rect=[0, bottom_space, 1, 1])

        output_path = self.figures_dir / f"error_yearly_values_per_country.{self.export_format}"
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
        load_shedding_label="hindcast-dyn-rolling",
    ):
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
            ax.plot(ref_series.index, ref_series, label="Europe reference", color=ref_color, linewidth=1.0, zorder=3)

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

    def generate_all_plots(self):
        """Generate all plots."""
        print("Generating plots...")

        # Print boxplot per country across sims
        # self.plot_boxplot_per_country()

        # Print scatter plot per country across sims
        # self.plot_yearly_values_per_country()

        # Print single boxplot for error metrics
        self.plot_error_by_simulation_and_year_all(x_length=6)

        # Plot individual boxplot for simulation per year
        # for error_metric in self.error_list:
        #    self.plot_error_by_simulation_and_year(error_metric, x_length=7)

        # Plot price simulations
        # self.plot_prices()

        # Plot Europe price reference + simulations
        # self.plot_europe_prices()

        print("All plots generated successfully!")


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
