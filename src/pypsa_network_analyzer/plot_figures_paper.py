import hydra
from omegaconf import DictConfig
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


class ResultsPlotter:
    """Class to handle loading and plotting of simulation results."""

    def __init__(self, cfg: DictConfig):
        # Config and directories
        self.cfg = cfg
        self.root_dir = Path(cfg.paths.root)
        self.data_dir = self.root_dir / "data"
        self.results_concat_dir = self.root_dir / "results_concat"
        self.figures_dir = self.root_dir / "figures_paper"
        self.figures_dir.mkdir(exist_ok=True)

        # Configuration
        self.sim_labels = list(cfg.config_results_concat.keys())
        self.error_list = ["mae", "rmse", "smape"]
        self.benchmark_name = "electricity_prices"
        self.export_format = cfg.plot_export_format
        self.error_units = {"mae": "EUR/MWh", "rmse": "EUR/MWh", "smape": "%"}

        # Config labels
        self.error_max_values = {"mae": 250, "rmse": 300, "smape": 150}
        self.error_axis_labels = {"mae": "[EUR/MWh]", "rmse": "[EUR/MWh]", "smape": "[%]"}

        self.setup_style()
        self.load_scores()
        self.load_prices()

    def setup_style(self):
        mpl.rcParams["axes.spines.right"] = False
        mpl.rcParams["axes.spines.top"] = False
        sns.set_theme(
            style="whitegrid",
            context="paper",
            font_scale=1.1,
        )
        self.phi = 1.618

        self.country_color = {"DE": "tab:blue"}

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

        plt.legend(title="Year", frameon=True)
        plt.tight_layout()

        output_path = self.figures_dir / f"{error_metric}_by_simulation_and_year.{self.export_format}"
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


    def plot_prices(self, x_length=8, resampling_rule="W", countries_list=["DE"], rolling_window=None):
        """Plot benchmark vs simulations per country."""

        for country in countries_list:
            fig, ax = plt.subplots(figsize=(x_length, x_length / self.phi))

            for label, df in self.prices_dict.items():
                if country not in df.columns:
                    continue

                series = df[country]

                if resampling_rule:
                    series = series.resample(resampling_rule).mean()

                if rolling_window:
                    series = series.rolling(rolling_window, center=True).mean()

                if label == "benchmark":
                    ax.plot(
                        series.index,
                        series,
                        label="Benchmark",
                        color="black",
                        linewidth=1,
                    )
                else:
                    ax.plot(
                        series.index,
                        series,
                        label=label,
                        alpha=0.8,
                    )

            ax.set_title(f"{country} electricity price")
            ax.set_ylabel("EUR/MWh")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False)

            plt.tight_layout()

            output_path = self.figures_dir / f"price_{country}.{self.export_format}"
            plt.savefig(output_path)
            plt.close()

            print(f"Saved: {output_path}")

    def generate_all_plots(self):
        """Generate all plots."""
        print("Generating plots...")

        # Print boxplot per country across sims
        self.plot_boxplot_per_country()

        # Plot individual boxplot for simulation per year
        for error_metric in self.error_list:
            self.plot_error_by_simulation_and_year(error_metric, x_length=7)

        # Plot price simulations
        self.plot_prices()

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
