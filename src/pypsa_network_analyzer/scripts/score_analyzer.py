from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from matplotlib import pyplot as plt
from pypsa_network_analyzer.utils import smape


class ScoreAnalyzer:
    """Compare benchmark vs hindcast electricity prices with MAE, RMSE, and SMAPE metrics."""

    def __init__(self, simulation_folder, file_to_examine, weather_years_list, logger):
        self.logger = logger
        self.weather_years_list = sorted(weather_years_list)
        self.file_to_examine = file_to_examine

        base = Path(__file__).resolve().parent.parent
        self.sim_folder_dir = base / "simulations" / simulation_folder
        self.scores_dir = self.sim_folder_dir / "scores"
        self.scores_dir.mkdir(parents=True, exist_ok=True)

        # Input files
        self.file_dir = self.sim_folder_dir / "results_concat" / f"combined_{file_to_examine}.csv"
        self.benchmark_dir = base / "simulations" / "benchmark" / f"{file_to_examine}.csv"

        # Load and prepare data
        self._read_data()
        self._interpolate_na()
        self._filter_years()
        self._get_common_columns()

    def _read_data(self):
        """Read benchmark and hindcast data, ensure UTC timestamps."""

        def _load(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df.tz_localize("UTC") if df.index.tz is None else df.tz_convert("UTC")

        self.df_benchmark_raw = _load(self.benchmark_dir)
        self.df_raw = _load(self.file_dir)

    def _filter_years(self):
        """Filter both dataframes to the years and align indices."""
        years = self.weather_years_list

        df = self.df_raw[self.df_raw.index.year.isin(years)]
        df_bm = self.df_benchmark_raw[self.df_benchmark_raw.index.year.isin(years)]

        # Align rows exactly (important for leap year behavior)
        df_bm = df_bm.loc[df.index]
        self.df = df
        self.df_benchmark = df_bm

    def _interpolate_na(self):
        """Interpolate missing values in both raw DataFrames."""
        self.df_benchmark_raw = self.df_benchmark_raw.interpolate().ffill().bfill()
        self.df_raw = self.df_raw.interpolate().ffill().bfill()

    def _get_common_columns(self):
        """Find intersection of countries."""
        self.common_cols = sorted(set(self.df_benchmark.columns).intersection(self.df.columns))
        self.logger.info(f"Found {len(self.common_cols)} common countries: {self.common_cols}")

    def compute_scores_by_year(self):
        """
        Compute MAE, RMSE, SMAPE for every country and for each weather year.
        Return: df_mae, df_rmse, df_smape
        """

        years = self.weather_years_list
        countries = self.common_cols

        df_mae = pd.DataFrame(index=years, columns=countries, dtype=float)
        df_rmse = pd.DataFrame(index=years, columns=countries, dtype=float)
        df_smape = pd.DataFrame(index=years, columns=countries, dtype=float)

        for year in years:
            try:
                bench_y = self.df_benchmark[self.df_benchmark.index.year == year]
                sim_y = self.df[self.df.index.year == year]

                # Exact index alignment
                idx = bench_y.index.intersection(sim_y.index)
                bench_y = bench_y.reindex(idx)
                sim_y = sim_y.reindex(idx)

                for c in countries:
                    b = bench_y[c].to_numpy()
                    s = sim_y[c].to_numpy()

                    df_mae.loc[year, c] = mean_absolute_error(b, s)
                    df_rmse.loc[year, c] = root_mean_squared_error(b, s)
                    df_smape.loc[year, c] = smape(b, s)
            except Exception as e:
                self.logger.error(f"Error computing scores for year {year}: {e}")

        return df_mae, df_rmse, df_smape

    def _plot_country(self, country, save=False, plot_dir=None):
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.df_benchmark.index, self.df_benchmark[country], label="Benchmark", color="tab:blue")
        ax.plot(self.df.index, self.df[country], label="Dynamic", color="#ff7f0e")

        ax.legend()
        ax.set_title(f"{country} – Electricity Prices (EUR/MWh)")
        ax.set_ylabel("EUR/MWh")
        ax.grid(True, linestyle="--", alpha=0.4)

        if save:
            out = Path(plot_dir or (self.sim_folder_dir / "plots"))
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{country}_comparison.png", dpi=300)
            plt.close(fig)
        else:
            plt.show()

    def save_scores(self, df, filename):
        path = self.scores_dir / filename
        df.to_csv(path)
        self.logger.info(f"Saved: {path}")
