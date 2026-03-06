from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from matplotlib import pyplot as plt
from pypsa_network_analyzer.utils import smape


class ScoreAnalyzer:
    """Compare benchmark vs hindcast electricity prices with MAE, RMSE, and SMAPE metrics."""

    def __init__(self, root_dir, res_concat_folder_dir, benchmark_file, years_list: list, logger):
        self.root_dir = Path(root_dir)
        self.res_concat_folder_dir = Path(res_concat_folder_dir)
        self.benchmark_file_path = Path(benchmark_file)
        self.years_list = sorted(years_list)
        self.logger = logger

        self.benchmark_file_suffix = self.benchmark_file_path.suffix
        self.benchmark_file = self.benchmark_file_path.stem

        self.scores_dir = self.res_concat_folder_dir / self.benchmark_file / "scores"
        self.scores_dir.mkdir(parents=True, exist_ok=True)

        # Input files
        self.file_dir = self.res_concat_folder_dir / self.benchmark_file / f"combined_{benchmark_file}"
        self.benchmark_file_dir = self.root_dir / "data" / "benchmark" / self.benchmark_file_path

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

        self.df_benchmark_raw = _load(self.benchmark_file_dir)
        # Remove Feb 29 so leap years match 8760-hour simulations
        self.df_benchmark_raw = self.df_benchmark_raw[
            ~((self.df_benchmark_raw.index.month == 2) &
            (self.df_benchmark_raw.index.day == 29))
        ]
        self.df_pypsa_raw = _load(self.file_dir)

    def _interpolate_na(self):
        """Interpolate missing values in both raw DataFrames."""
        self.df_benchmark_interp = self.df_benchmark_raw.interpolate().ffill().bfill()
        self.df_pypsa_interp = self.df_pypsa_raw.interpolate().ffill().bfill()

    def _filter_years(self):
        """Filter both dataframes to the years and align indices."""

        self.df_benchmark_fyears = self.df_benchmark_interp[self.df_benchmark_interp.index.year.isin(self.years_list)]
        self.df_pypsa_fyears = self.df_pypsa_interp[self.df_pypsa_interp.index.year.isin(self.years_list)]

        self.df_pypsa = self.df_pypsa_fyears

        # If statement to check if time index coincides. If not, proobably due to leap years
        if not self.df_pypsa_fyears.index.equals(self.df_benchmark_fyears.index):
            self.logger.warning(
                "Benchmark dataframe and target dataframe have different indices. Check for leap years."
            )

            # Align rows exactly (important for leap year behavior)
            self.df_benchmark = self.df_benchmark_fyears.loc[self.df_pypsa_fyears.index]
            self.logger.info("Aligned dataframes indices to account for leap years.")
        else:
            self.df_benchmark = self.df_benchmark_fyears

    def _get_common_columns(self):
        """Find intersection of countries."""
        self.common_cols = sorted(set(self.df_benchmark.columns).intersection(self.df_pypsa.columns))
        self.logger.info(f"Found {len(self.common_cols)} common countries: {self.common_cols}")

    def compute_scores_by_year(self):
        """
        Compute MAE, RMSE, SMAPE for every country and for each weather year.
        Return: df_mae, df_rmse, df_smape
        """

        resolutions = {
            "hourly": None,
            "daily": "D",
            "weekly": "W-MON"
        }

        results = {}

        for res_name, res_rule in resolutions.items():

            df_mae = pd.DataFrame(index=self.years_list, columns=self.common_cols, dtype=float)
            df_rmse = pd.DataFrame(index=self.years_list, columns=self.common_cols, dtype=float)
            df_smape = pd.DataFrame(index=self.years_list, columns=self.common_cols, dtype=float)

            for year in self.years_list:
                try:
                    bench_y = self.df_benchmark[self.df_benchmark.index.year == year]
                    sim_y = self.df_pypsa[self.df_pypsa.index.year == year]

                    # Resample if needed
                    if res_rule:
                        bench_y = bench_y.resample(res_rule).mean()
                        sim_y = sim_y.resample(res_rule).mean()

                    # Align AFTER resampling
                    idx = bench_y.index.intersection(sim_y.index)
                    bench_y = bench_y.loc[idx]
                    sim_y = sim_y.loc[idx]

                    for c in self.common_cols:

                        df_pair = pd.concat(
                            [bench_y[c], sim_y[c]],
                            axis=1,
                            keys=["bench", "sim"]
                        ).dropna()

                        if df_pair.empty:
                            self.logger.warning(f"No valid data for {c} in {year}")
                            continue

                        b = df_pair["bench"].to_numpy()
                        s = df_pair["sim"].to_numpy()

                        df_mae.loc[year, c] = mean_absolute_error(b, s)
                        df_rmse.loc[year, c] = root_mean_squared_error(b, s)
                        df_smape.loc[year, c] = smape(b, s)
                except Exception as e:
                    self.logger.error(f"Error computing scores for year {year}: {e}")
            results[res_name] = (df_mae, df_rmse, df_smape)

        return results
    
    def compute_europe_scores_by_year(self):
        """
        Compute MAE, RMSE, SMAPE for Europe price on multiple resolutions:
        benchmark: europe_price_ref
        simulation: europe_price
        Returns a dict with keys: "hourly", "daily", "weekly"
        """

        required_cols = {"europe_price_ref", "europe_price"}
        if not required_cols.issubset(self.df_pypsa.columns):
            raise ValueError(
                f"Missing required columns in simulation file: {required_cols}"
            )

        resolutions = {
            "hourly": "H",
            "daily": "D",
            "weekly": "W"
        }

        results = {}

        for res_name, res_rule in resolutions.items():

            df_mae = pd.Series(index=self.years_list, dtype=float, name="MAE")
            df_rmse = pd.Series(index=self.years_list, dtype=float, name="RMSE")
            df_smape = pd.Series(index=self.years_list, dtype=float, name="SMAPE")

            for year in self.years_list:
                try:
                    df_y = self.df_pypsa[self.df_pypsa.index.year == year]

                    # select columns
                    benchmark = df_y["europe_price_ref"]
                    simulation = df_y["europe_price"]

                    # resample if needed
                    if res_rule != "H":
                        benchmark = benchmark.resample(res_rule).mean()
                        simulation = simulation.resample(res_rule).mean()

                    # align timestamps
                    benchmark, simulation = benchmark.align(simulation, join="inner")

                    # drop NaNs
                    mask = benchmark.notna() & simulation.notna()
                    b = benchmark[mask].to_numpy()
                    s = simulation[mask].to_numpy()

                    if len(b) == 0:
                        continue

                    df_mae.loc[year] = mean_absolute_error(b, s)
                    df_rmse.loc[year] = root_mean_squared_error(b, s)
                    df_smape.loc[year] = smape(b, s)

                except Exception as e:
                    self.logger.error(f"Error computing Europe scores for year {year} ({res_name}): {e}")

            results[res_name] = (df_mae, df_rmse, df_smape)

        return results

    def save_scores(self, df, filename):
        path = self.scores_dir / filename
        df.to_csv(path)
        self.logger.info(f"Saved: {path}")
