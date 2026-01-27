from scripts import NetworkAnalyzer, ResultAnalyzer
from utils import get_logger, merge_dataframes
from tqdm import tqdm
import gc

logger = get_logger()

simulations = [
    # "hindcast_dyn_old",
    # "hindcast_std_old",
    "hindcast-dyn",
    # "hindcast-std",
    # "hindcast-dyn-2022",
    # "hindcast-dyn-spec-cap",
    # "hindcast-dyn-spec-cap-rolling",
    # "hindcast-dyn-spec-cap-irena",
    # "hindcast-dyn-spec-cap-rolling-irena",
    # "hindcast-dyn-rolling",
]

weather_year_dict = {
    "weather_year_2020": 2020,
    "weather_year_2021": 2021,
    "weather_year_2022": 2022,
    "weather_year_2023": 2023,
    "weather_year_2024": 2024
}

network_file = "base_s_39_elec_Ept.nc"
# network_file = "base_s_39_elec_.nc"
export_format = "pdf"
exclude_countries = ['MK']  # Add problematic countries here

logger.info("=== Starting Batch Run ===")

# for simulation in tqdm(simulations, desc="Processing Simulations"):

#     # === PHASE 1: Network Analysis (Optional) ===
#     # Uncomment to process network files
#     for weather_year in weather_year_dict.keys():
#         try:
#             analyzer = NetworkAnalyzer(
#                 simulation_folder=simulation,
#                 weather_year=weather_year,
#                 network_file=network_file,
#                 plots_format_export=export_format,
#                 logger=logger,
#             )
#             analyzer.extract_summary()
#             analyzer.plot_all_figures()
#             logger.info(f"Success {simulation} | {weather_year}")
#         except Exception as e:
#             logger.error(f"Error {simulation} | {weather_year}: {e}")
#         finally:
#             del analyzer
#             gc.collect()

#     # === PHASE 2: Merge DataFrames ===
#     #  Uncomment to merge results from weather years
#     logger.info(f"Merging data for {simulation}...")
#     for df_name in ['electricity_prices', 'generators_dispatch']:
#         merge_dataframes(
#             simulation_folder=simulation,
#             df_to_merge=df_name,
#             weather_year_dict=weather_year_dict,
#             logger=logger
#         )

#     # === PHASE 3: Compute Scores ===

#     logger.info(f"\n--- Analyzing {simulation} ---")

#     # --- COMPUTE SCORES ---
#     r = ResultAnalyzer(
#         simulation_folder=simulation,
#         file_to_examine="electricity_prices",
#         weather_years_list=list(weather_year_dict.values()),
#         logger=logger,
#     )

#     # Compute metrics
#     df_mae, df_rmse, df_smape = r.compute_scores_by_year()

#     # Drop any excluded countries
#     if exclude_countries:
#         df_mae = df_mae.drop(columns=exclude_countries, errors="ignore")
#         df_rmse = df_rmse.drop(columns=exclude_countries, errors="ignore")
#         df_smape = df_smape.drop(columns=exclude_countries, errors="ignore")

#     # Save each score type
#     r.save_scores(df_mae, filename="scores_mae.csv")
#     r.save_scores(df_rmse, filename="scores_rmse.csv")
#     r.save_scores(df_smape, filename="scores_smape.csv")

#     del r
#     gc.collect()

#     logger.info(f"Completed {simulation}\n")

# logger.info("=== Batch Run Completed ===")