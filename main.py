import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from pathlib import Path

from pypsa_network_analyzer import NetworkAnalyzer
from pypsa_network_analyzer import setup_logger

log = setup_logger()


@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    # Setup logger
    logger = setup_logger(log_dir=cfg.paths.log)
    logger.info("Starting PyPSA Network Analysis Pipeline")

    # Iterate over simulations and weather years
    for simulation in tqdm(cfg.simulations, desc="Simulations"):
        for weather_year in tqdm(cfg.weather_years, desc="Weather Years", leave=False):
            try:
                network_analyzer = NetworkAnalyzer(
                    config=cfg, 
                    simulation=simulation,
                    weather_year=weather_year,
                    logger=logger
                )
                # network_analyzer.extract_summary()
                # network_analyzer.plot_all_figures()
                logger.info(f"Yay! Completed {simulation} | {weather_year}")

            except Exception as e:
                logger.error(f"Ney! Error in {simulation} | {weather_year}: {e}")


if __name__ == "__main__":
    main()


# TODO


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