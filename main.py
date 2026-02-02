import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from pathlib import Path
import gc

from pypsa_network_analyzer import NetworkAnalyzer, ScoreAnalyzer
from pypsa_network_analyzer import setup_logger, merge_dataframes

log = setup_logger()


@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    # Setup logger
    logger = setup_logger(log_dir=cfg.paths.log)
    logger.info("Starting PyPSA Network Analysis")

    # Iterate over network files
    for network_file in tqdm(cfg.network_files, desc="Simulations"):
        try:
            network_analyzer = NetworkAnalyzer(
                config=cfg, 
                network_file=network_file,
                logger=logger
            )
            network_analyzer.extract_summary()
            #network_analyzer.plot_all_figures()
            logger.info(f"YES! Processed {network_file}")
            gc.collect()

        except Exception as e:
            logger.error(f"NOPE! Processing {network_file} returns error {e}", exc_info=True)

    # Merge results from weather years
    for simulation_to_be_merged in tqdm(cfg.config_results_concat.keys()):
        
        logger.info(f"START - Merge data for {simulation_to_be_merged}")
    #     for df_name in ['electricity_prices', 'generators_dispatch']:
    #         merge_dataframes(
    #             simulation_folder=simulation,
    #             df_to_merge=df_name,
    #             weather_year_dict=weather_year_dict,
    #             logger=logger
    #         )
            
            
if __name__ == "__main__":
    main()


# TODO
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