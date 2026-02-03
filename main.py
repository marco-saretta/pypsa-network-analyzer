import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from pathlib import Path
import gc

from pypsa_network_analyzer import NetworkAnalyzer, ScoreAnalyzer
from pypsa_network_analyzer import setup_logger, merge_dataframes


@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    logger = setup_logger(log_dir=cfg.paths.log)
    logger.info("Starting PyPSA Network Analysis")

    # Process network files
    for network_file in tqdm(cfg.network_files, desc="Processing networks"):
        try:
            analyzer = NetworkAnalyzer(config=cfg, network_file=network_file, logger=logger)
            analyzer.extract_summary()
            analyzer.plot_all_figures()
            gc.collect()
        except Exception as e:
            logger.error(f"Failed to process {network_file}: {e}", exc_info=True)

    # Merge results by weather year groups
    for group_name, folder_list in tqdm(cfg.config_results_concat.items(), desc="Merging results"):
        output_dir = Path("results_concat") / group_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for df_name in ["electricity_prices", "generators_dispatch"]:
            merge_dataframes(
                root=cfg.paths.root,
                res_concat_folder=output_dir,
                file_concat_folder_dict=folder_list,
                df_to_merge=df_name,
                logger=logger,
            )
            
    # # Compute scores
    #     score_analyzer = ScoreAnalyzer(
    #         simulation_folder=simulation,
    #         file_to_examine="electricity_prices",
    #         weather_years_list=list(weather_year_dict.values()),
    #         logger=logger,
    #         )
        
    #     # Compute metrics
    #     df_mae, df_rmse, df_smape = score_analyzer.compute_scores_by_year()
    #     # Drop any excluded countries
    #     if cfg.exclude_countries:
    #         df_mae = df_mae.drop(columns=exclude_countries, errors="ignore")
    #         df_rmse = df_rmse.drop(columns=exclude_countries, errors="ignore")
    #         df_smape = df_smape.drop(columns=exclude_countries, errors="ignore")
            
    #     # Save each score type
    #     score_analyzer.save_scores(df_mae, filename="scores_mae.csv")
    #     score_analyzer.save_scores(df_rmse, filename="scores_rmse.csv")
    #     score_analyzer.save_scores(df_smape, filename="scores_smape.csv")
    #     del score_analyzer
    #     gc.collect()
        
    # logger.info(f"Completed {simulation}\n")
    # logger.info("=== Batch Run Completed ===")


if __name__ == "__main__":
    main()
    