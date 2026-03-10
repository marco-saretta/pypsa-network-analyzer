import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from pathlib import Path
import gc

from pypsa_network_analyzer.network_analyzer import NetworkAnalyzer
from pypsa_network_analyzer.score_analyzer import ScoreAnalyzer
from pypsa_network_analyzer.utils import setup_logger, merge_dataframes
from pypsa_network_analyzer.entsoe_retrieval import fetch_and_save_entsoe_capacity

years_list = [2020, 2021, 2022, 2023, 2024]

@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    logger = setup_logger(log_dir=cfg.paths.log)
    logger.info("Starting PyPSA Network Analysis")

    # Fetch ENTSO-E installed capacities for each benchmark year
    fetch_and_save_entsoe_capacity(
        cfg=cfg,
        years_list=cfg.years_list,          # e.g. [2019, 2020, 2021, 2022]
        logger=logger,
    )

    # Process network files
    for network_file in tqdm(cfg.network_files, desc="Processing networks"):
        try:
            analyzer = NetworkAnalyzer(config=cfg, network_file=network_file, logger=logger)
            analyzer.extract_summary()
            analyzer.extract_pypsa_capacity()
            analyzer.plot_all_figures()
            gc.collect()
        except Exception as e:
            logger.error(f"Failed to process {network_file}: {e}", exc_info=True)

    # Merge results by weather year groups
    for group_name, folder_list in tqdm(cfg.config_results_concat.items(), desc="Merging results"):
        res_concat_folder_dir = Path(cfg.paths.results_concat) / group_name
        res_concat_folder_dir.mkdir(parents=True, exist_ok=True)

        for df_name in tqdm(cfg.merge_dataframes, desc="Merging dataframes"):
            merge_dataframes(
                results_dir=Path(cfg.paths.results),
                res_concat_folder=res_concat_folder_dir,
                file_concat_folder_dict=folder_list,
                df_to_merge_file=df_name,
                logger=logger,
            )

        # Compute scores
        for benchmark_file in tqdm(cfg.benchmark_score_files, desc="Computing scores"):
            score_analyzer = ScoreAnalyzer(
                root_dir=cfg.paths.root,
                res_concat_folder_dir=res_concat_folder_dir,
                benchmark_file=benchmark_file,
                years_list=cfg.years_list,
                logger=logger,
            )
        # df_mae, df_rmse, df_smape = score_analyzer.compute_scores_by_year()
        results = score_analyzer.compute_scores_by_year()

        df_mae_hourly, df_rmse_hourly, df_smape_hourly = results["hourly"]
        df_mae_daily, df_rmse_daily, df_smape_daily = results["daily"]
        df_mae_weekly, df_rmse_weekly, df_smape_weekly = results["weekly"]

        
        # Drop any excluded countries
        if cfg.exclude_countries:
            df_mae_hourly = df_mae_hourly.drop(columns=cfg.exclude_countries, errors="ignore")
            df_rmse_hourly = df_rmse_hourly.drop(columns=cfg.exclude_countries, errors="ignore")
            df_smape_hourly = df_smape_hourly.drop(columns=cfg.exclude_countries, errors="ignore")
            df_mae_daily = df_mae_daily.drop(columns=cfg.exclude_countries, errors="ignore")
            df_rmse_daily = df_rmse_daily.drop(columns=cfg.exclude_countries, errors="ignore")
            df_smape_daily = df_smape_daily.drop(columns=cfg.exclude_countries, errors="ignore")
            df_mae_weekly = df_mae_weekly.drop(columns=cfg.exclude_countries, errors="ignore")
            df_rmse_weekly = df_rmse_weekly.drop(columns=cfg.exclude_countries, errors="ignore")
            df_smape_weekly = df_smape_weekly.drop(columns=cfg.exclude_countries, errors="ignore")

        # Save each score type
        score_analyzer.save_scores(df_mae_hourly, filename="scores_mae.csv")
        score_analyzer.save_scores(df_rmse_hourly, filename="scores_rmse.csv")
        score_analyzer.save_scores(df_smape_hourly, filename="scores_smape.csv")
        score_analyzer.save_scores(df_mae_daily, filename="scores_mae_daily.csv")
        score_analyzer.save_scores(df_rmse_daily, filename="scores_rmse_daily.csv")
        score_analyzer.save_scores(df_smape_daily, filename="scores_smape_daily.csv")
        score_analyzer.save_scores(df_mae_weekly, filename="scores_mae_weekly.csv")
        score_analyzer.save_scores(df_rmse_weekly, filename="scores_rmse_weekly.csv")
        score_analyzer.save_scores(df_smape_weekly, filename="scores_smape_weekly.csv")

        results_europe = score_analyzer.compute_europe_scores_by_year()
        df_eu_mae_hourly, df_eu_rmse_hourly, df_eu_smape_hourly = results_europe["hourly"]
        df_eu_mae_daily, df_eu_rmse_daily, df_eu_smape_daily = results_europe["daily"]
        df_eu_mae_weekly, df_eu_rmse_weekly, df_eu_smape_weekly = results_europe["weekly"]

        score_analyzer.save_scores(df_eu_mae_hourly, "europe_mae.csv")
        score_analyzer.save_scores(df_eu_rmse_hourly, "europe_rmse.csv")
        score_analyzer.save_scores(df_eu_smape_hourly, "europe_smape.csv")
        score_analyzer.save_scores(df_eu_mae_daily, "europe_mae_daily.csv")
        score_analyzer.save_scores(df_eu_rmse_daily, "europe_rmse_daily.csv")
        score_analyzer.save_scores(df_eu_smape_daily, "europe_smape_daily.csv")
        score_analyzer.save_scores(df_eu_mae_weekly, "europe_mae_weekly.csv")
        score_analyzer.save_scores(df_eu_rmse_weekly, "europe_rmse_weekly.csv")
        score_analyzer.save_scores(df_eu_smape_weekly, "europe_smape_weekly.csv")
        

        logger.info(f"Completed {group_name}\n")

        del score_analyzer
        gc.collect()

    logger.info("=== Batch Run Completed ===")


if __name__ == "__main__":
    main()
