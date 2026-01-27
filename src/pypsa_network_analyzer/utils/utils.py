import numpy as np
from pathlib import Path
import pandas as pd
from typing import Optional

def smape(A, F):
    """_summary_
    Symmetric Mean Absolute Percentage Error
    
    Acutal value
    Forecast value
    
    A = np.array([2,3,4,5,6,7,8,9])
    F = np.array([1,3,5,4,6,7,10,7])
    print(smape(A, F))
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        tmp = 2 * np.abs(F-A) / (np.abs(A) + np.abs(F))
    tmp[np.isnan(tmp)] = 0
    return np.sum(tmp) / len(tmp) * 100

def merge_dataframes(
    simulation_folder: str,
    df_to_merge: str,
    weather_year_dict: dict,
    base_path: Optional[Path] = None,
    resample_rule: Optional[str] = None,
    logger=None,
) -> Optional[Path]:
    """
    Merge CSV files from multiple weather year folders into a single combined CSV.
    
    Reads CSV files from subdirectories (e.g., weather_year_2020/summary/),
    creates proper datetime indices for each year, concatenates them chronologically,
    and saves the result in the results_concat folder.
    
    Args:
        simulation_folder: Name of the simulation folder (e.g., "hindcast-dyn")
        df_to_merge: Name of the CSV file to merge (without .csv extension)
        weather_year_dict: Dictionary mapping folder names to years 
                         (e.g., {"weather_year_2020": 2020, "weather_year_2021": 2021})
        base_path: Base path to project root. If None, uses current file location's parent
        resample_rule: Optional resampling rule (e.g., "H" for hourly). If None, no resampling.
        logger: Optional logger instance
    
    Returns:
        Path to the written file, or None if merge failed
    
    Usage:
        merge_dataframes(
            simulation_folder="hindcast-dyn",
            df_to_merge="electricity_prices",
            weather_year_dict={"weather_year_2020": 2020, "weather_year_2021": 2021},
            logger=logger
        )
    """
    
    # Set base path (project root)
    if base_path is None:
        base_path = Path(__file__).resolve().parent.parent  # Go up from utils/ to project root
    else:
        base_path = Path(base_path)
    
    # Define paths
    sim_folder = base_path / 'simulations' / simulation_folder
    res_concat_folder = sim_folder / "results_concat"
    res_concat_folder.mkdir(parents=True, exist_ok=True)
    output_filename = f"combined_{df_to_merge}.csv"
    
    if logger:
        logger.info(f"Starting merge for {simulation_folder}/{df_to_merge}")
    
    try:
        df_dict = {}
        
        # Process each weather year
        for wy_string, year in weather_year_dict.items():
            # Construct path to CSV file in results/summary
            csv_path = sim_folder / wy_string / "results" / "summary" / f"{df_to_merge}.csv"
            
            if not csv_path.exists():
                if logger:
                    logger.warning(f"CSV not found: {csv_path}")
                continue
            
            try:
                # Read CSV
                df_temp = pd.read_csv(csv_path, index_col=0)
                
                # # Create hourly datetime index for this year
                # df_temp.index = pd.date_range(
                #     start=f"{year}-01-01 00:00:00",
                #     periods=len(df_temp),
                #     freq='H'
                # )
                
                df_dict[wy_string] = df_temp
                if logger:
                    logger.info(f"Loaded {wy_string}: {len(df_temp)} rows")
                
            except Exception as e:
                if logger:
                    logger.error(f"Failed to read {csv_path}: {type(e).__name__}: {e}")
                continue
        
        if not df_dict:
            if logger:
                logger.error("No dataframes could be loaded")
            return None
        
        # Concatenate all dataframes chronologically
        df_all = pd.concat(df_dict.values(), axis=0)
        
        # Sort by datetime index to ensure chronological order
        df_all.sort_index(inplace=True)
        
        # Remove any duplicate timestamps (keep first occurrence)
        df_all = df_all[~df_all.index.duplicated(keep='first')]
        
        # Optional resampling
        if resample_rule:
            if logger:
                logger.info(f"Resampling to {resample_rule}")
            df_all = df_all.resample(resample_rule).mean()
        
        # Write combined dataframe to CSV
        out_path = res_concat_folder / output_filename
        df_all.to_csv(out_path)
        
        if logger:
            logger.info(
                f"Successfully merged {len(df_dict)} files into {out_path} "
                f"({len(df_all)} rows, {len(df_all.columns)} columns)"
            )
        
        return out_path
        
    except Exception as e:
        if logger:
            logger.error(f"Merge failed: {type(e).__name__}: {e}")
        return None


if __name__ == "__main__":
    # Example usage when run as a script
    weather_year_dict = {
        "weather_year_2020": 2020,
        "weather_year_2021": 2021,
        "weather_year_2022": 2022,
        "weather_year_2023": 2023,
        "weather_year_2024": 2024,
    }
    
    output_path = merge_dataframes(
        simulation_folder="hindcast-dyn",
        df_to_merge="electricity_prices",
        weather_year_dict=weather_year_dict
    )
    
    if output_path:
        print(f"Combined file saved to: {output_path}")
    else:
        print("Merge failed")