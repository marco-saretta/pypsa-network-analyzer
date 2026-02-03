import numpy as np
from pathlib import Path
import pandas as pd
from typing import Optional
import re

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
    root: Path,
    res_concat_folder: Path,
    file_concat_folder_dict: list,
    df_to_merge: str,
    logger=None,
    resample_rule: Optional[str] = None,
    ) -> Optional[Path]:
    """
    Merge CSV files from multiple weather year folders into a single combined CSV.
    
    Args:
        root: Root path of the project
        res_concat_folder: Output folder path (e.g., results_concat/hindcast_dyn_old)
        file_concat_folder_dict: List of folder names to merge
        df_to_merge: Name of CSV file to merge (without .csv extension)
        logger: Optional logger instance
        resample_rule: Optional resampling rule (e.g., "H" for hourly)
    
    Returns:
        Path to merged file, or None if failed
    """
    root = Path(root)
    output_path = Path(res_concat_folder) / df_to_merge / f"combined_{df_to_merge}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df_list = []
        
        for folder_name in file_concat_folder_dict:
            csv_path = root / "results" / folder_name / "summary" / f"{df_to_merge}.csv"
            
            if not csv_path.exists():
                if logger:
                    logger.warning(f"File not found: {csv_path}")
                continue
            
            # Read CSV with datetime index
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df_list.append(df)
        
        if not df_list:
            if logger:
                logger.error(f"No files found for {df_to_merge}")
            return None
        
        # Concatenate, sort, and deduplicate
        df_all = pd.concat(df_list).sort_index()
        df_all = df_all[~df_all.index.duplicated(keep='first')]
        
        # Optional resampling
        if resample_rule:
            df_all = df_all.resample(resample_rule).mean()
        
        # Save
        df_all.to_csv(output_path)
        
        if logger:
            logger.info(
                f"Merged {df_to_merge}: {len(df_list)} files --> "
                f"{len(df_all)} rows, {len(df_all.columns)} columns"
            )
        
        return output_path
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to merge {df_to_merge}: {e}", exc_info=True)
        return None
