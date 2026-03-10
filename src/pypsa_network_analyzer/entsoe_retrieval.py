"""
fetch_entsoe_capacity.py
------------------------
Fetches installed generation capacity from ENTSO-E Transparency Platform
for a list of country codes and a given year, then saves the result as a
CSV in the configured benchmark folder.

Usage (called from main.py):
    from fetch_entsoe_capacity import fetch_and_save_entsoe_capacity
    fetch_and_save_entsoe_capacity(cfg, years_list, logger)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from entsoe import EntsoePandasClient


# ---------------------------------------------------------------------------
# ENTSO-E carrier  →  PyPSA carrier mapping
# ---------------------------------------------------------------------------
CARRIER_MAPPING: dict[str, str] = {
    "Biomass": "biomass",
    "Fossil Brown coal/Lignite": "lignite",
    "Fossil Coal-derived gas": "coal",
    "Fossil Gas": "Combined-Cycle Gas",
    "Fossil Hard coal": "coal",
    "Fossil Oil": "oil",
    "Fossil Oil shale": "oil",
    "Fossil Peat": "coal",
    "Geothermal": "geothermal",
    "Hydro Pumped Storage": "Pumped Hydro Storage",
    "Hydro Run-of-river and poundage": "Run of River",
    "Hydro Water Reservoir": "Reservoir & Dam",
    "Marine": "other",
    "Nuclear": "nuclear",
    "Other": "other",
    "Other renewable": "other",
    "Solar": "Solar",
    "Waste": "biomass",
    "Wind Offshore": "Offshore Wind (AC)",
    "Wind Onshore": "Onshore Wind",
}

COUNTRY_CODES: list[str] = [
    "AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "DK", "EE",
    "ES", "FI", "FR", "GB", "GR", "HR", "HU", "IE", "IT", "LT",
    "LU", "LV", "ME", "MK", "NL", "NO", "PL", "PT", "RO", "RS",
    "SE", "SI", "SK", "XK",
]


def _query_year(
    client: EntsoePandasClient,
    year: int,
    country_codes: Sequence[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Query installed generation capacity for *all* country_codes for a single year.

    Returns a tidy DataFrame indexed by country with one column per ENTSO-E
    technology (before carrier mapping).
    """
    start = pd.Timestamp(f"{year}-01-01", tz="Europe/Brussels")
    end   = pd.Timestamp(f"{year}-12-31 23:59", tz="Europe/Brussels")

    results: list[pd.DataFrame] = []
    for code in country_codes:
        logger.info(f"  [ENTSO-E] Querying {code} for {year} …")
        try:
            df = client.query_installed_generation_capacity(
                country_code=code,
                start=start,
                end=end,
            )
            if isinstance(df, pd.Series):
                df = df.to_frame(name="capacity_mw")
            df["country"] = code
            results.append(df)
        except Exception as exc:
            logger.warning(f"  [ENTSO-E] Skipping {code} ({year}): {exc}")

    if not results:
        logger.error(f"No ENTSO-E data retrieved for year {year}.")
        return pd.DataFrame()

    all_data = pd.concat(results)
    return all_data


def _process(raw: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Apply carrier mapping, group duplicate columns, add EU aggregate row.

    Returns a DataFrame indexed by country with one column per PyPSA carrier.
    """
    df = raw.reset_index(drop=True).set_index("country")

    # Rename ENTSO-E technologies → PyPSA carriers
    df = df.rename(columns=CARRIER_MAPPING)

    # Sum columns that now share the same carrier name
    df_grouped = df.T.groupby(level=0).sum().T

    # EU aggregate row
    df_grouped.loc["EU"] = df_grouped.sum()

    # Warn about and deduplicate country rows
    if df_grouped.index.duplicated().any():
        logger.warning("Duplicated country rows found — keeping first occurrence.")
        df_grouped = df_grouped[~df_grouped.index.duplicated(keep="first")]

    return df_grouped


def fetch_and_save_entsoe_capacity(
    cfg,
    years_list: Sequence[int],
    logger: logging.Logger,
) -> None:
    """
    Main entry point called from main.py.

    For each year in *years_list*, fetches ENTSO-E installed capacities,
    processes them, and saves a CSV to:
        <cfg.paths.data>/benchmark/entsoe_installed_capacity_<year>.csv

    Parameters
    ----------
    cfg        : Hydra DictConfig from main.py — must contain cfg.entsoe_api_key
    years_list : iterable of integer years, e.g. [2019, 2020, 2021, 2022]
    logger     : shared logger instance
    """
    benchmark_dir = Path(cfg.paths.data) / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Fetching ENTSO-E installed capacities ===")

    api_token = cfg.get("entsoe_api_key")
    if not api_token:
        logger.error("'entsoe_api_key' not found in Hydra config — skipping ENTSO-E fetch.")
        return

    client = EntsoePandasClient(api_key=api_token)

    for year in years_list:
        out_path = benchmark_dir / f"entsoe_installed_capacity_{year}.csv"

        if out_path.exists():
            logger.info(f"  Skipping year {year} — file already exists: {out_path}")
            continue

        logger.info(f"  Processing year {year} …")
        raw = _query_year(client, year, COUNTRY_CODES, logger)

        if raw.empty:
            continue

        df_processed = _process(raw, logger)
        df_processed.to_csv(out_path)
        logger.info(f"  Saved: {out_path}")

    logger.info("=== ENTSO-E capacity fetch complete ===")