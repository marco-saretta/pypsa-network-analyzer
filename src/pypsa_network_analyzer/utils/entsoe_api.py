"""
ENTSO-E API Data Collector

This module provides a class to collect electricity load and day-ahead price data
from the ENTSO-E Transparency Platform API.

Code sources:
    - https://thesmartinsights.com/how-to-query-data-from-the-entso-e-transparency-platform-using-python/
    - https://github.com/EnergieID/entsoe-py

Example:
    Basic usage:
    ```python
    collector = EntsoeAPIDataCollector(
        start=pd.Timestamp('20240101', tz='UTC'),
        end=pd.Timestamp('20240131', tz='UTC')
    )
    
    # Collect data for specific countries
    load_dict = collector.collect_load_dict(countries=['DE_LU', 'FR'])
    price_data = collector.collect_day_ahead_prices(countries=['DE_LU', 'FR'])
    ```

Author: Marco Saretta
Date: 2025
"""

import pandas as pd
from entsoe import EntsoePandasClient
from tqdm import tqdm
import logging
from typing import Optional, List, Dict
from pathlib import Path


class EntsoeAPIDataCollector:
    """
    A class to collect electricity load data from the ENTSO-E Transparency Platform API.

    Attributes:
        client (EntsoePandasClient): The ENTSO-E API client.
        start (pd.Timestamp): Start date for data collection.
        end (pd.Timestamp): End date for data collection.
        countries_dict (Dict[str, str]): Mapping of country codes to full names.
        countries_list (List[str]): List of countries to collect data for.
        output_dir (Path): Directory to store output files.
        logger (logging.Logger): Logger instance for logging messages.
    """

    # Class-level constants
    DEFAULT_API_KEY = 'INSERT API HERE'
    DEFAULT_START_DATE = pd.Timestamp('20000101', tz='UTC')
    DEFAULT_END_DATE = pd.Timestamp('20500101', tz='UTC')
    OUTPUT_FOLDER = 'api_output'
    
    def __init__(
            self,
            start: pd.Timestamp = DEFAULT_START_DATE,
            end: pd.Timestamp = DEFAULT_END_DATE,
            api_key: str = DEFAULT_API_KEY,
            countries_list: Optional[List[str]] = None,
            output_dir = OUTPUT_FOLDER
        ):
            """
            Initialize the data collector with API key, date range, countries, and output directory.
            """
            # Initialize the query parameters
            self.client = EntsoePandasClient(api_key=api_key)
            self.start = start
            self.end = end
            
            # Set the output folder directory
            script_dir = Path(__file__).resolve().parent
            parent_dir = script_dir.parent
            self.output_dir = (parent_dir / output_dir).resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Extract the general country list
            self.countries_dict = self._get_countries_dict()
            self.countries_list = countries_list or list(self.countries_dict.keys())

            self.logger = self._setup_logger()
            self._validate_date_range()
        
    def _setup_logger(self) -> logging.Logger:
        """
        Setup logger to log messages to both console and file.

        Returns:
            logging.Logger: Configured logger object.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh = logging.FileHandler(self.output_dir / 'entsoe_data_collection.log')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        return logger
        
    def _get_countries_dict(self) -> Dict[str, str]:
        """
        Get the mapping of country codes to full country names.
        
        Returns:
            Dict[str, str]: Dictionary mapping country codes to full names
        """
        return {
            'AL': 'Albania',
            'AT': 'Austria',
            'BA': 'Bosnia and Herzegovina',
            'BE': 'Belgium',
            'BG': 'Bulgaria',
            'CH': 'Switzerland',
            'CY': 'Cyprus',
            'CZ': 'Czech Republic',
            'DE_LU': 'Germany/Luxembourg',
            'DK_1': 'Denmark (West)',
            'DK_2': 'Denmark (East)',
            'EE': 'Estonia',
            'ES': 'Spain',
            'FI': 'Finland',
            'FR': 'France',
            'GE': 'Georgia',
            'GR': 'Greece',
            'HR': 'Croatia',
            'HU': 'Hungary',
            'IE_SEM': 'Ireland (Single Electricity Market)',
            'IS': 'Iceland',
            'IT_CNOR': 'Italy (Central-North)',
            'IT_CSUD': 'Italy (Central-South)',
            'IT_NORD': 'Italy (North)',
            'IT_SUD': 'Italy (South)',
            'IT_SARD': 'Italy (Sardinia)',
            'IT_SICI': 'Italy (Sicily)',
            'LV': 'Latvia',
            'LT': 'Lithuania',
            'MD': 'Moldova',
            'MK': 'North Macedonia',
            'NL': 'Netherlands',
            'NO_1': 'Norway (Region 1)',
            'NO_2': 'Norway (Region 2)',
            'NO_3': 'Norway (Region 3)',
            'NO_4': 'Norway (Region 4)',
            'NO_5': 'Norway (Region 5)',
            'PL': 'Poland',
            'PT': 'Portugal',
            'RO': 'Romania',
            'RS': 'Serbia',
            'SE_1': 'Sweden (Region 1)',
            'SE_2': 'Sweden (Region 2)',
            'SE_3': 'Sweden (Region 3)',
            'SE_4': 'Sweden (Region 4)',
            'SI': 'Slovenia',
            'SK': 'Slovakia',
            'UA_IPS': 'Ukraine Integrated Power System',
            'XK': 'Kosovo',
        
            # ----- ADDED (from your new list / API docs) -----
            # country / region level and TSO-specific entries (kept long names from API where present)
            'DE': 'Germany',
            'DE_50hertz': '50Hertz CA, DE(50HzT) BZA',
            'DE_AT_LU': 'DE-AT-LU BZ',
            'DE_amprion': 'Amprion CA',
            'DE_tennet': 'TenneT GER CA',
            'DE_transnetbw': 'TransnetBW CA',
            'DK': 'Denmark',
            'DK_energinet': 'Denmark, Energinet CA',
            'EE': 'Estonia',    # already present above, kept original entry unchanged
            'ES': 'Spain',      # already present above, kept original entry unchanged
            'FI': 'Finland',    # already present above, kept original entry unchanged
            'FR': 'France',     # already present above, kept original entry unchanged
            # Great Britain / UK variants:
            'GB_GBN': 'National Grid BZ / CA/ MBA',   # mapped to same long-name as GB
            'GB_NIR': 'Northern Ireland, SONI CA',
            'GB_UKM': 'United Kingdom',               # mapped to UK / 'United Kingdom'
            'GR': 'Greece',    # already present above, kept original entry unchanged
            'HR': 'Croatia',   # already present above, kept original entry unchanged
            'HU': 'Hungary',   # already present above, kept original entry unchanged
            # Ireland (two variants requested):
            'IE': 'Ireland, EirGrid CA',
            'IE_sem': 'Ireland (SEM) BZ / MBA',
            # Italy top-level
            'IT': 'Italy, IT CA / MBA',
            'LU': 'Luxembourg, CREOS CA',
            'ME': 'Montenegro, CGES BZ / CA / MBA',
            'NO': 'Norway, Norway MBA, Stattnet CA',  # top-level Norway
            'UA': 'Ukraine, Ukraine BZ, MBA',
            'UA_east': 'Ukraine (East)',
            'UA_west': 'Ukraine (West)',
        }


    def _validate_date_range(self):
        """
        Ensure the start date is before the end date.

        Raises:
            ValueError: If the start date is not before the end date.
        """
        if self.start >= self.end:
            raise ValueError(f"Start date ({self.start}) must be before end date ({self.end})")

    def collect_load_data(self, output_filename: str = 'demand_single.csv'):
        """
        Collects hourly electricity load (demand) data for the specified countries,
        resamples to hourly resolution, and exports the data to a CSV file.

        Args:
            output_filename (str): Name of the output CSV file.
        """

        self.load_dict = {}                # Dictionary to store per-country load data
        self.combined_load_df = pd.DataFrame()   # Combined DataFrame for all countries
        failed_countries = []               # Track countries where data collection failed


        # Define output directory and save the combined load data
        output_dir = self.output_dir / 'load_data' / 'Lukas'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        
        self.logger.info(f"Collecting load data from {self.start} to {self.end} for: {self.countries_list}")

        for country in tqdm(self.countries_list, desc="Collecting load data"):
            try:
                # Query raw load data
                raw_load = self.client.query_load(country, start=self.start, end=self.end)

                # Resample to hourly resolution
                hourly_load = raw_load.resample('h').mean()

                # Convert timezone to UTC+0
                hourly_load.index = hourly_load.index.tz_convert('UTC')

                # Extract the relevant series
                if isinstance(hourly_load, pd.DataFrame) and 'Actual Load' in hourly_load.columns:
                    load_series = hourly_load['Actual Load']
                else:
                    load_series = hourly_load  # fallback if it's already a Series

                # Store in country-specific dictionary
                self.load_dict[country] = load_series

                # Add to combined dataframe
                self.combined_load_df[country] = load_series

                self.logger.info(f"Success: {country} with {len(load_series)} hourly data points")

            except Exception as e:
                self.logger.error(f"Failed to collect load data for {country}: {e}")
                failed_countries.append(country)


        self.combined_load_df.to_csv(output_path)
        self.logger.info(f"Electricity load data saved to {output_path}")

    def collect_spot_prices(self, output_filename: str = 'electricity_prices.csv'):
        """
        Collects hourly electricity spot prices for the specified countries and saves them into a single CSV file.
        
        Args:
            output_filename (str): Name of the output CSV file.
        """

        prices_by_country = {}              # Dictionary to store per-country price series
        combined_prices_df = pd.DataFrame() # DataFrame to hold combined price data across all countries
        failed_countries = []               # Track countries that fail to fetch data
        
        # Define output directory and write to CSV
        output_dir = self.output_dir / 'electricity_prices_data'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        self.logger.info(f"Collecting electricity price data from {self.start} to {self.end} for: {self.countries_list}")

        for country in tqdm(self.countries_list, desc='Collecting electricity prices'):
            try:
                # Query hourly day-ahead spot prices
                raw_prices = self.client.query_day_ahead_prices(
                    country, start=self.start, end=self.end
                )

                # Remove timezone info for consistency
                raw_prices.index = raw_prices.index.tz_convert('UTC')
                #raw_prices.index = raw_prices.index.tz_convert(None)   # This is to remove the timestamp of the UTC

                # Store per-country prices
                prices_by_country[country] = raw_prices

                # Add country prices as a column to the combined DataFrame
                combined_prices_df[country] = raw_prices

                self.logger.info(f"Success: {country} with {len(raw_prices)} data points")

            except Exception as e:
                self.logger.error(f"Failed to retrieve prices for {country}: {e}")
                failed_countries.append(country)

        # Add datetime-based features to the DataFrame for time-based grouping
        combined_prices_df = combined_prices_df.copy()
        combined_prices_df['year'] = combined_prices_df.index.year
        combined_prices_df['month'] = combined_prices_df.index.month
        combined_prices_df['week'] = combined_prices_df.index.isocalendar().week
        combined_prices_df['day'] = combined_prices_df.index.day
        combined_prices_df['hour'] = combined_prices_df.index.hour


        combined_prices_df.to_csv(output_path)

        self.logger.info(f"Electricity price data saved to {output_path}")

    def collect_generation_data(self, output_filename: str = 'generation'):
        """
        Collects hourly power generation data for the specified countries and exports both raw and resampled data to CSV files.
        
        - Saves raw data in:     <output_dir>/generation/generation_raw_data/
        - Saves hourly data in:  <output_dir>/generation/generation_hourly_data/
        
        If 'Hydro Pumped Storage' data is present, net storage is computed as:
            Net = Actual Aggregated - Actual Consumption
        and stored under 'Hydro Pumped Storage Net'.
        """
        
        # Initialize result containers
        self.generation_raw_data_dict = {}
        self.generation_hourly_data_dict = {}
        failed_countries = []

        # Define and create output directories
        raw_data_dir = self.output_dir / 'generation' / 'generation_raw_data'
        hourly_data_dir = self.output_dir / 'generation' / 'generation_hourly_data'
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        hourly_data_dir.mkdir(parents=True, exist_ok=True)

        # Log task overview
        self.logger.info(f"Collecting generation data from {self.start} to {self.end} for countries: {self.countries_list}")

        for country in tqdm(self.countries_list, desc='Fetching generation data'):
            try:
                # Fetch raw generation data from API/client
                raw_data = self.client.query_generation(country, start=self.start, end=self.end)
                self.generation_raw_data_dict[country] = raw_data
                self.logger.info(f"Success: {country} with {len(raw_data)} raw data points")

                # Save raw data to CSV
                raw_data_path = raw_data_dir / f'{output_filename}_{country}_raw_data.csv'
                raw_data.to_csv(raw_data_path)

                # Save raw data to CSV
                raw_data_path = raw_data_dir / f'{output_filename}_{country}_raw_data.csv'
                raw_data.to_csv(raw_data_path)

                # Handle Hydro Pumped Storage net calculation 
                if isinstance(raw_data.columns, pd.MultiIndex) and \
                ('Hydro Pumped Storage', 'Actual Aggregated') in raw_data.columns and \
                ('Hydro Pumped Storage', 'Actual Consumption') in raw_data.columns:
                    
                    net_pumped_hydro = raw_data[('Hydro Pumped Storage', 'Actual Aggregated')] - \
                                    raw_data[('Hydro Pumped Storage', 'Actual Consumption')]
                    raw_data[('Hydro Pumped Storage Net', '')] = net_pumped_hydro

                    # Drop both original Hydro Pumped Storage columns
                    raw_data.drop(columns=[
                        ('Hydro Pumped Storage', 'Actual Aggregated'),
                        ('Hydro Pumped Storage', 'Actual Consumption')
                    ], inplace=True)
                    
                    self.generation_raw_data_dict[country].drop(columns=[col for col in self.generation_raw_data_dict[country].columns if col[1] == 'Actual Consumption'], inplace=True)
                        
                    self.generation_raw_data_dict[country].columns = self.generation_raw_data_dict[country].columns.droplevel(-1)
                    
                # Drop all 'Actual Consumption' columns 
                if isinstance(raw_data.columns, pd.MultiIndex):
                    raw_data.drop(columns=[
                        col for col in raw_data.columns if col[1] == 'Actual Consumption'
                    ], inplace=True)

                    # Drop the second header level
                    raw_data.columns = raw_data.columns.droplevel(-1)
                    
                # Resample to hourly mean
                hourly_data = raw_data.resample('h').mean()
                hourly_data.index = hourly_data.index.tz_convert('UTC')  # Convert UTC +0
                self.generation_hourly_data_dict[country] = hourly_data

                # Save hourly data to CSV
                hourly_data_path = hourly_data_dir / f'{output_filename}_{country}_hourly_data.csv'
                hourly_data.to_csv(hourly_data_path)

            except Exception as e:
                self.logger.error(f"Failed to process {country}: {e}")
                failed_countries.append(country)   
                
    def collect_generation_capacity_data(self, output_filename_prefix: str = 'generation_capacity'):
        """
        Collects installed generation capacity per unit for specified countries and saves them as separate CSV files.

        Args:
            output_filename_prefix (str): Prefix for output CSV filenames.
        """

        self.capacities_by_country = {}  # Dictionary to store generation capacity data
        failed_countries = []       # List to keep track of countries that failed

        # Output directory for capacity data
        output_dir = self.output_dir / 'generation_capacity_data'
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Collecting installed generation capacity from {self.start} to {self.end} for: {self.countries_list}")

        for country in tqdm(self.countries_list, desc='Collecting generation capacities'):
            try:
                # Query installed generation capacity per unit
                capacity_df = self.client.query_installed_generation_capacity_per_unit(
                    country,
                    start=self.start,
                    end=self.end
                )

                # Store in dictionary
                self.capacities_by_country[country] = capacity_df

                # Define output file path for this country
                country_filename = f"{output_filename_prefix}_{country}.csv"
                output_path = output_dir / country_filename

                # Save to CSV
                capacity_df.to_csv(output_path)

                self.logger.info(f"Success: {country}, data saved to {output_path}")

            except Exception as e:
                self.logger.error(f"Failed to retrieve capacity data for {country}: {e}")
                failed_countries.append(country)           

#%%
# Example usage and testing
if __name__ == "__main__":
    # Basic usage with default parameters
    collector = EntsoeAPIDataCollector(
                start=pd.Timestamp('2018-01-01', tz='UTC'),
                end=pd.Timestamp('2025-07-01', tz='UTC'),
                output_dir='api_output',
                countries_list = [
                    # # 'AL', # no data available
                    # 'AT', # succsess
                    # 'BA',
                    # 'BE',
                    # 'BG',
                    # 'CH',
                    # 'CY',
                    # 'CZ',
                    # 'DE',
                    # 'DE_50HZ', #
                    # 'DE_AT_LU',
                    # 'DE_LU',
                    # 'DE_AMPRION',
                    # 'DE_TENNET',
                    #'DE_TRANSNET', # Failed, needs to be retrieved again and saved separately
                    #'DK',
                    #'DK_1',
                    #'DK_2',
                    # # 'DK_energinet', #invalid code
                    # 'DK_CA', #new energinet
                    # 'EE',
                    # 'ES',
                    # 'FI',
                    # 'FR',
                    # 'GB', 
                    # # 'GB_GBN', # invalid code
                    # # 'GB_NIR', # invalid code
                    # # 'GB_UKM', # invalid code
                    # 'GB_IFA', # the new GB ones you need to find out what they correspond to
                    # 'GB_IFA2', #  Never mind they are all not working, since UK left the EU
                    #   # Maybe try manually to see if reported
                    #'GR', # Greece is missing as well, because  I missed on comma
                    # 'HR',
                    # 'HU',
                    # 'IE',
                    # 'IE_sem',
                    # 'IT',     
                    # 'IT_CNOR',
                    # 'IT_CSUD',
                    # 'IT_NORD',
                    # 'IT_SARD',
                     'IT_SICI',
                    # 'IT_SUD',
                     'LT',
                     'LU',
                    # 'LV',
                     'ME',
                    # 'MK',  
                    # 'NL',
                     'NO',
                    # 'NO_1',
                    # 'NO_2',
                    # 'NO_3',
                    # 'NO_4',
                    # 'NO_5',
                    # 'PL',
                    # 'PT',
                     'RO',
                    # 'RS',
                     'SE',
                    # 'SE_1',
                    # 'SE_2',
                    # 'SE_3',
                    # 'SE_4',
                    # 'SI',
                    # 'SK', 
                    # 'UA', # no data
                    # # 'UA_east', # wrong key
                    # # 'UA_west',
                    # 'UA_DOBTPP', # no data
                    # 'UA_IPS', # no data
                    # 'UA_BEI', # no data
                    # 'XK'
                ]
                # countries_list = [
                #     'AL',      # Albania
                #     'BE',      # Belgium
                #     'CZ',      # Czech Republic
                #     'DK_1',    # Denmark (West)
                #     'FR',      # France
                #     'HR',      # Croatia
                #     'IS',      # Iceland
                #     'IT_CNOR', # Italy (Central‑North)
                #     'IT_CSUD', # Italy (Central‑South)
                #     'IT_NORD', # Italy (North)
                #     'IT_SARD', # Italy (Sardinia)
                #     'IT_SICI', # Italy (Sicily)
                #     'LT',      # Lithuania
                #     'MD',      # Moldova
                #     'RO',      # Romania
                #     'UA_IPS'   # Ukraine Integrated Power System
                # ]
    )

    #collector.collect_load_data()
    #collector.collect_spot_prices()
    collector.collect_generation_data()
    # collector.collect_generation_capacity_data()

# Mapping for Italy 
# """
#     IT =            '10YIT-GRTN-----B', 'Italy, IT CA / MBA',                           'Europe/Rome',
#     IT_SACO_AC =    '10Y1001A1001A885', 'Italy_Saco_AC',                                'Europe/Rome',
#     IT_CALA =   '10Y1001C--00096J', 'IT-Calabria BZ',                                'Europe/Rome',
#     IT_SACO_DC =    '10Y1001A1001A893', 'Italy_Saco_DC',                                'Europe/Rome',
#     IT_BRNN =       '10Y1001A1001A699', 'IT-Brindisi BZ',                               'Europe/Rome',
#     IT_CNOR =       '10Y1001A1001A70O', 'IT-Centre-North BZ',                           'Europe/Rome',
#     IT_CSUD =       '10Y1001A1001A71M', 'IT-Centre-South BZ',                           'Europe/Rome',
#     IT_FOGN =       '10Y1001A1001A72K', 'IT-Foggia BZ',                                 'Europe/Rome',
#     IT_GR =         '10Y1001A1001A66F', 'IT-GR BZ',                                     'Europe/Rome',
#     IT_MACRO_NORTH = '10Y1001A1001A84D', 'IT-MACROZONE NORTH MBA',                      'Europe/Rome',
#     IT_MACRO_SOUTH = '10Y1001A1001A85B', 'IT-MACROZONE SOUTH MBA',                      'Europe/Rome',
#     IT_MALTA =      '10Y1001A1001A877', 'IT-Malta BZ',                                  'Europe/Rome',
#     IT_NORD =       '10Y1001A1001A73I', 'IT-North BZ',                                  'Europe/Rome',
#     IT_NORD_AT =    '10Y1001A1001A80L', 'IT-North-AT BZ',                               'Europe/Rome',
#     IT_NORD_CH =    '10Y1001A1001A68B', 'IT-North-CH BZ',                               'Europe/Rome',
#     IT_NORD_FR =    '10Y1001A1001A81J', 'IT-North-FR BZ',                               'Europe/Rome',
#     IT_NORD_SI =    '10Y1001A1001A67D', 'IT-North-SI BZ',                               'Europe/Rome',
#     IT_PRGP =       '10Y1001A1001A76C', 'IT-Priolo BZ',                                 'Europe/Rome',
#     IT_ROSN =       '10Y1001A1001A77A', 'IT-Rossano BZ',                                'Europe/Rome',
#     IT_SARD =       '10Y1001A1001A74G', 'IT-Sardinia BZ',                               'Europe/Rome',
#     IT_SICI =       '10Y1001A1001A75E', 'IT-Sicily BZ',                                 'Europe/Rome',
#     IT_SUD =        '10Y1001A1001A788', 'IT-South BZ',                                  'Europe/Rome',
    
# """
# =============================================================================
# # Generation capacity (MW)
# GEN_cap = {}
# countries_list = ['AT','BE','BG','CZ','DK_1','DK_2','EE','FI','FR',
#                  'DE_LU','GR','HU','IE_SEM','IT_CNOR','IT_CSUD','IT_NORD',
#                  'IT_SARD','LV','LT','NL','NO_1',
#                  'NO_2','NO_3','NO_4','NO_5','PL','PT','RO','RS','SK','SI','ES',
#                  'SE_1','SE_2','SE_3','SE_4','CH']
# for c in countries_list:
#     GEN_cap[c] = client.query_installed_generation_capacity_per_unit(c, start=start,end=end)
#     GEN_cap[c]['zone'] = c
# 
# GEN_cap_df = GEN_cap[countries_list[0]]
# for c in countries_list[1:]:
#     GEN_cap_df = pd.concat([GEN_cap_df, GEN_cap[c]], axis=0)
# 
# GEN_cap_df.to_csv('GEN_cap.csv')
# =============================================================================


#generation = client.query_generation(country_code, start=start,end=end)
#generation_per_plant = client.query_generation_per_plant(country_code_1, start=start,end=end)
#generation_forecast = client.query_generation_forecast(country_code_1, start=start,end=end)
#wind_solar_forecast = client.query_wind_and_solar_forecast(country_code_1, start=start,end=end, psr_type=None)
#installed_generation_capacity = client.query_installed_generation_capacity(country_code_1, start=start,end=end)
#installed_generation_capacity_per_unit = client.query_installed_generation_capacity_per_unit(country_code_1, start=start,end=end)


# =============================================================================
# #cross-border flows (physical) (MW): to get resulting flow both directions need to be considerd, e.g netflow_AT_DE = (AT-DE) - (DE-AT) 
# lines = [('DK_1', 'DK_2'),('DK_1', 'NL'),('DK_1', 'NO_2'),('DK_1', 'SE_3')
#          ,('DK_2', 'SE_4'),('SE_1', 'FI'),('SE_1', 'NO_4'),('SE_1', 'SE_2')
#          ,('SE_2', 'NO_3'),('SE_2', 'NO_4'),('SE_2', 'SE_3'),('SE_3', 'FI')
#          ,('SE_3', 'NO_1'),('SE_3', 'SE_4'),('SE_4', 'LT'),('SE_4', 'PL')
#          ,('NO_1', 'NO_2'),('NO_1', 'NO_3'),('NO_1', 'NO_5'),('NO_2', 'NL')
#          ,('NO_2', 'NO_5'),('NO_3', 'NO_4'),('NO_3', 'NO_5'),('FI', 'EE')
#          ,('AT', 'CH'),('AT', 'CZ'),('AT', 'HU'),('AT', 'IT'),('AT', 'SI')
#          ,('BE', 'FR'),('BE', 'NL'),('CZ', 'PL'),('CZ', 'SK'),('EE', 'LV')
#          ,('FR', 'IT'),('FR', 'ES')
#          #,('FR', 'IE')
#          ,('LT', 'LV'),('LT', 'PL')
#          ,('AL', 'GR'),('AL', 'ME')
#          #,('AL', 'MK')
#          ,('AL', 'RS'),('BA', 'HR')
#          ,('BA', 'ME'),('BA', 'RS'),('BG', 'GR')
#          #,('BG', 'MK')
#          ,('BG', 'RO')
#          ,('BG', 'RS'),('CH', 'FR'),('CH', 'IT'),('ES', 'PT'),('GR', 'IT')
#          #,('GR', 'MK')
#          ,('HR', 'HU'),('HR', 'RS'),('HR', 'SI'),('HU', 'RO')
#          ,('HU', 'RS')
#          #,('HU', 'SI')
#          ,('HU', 'SK')
#          #,('IE', 'NI')
#          ,('IT', 'ME')
#          ,('IT', 'SI'),('ME', 'RS')
#          #,('MK', 'RS')
#          ,('PL', 'SK'),('RO', 'RS')
#          ,('BE', 'LU')
#          #,('XK', 'AL'),('XK', 'ME'),('XK', 'MK')
#          ]
# 
# lines_DELU = [('DK_1', 'DE_LU'),('DK_2', 'DE_LU'),('SE_4', 'DE_LU'),('NO_2', 'DE_LU')
#               ,('AT', 'DE_LU'),('BE', 'DE_LU'),('CZ', 'DE_LU')
#               ,('DE_LU', 'FR')
#               ,('DE_LU', 'NL'),('DE_LU', 'PL'),('CH', 'DE_LU')]
# 
# lines_DEATLU = [('DK_1', 'DE_AT_LU'),('DK_2', 'DE_AT_LU'),('SE_4', 'DE_AT_LU')
#                 #,('NO_2', 'DE_AT_LU')
#                 #,('AT', 'DE_AT_LU')
#                 ,('BE', 'DE_AT_LU')
#                 ,('CZ', 'DE_AT_LU'),('DE_AT_LU', 'FR'),('DE_AT_LU', 'NL')
#                 ,('DE_AT_LU', 'PL'),('CH', 'DE_AT_LU')]
# 
# lines_UK = [('DK_1', 'UK'),('NO_2', 'UK'),('BE', 'UK'),('FR', 'UK'),('UK', 'IE')
#             ,('UK', 'NI'),('UK', 'NL'),('DE_LU', 'UK')]
# 
# start = pd.Timestamp('20150101', tz ='UTC')
# end = pd.Timestamp('20201231', tz ='UTC')
# =============================================================================

# =============================================================================
# for l in lines:
#     country_code_1 = l[0]
#     country_code_2 = l[1]
#     
#     crossborder_flows_1 = client.query_crossborder_flows(country_code_1, country_code_2, start=start,end=end)
#     crossborder_flows_2 = client.query_crossborder_flows(country_code_2,country_code_1, start=start,end=end) 
#     crossborder_flow_net = crossborder_flows_1 - crossborder_flows_2
#     
#     filename = 'API queries new/flow_{}-{}.csv'.format(country_code_1, country_code_2)
#     crossborder_flow_net.to_csv(filename)  
#     
# for l in lines_DELU:
#     country_code_1 = l[0]
#     country_code_2 = l[1]
#     
#     crossborder_flows_1 = client.query_crossborder_flows(country_code_1, country_code_2, start=start,end=end)
#     crossborder_flows_2 = client.query_crossborder_flows(country_code_2,country_code_1, start=start,end=end) 
#     crossborder_flow_net = crossborder_flows_1 - crossborder_flows_2
#     
#     filename = 'API queries new/flow_{}-{}.csv'.format(country_code_1, country_code_2)
#     crossborder_flow_net.to_csv(filename)
# 
# for l in lines_DEATLU:
#     country_code_1 = l[0]
#     country_code_2 = l[1]
#     
#     crossborder_flows_1 = client.query_crossborder_flows(country_code_1, country_code_2, start=start,end=end)
#     crossborder_flows_2 = client.query_crossborder_flows(country_code_2,country_code_1, start=start,end=end) 
#     crossborder_flow_net = crossborder_flows_1 - crossborder_flows_2
#     
#     filename = 'API queries new/flow_{}-{}.csv'.format(country_code_1, country_code_2)
#     crossborder_flow_net.to_csv(filename)
# 
# for l in lines_UK:
#     country_code_1 = l[0]
#     country_code_2 = l[1]
#     
#     crossborder_flows_1 = client.query_crossborder_flows(country_code_1, country_code_2, start=start,end=end)
#     crossborder_flows_2 = client.query_crossborder_flows(country_code_2,country_code_1, start=start,end=end) 
#     crossborder_flow_net = crossborder_flows_1 - crossborder_flows_2
#     
#     filename = 'API queries new/flow_{}-{}.csv'.format(country_code_1, country_code_2)
#     crossborder_flow_net.to_csv(filename)
# =============================================================================


# =============================================================================
# for l in lines[28:]:
#     country_code_1 = l[0]
#     country_code_2 = l[1]
#     
#     scheduled_exchanges_1 = client.query_scheduled_exchanges(country_code_1, country_code_2, start=start,end=end)
#     filename = 'API queries new/dayahead flow_{}-{}.csv'.format(country_code_1, country_code_2)
#     scheduled_exchanges_1.to_csv(filename)
#     
# for l in lines_DELU:
#     country_code_1 = l[0]
#     country_code_2 = l[1]
#     
#     scheduled_exchanges_1 = client.query_scheduled_exchanges(country_code_1, country_code_2, start=start,end=end)
#     filename = 'API queries new/dayahead flow_{}-{}.csv'.format(country_code_1, country_code_2)
#     scheduled_exchanges_1.to_csv(filename)
# 
# for l in lines_DEATLU:
#     country_code_1 = l[0]
#     country_code_2 = l[1]
#     
#     scheduled_exchanges_1 = client.query_scheduled_exchanges(country_code_1, country_code_2, start=start,end=end)
#     filename = 'API queries new/dayahead flow_{}-{}.csv'.format(country_code_1, country_code_2)
#     scheduled_exchanges_1.to_csv(filename)
# 
# for l in lines_UK:
#     country_code_1 = l[0]
#     country_code_2 = l[1]
#     
#     scheduled_exchanges_1 = client.query_scheduled_exchanges(country_code_1, country_code_2, start=start,end=end)
#     filename = 'API queries new/dayahead flow_{}-{}.csv'.format(country_code_1, country_code_2)
#     scheduled_exchanges_1.to_csv(filename)
# =============================================================================

#works only for countries_list without flow-based border (MW)
#net_transfer_capacity_dayahead = client.query_net_transfer_capacity_dayahead(country_code_1, country_code_3, start=start,end=end)
#net_transfer_capacity_monthahead = client.query_net_transfer_capacity_monthahead(country_code_1, country_code_3, start=start,end=end)
#net_transfer_capacity_weekahead = client.query_net_transfer_capacity_weekahead(country_code_1, country_code_3, start=start,end=end)
#net_transfer_capacity_yearahead = client.query_net_transfer_capacity_yearahead(country_code_1, country_code_3, start=start,end=end)

#contracted reserves (MW) and prices (€/MW/period)
#contracted_reserve_amount = client.query_contracted_reserve_amount(country_code_1, start=start, end=end, type_marketagreement_type='A01')
#contracted_reserve_prices = client.query_contracted_reserve_prices(country_code_1, start=start, end=end, type_marketagreement_type='A01')

# =============================================================================
# reserves = {}
# countries_list = ['AT','DE_LU','IT','NL','CH']
# 
# for c in countries_list:
#     reserves = client.query_contracted_reserve_amount(c, start=start, end=end, type_marketagreement_type='A01')
#     file_name = 'reserves_VOL_{}.csv'.format(c)
#     reserves.to_csv(file_name)
# 
# reserves = {}
# for c in countries_list[0:2]:
#     reserves = client.query_contracted_reserve_prices(c, start=start, end=end, type_marketagreement_type='A01')
#     file_name = 'reserves_PRICE_{}.csv'.format(c)
#     reserves.to_csv(file_name)
# =============================================================================


# =============================================================================
# # unavailability of generation and production units
# Unavailability = {}
# 
# for c in countries_list:
#     try:
#         Unavailability[c] = client.query_unavailability_of_generation_units(c, start=start,end=end)
#     except:
#         pass
# 
# Unavailability_df = Unavailability[list(Unavailability.keys())[0]]
# for c in list(Unavailability.keys())[1:]:
#     Unavailability_df = pd.concat([Unavailability_df, Unavailability[c]], axis=0)
# 
# Unavailability_df.to_csv('Unavailability.csv')
# =============================================================================

#unavailability_of_generation_units = client.query_unavailability_of_generation_units(country_code_1, start=start,end=end)
#unavailability_of_production_units = client.query_unavailability_of_production_units(country_code_1, start=start,end=end)


# =============================================================================
# country_code_from = 'DK_1'
# country_code_to = 'DK_2'
# start = pd.Timestamp('20150101', tz ='UTC')
# end = pd.Timestamp('20201231', tz ='UTC')
# 
# ts = client.query_unavailability_transmission(country_code_from, country_code_to, start=start, end=end, docstatus=None, periodstartupdate=None, periodendupdate=None)
# filename = 'Unavailability {} - {} {}.csv'.format(country_code_from, country_code_to, start.year)
# ts.to_csv(filename)
# =============================================================================


# =============================================================================
# # methods that return Pandas Series
# client.query_day_ahead_prices(country_code, start=start,end=end)
# client.query_net_position(country_code, start=start, end=end, dayahead=True)
# client.query_crossborder_flows(country_code_from, country_code_to, start, end)
# client.query_scheduled_exchanges(country_code_from, country_code_to, start, end, dayahead=False)
# client.query_net_transfer_capacity_dayahead(country_code_from, country_code_to, start, end)
# client.query_net_transfer_capacity_weekahead(country_code_from, country_code_to, start, end)
# client.query_net_transfer_capacity_monthahead(country_code_from, country_code_to, start, end)
# client.query_net_transfer_capacity_yearahead(country_code_from, country_code_to, start, end)
# client.query_intraday_offered_capacity(country_code_from, country_code_to, start, end,implicit=True)
# client.query_offered_capacity(country_code_from, country_code_to, start, end, contract_marketagreement_type, implicit=True)
# client.query_aggregate_water_reservoirs_and_hydro_storage(country_code, start, end)
# 
# # methods that return Pandas DataFrames
# client.query_load(country_code, start=start,end=end)
# client.query_load_forecast(country_code, start=start,end=end)
# client.query_load_and_forecast(country_code, start=start, end=end)
# client.query_generation_forecast(country_code, start=start,end=end)
# client.query_wind_and_solar_forecast(country_code, start=start,end=end, psr_type=None)
# client.query_generation(country_code, start=start,end=end, psr_type=None)
# client.query_generation_per_plant(country_code, start=start,end=end, psr_type=None)
# client.query_installed_generation_capacity(country_code, start=start,end=end, psr_type=None)
# client.query_installed_generation_capacity_per_unit(country_code, start=start,end=end, psr_type=None)
# client.query_imbalance_prices(country_code, start=start,end=end, psr_type=None)
# client.query_contracted_reserve_prices(country_code, start, end, type_marketagreement_type, psr_type=None)
# client.query_contracted_reserve_amount(country_code, start, end, type_marketagreement_type, psr_type=None)
# client.query_unavailability_of_generation_units(country_code, start=start,end=end, docstatus=None, periodstartupdate=None, periodendupdate=None)
# client.query_unavailability_of_production_units(country_code, start, end, docstatus=None, periodstartupdate=None, periodendupdate=None)
# client.query_unavailability_transmission(country_code_from, country_code_to, start, end, docstatus=None, periodstartupdate=None, periodendupdate=None)
# client.query_withdrawn_unavailability_of_generation_units(country_code, start, end)
# client.query_import(country_code, start, end)
# client.query_generation_import(country_code, start, end)
# client.query_procured_balancing_capacity(country_code, start, end, process_type, type_marketagreement_type=None)
# =============================================================================

# %%
