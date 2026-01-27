#%%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
#%%
scenarios = ['hindcast_std_old', 'hindcast_dyn_old', 'hindcast-std', 'hindcast-dyn', 'hindcast-dyn-rolling', 'hindcast-dyn-2022', 'hindcast-dyn-spec-cap', 'hindcast-dyn-spec-cap-rolling']

#%%
# Get parent directory
base_path = Path(__file__).resolve().parent.parent

scenario_co2_dfs = {}

for scenario in scenarios:
    years = [2020, 2021, 2022, 2023, 2024]
    if scenario == 'hindcast-dyn-2022':
        years = [2022]
    parts = []
    for year in years:
        file_path = f"simulations/{scenario}/weather_year_{year}/results/summary/co2_emissions_by_country.csv"
        full_path = base_path / file_path
        # read and parse snapshot as datetime
        df = pd.read_csv(full_path, parse_dates=['snapshot'])
        # set snapshot as index
        df = df.set_index('snapshot')
        parts.append(df)

    # concat along the time index (no ignore_index)
    df_all = pd.concat(parts, axis=0)
    # sort by datetime index
    df_all = df_all.sort_index()

    # report overlap / duplicates if any, and then drop duplicates keeping first
    dup_count = df_all.index.duplicated().sum()
    if dup_count:
        print(f"{scenario}: found {dup_count} duplicate timestamps — keeping first occurrence for each.")
        df_all = df_all[~df_all.index.duplicated(keep='first')]

    scenario_co2_dfs[scenario] = df_all
    print(f"{scenario}: concatenated shape = {df_all.shape}")

df_std_old_co2 = scenario_co2_dfs['hindcast_std_old']
df_dyn_old_co2 = scenario_co2_dfs['hindcast_dyn_old']
df_std_co2 = scenario_co2_dfs['hindcast-std']
df_dyn_co2 = scenario_co2_dfs['hindcast-dyn']
df_dyn_rolling_co2 = scenario_co2_dfs['hindcast-dyn-rolling']
df_dyn_2022_co2 = scenario_co2_dfs['hindcast-dyn-2022']
df_dyn_spec_cap_co2 = scenario_co2_dfs['hindcast-dyn-spec-cap']
df_dyn_spec_cap_rolling_co2 = scenario_co2_dfs['hindcast-dyn-spec-cap-rolling']


# Get the load data to calculate CO2 intensity
scenario_load_dfs = {}
for scenario in scenarios:
    years = [2020, 2021, 2022, 2023, 2024]
    parts = []
    if scenario == 'hindcast-dyn-2022':
        years = [2022]
    for year in years:
        file_path = f"simulations/{scenario}/weather_year_{year}/results/summary/electric_load.csv"
        full_path = base_path / file_path
        # read and parse snapshot as datetime
        df_load = pd.read_csv(full_path, parse_dates=['snapshot'])
        # set snapshot as index
        df_load = df_load.set_index('snapshot')
        parts.append(df_load)

    # concat along the time index (no ignore_index)
    df_all_load = pd.concat(parts, axis=0)
    # sort by datetime index
    df_all_load = df_all_load.sort_index()

    # report overlap / duplicates if any, and then drop duplicates keeping first
    dup_count = df_all_load.index.duplicated().sum()
    if dup_count:
        print(f"{scenario} load: found {dup_count} duplicate timestamps — keeping first occurrence for each.")
        df_all_load = df_all_load[~df_all_load.index.duplicated(keep='first')]

    scenario_load_dfs[scenario] = df_all_load
    print(f"{scenario} load: concatenated shape = {df_all_load.shape}")


df_std_load_old = scenario_load_dfs['hindcast_std_old']
df_dyn_load_old = scenario_load_dfs['hindcast_dyn_old']
df_std_load = scenario_load_dfs['hindcast-std']
df_dyn_load = scenario_load_dfs['hindcast-dyn']
df_dyn_rolling_load = scenario_load_dfs['hindcast-dyn-rolling']
df_dyn_2022_load = scenario_load_dfs['hindcast-dyn-2022']
df_dyn_spec_cap_load = scenario_load_dfs['hindcast-dyn-spec-cap']
df_dyn_spec_cap_rolling_load = scenario_load_dfs['hindcast-dyn-spec-cap-rolling']

# Calculate CO2 intensity (emissions / load) 
# since columns and rows are aligned, this works directly
df_co2_int_std_old = df_std_old_co2.div(df_std_load_old)
df_co2_int_dyn_old =  df_dyn_old_co2.div(df_dyn_load_old)
df_co2_int_std = df_std_co2.div(df_std_load)
df_co2_int_dyn = df_dyn_co2.div(df_dyn_load)
df_co2_int_dyn_rolling = df_dyn_rolling_co2.div(df_dyn_rolling_load)
df_co2_int_dyn_2022 = df_dyn_2022_co2.div(df_dyn_2022_load)
df_co2_int_dyn_spec_cap = df_dyn_spec_cap_co2.div(df_dyn_spec_cap_load)
df_co2_int_dyn_spec_cap_rolling = df_dyn_spec_cap_rolling_co2.div(df_dyn_spec_cap_rolling_load)

# The result is in tons CO2 per MWh, but we want g per kWh
df_co2_int_std_old = df_co2_int_std_old * 1000 
df_co2_int_dyn_old = df_co2_int_dyn_old * 1000 
df_co2_int_std = df_co2_int_std * 1000
df_co2_int_dyn = df_co2_int_dyn * 1000
df_co2_int_dyn_rolling = df_co2_int_dyn_rolling * 1000
df_co2_int_dyn_2022 = df_co2_int_dyn_2022 * 1000
df_co2_int_dyn_spec_cap = df_co2_int_dyn_spec_cap * 1000
df_co2_int_dyn_spec_cap_rolling = df_co2_int_dyn_spec_cap_rolling * 1000


#%%
# Read in the reference data
years_reference = [2021, 2022, 2023, 2024]
co2_ref_data = []
for year in years_reference:
    ref_file_path = f"data/CO2_emissions/aggregated_data/co2_intensity_{year}.csv"
    ref_full_path = base_path / ref_file_path
    # read and parse snapshot as datetime
    df_co2 = pd.read_csv(ref_full_path, parse_dates=['DateTime (UTC)'])
    print(df_co2.head())
    # set snapshot as index
    df_co2 = df_co2.set_index('DateTime (UTC)')
    co2_ref_data.append(df_co2)

df_co2_ref = pd.concat(co2_ref_data, axis=0)
df_co2_ref = df_co2_ref.sort_index()
print(f"Reference data: concatenated shape = {df_co2_ref.shape}")
# report overlap / duplicates if any, and then drop duplicates keeping first
dup_count_ref = df_co2_ref.index.duplicated().sum()
if dup_count_ref:
    print(f"Reference data: found {dup_count_ref} duplicate timestamps — keeping first occurrence for each.")
    df_co2_ref = df_co2_ref[~df_co2_ref.index.duplicated(keep='first')]


# Resample reference and co2 intensity data to daily frequency for smoother plots
df_co2_ref = df_co2_ref.resample('D').mean()
df_co2_int_std_old = df_co2_int_std_old.resample('D').mean()
df_co2_int_dyn_old = df_co2_int_dyn_old .resample('D').mean()
df_co2_int_std = df_co2_int_std.resample('D').mean()
df_co2_int_dyn = df_co2_int_dyn.resample('D').mean()
df_co2_int_dyn_rolling = df_co2_int_dyn_rolling.resample('D').mean()
df_co2_int_dyn_2022 = df_co2_int_dyn_2022.resample('D').mean()
df_co2_int_dyn_spec_cap = df_co2_int_dyn_spec_cap.resample('D').mean()
df_co2_int_dyn_spec_cap_rolling = df_co2_int_dyn_spec_cap_rolling.resample('D').mean()  


# Filter all datasets to only include columns present in the reference data
common_countries = df_co2_ref.columns.intersection(df_co2_int_std_old.columns).tolist()
df_co2_ref = df_co2_ref[common_countries]
df_co2_int_std_old = df_co2_int_std_old[common_countries]
df_co2_int_dyn_old = df_co2_int_dyn_old[common_countries]
df_co2_int_std = df_co2_int_std[common_countries]
df_co2_int_dyn = df_co2_int_dyn[common_countries]
df_co2_int_dyn_rolling = df_co2_int_dyn_rolling[common_countries]
df_co2_int_dyn_2022 = df_co2_int_dyn_2022[common_countries]
df_co2_int_dyn_spec_cap = df_co2_int_dyn_spec_cap[common_countries]
df_co2_int_dyn_spec_cap_rolling = df_co2_int_dyn_spec_cap_rolling[common_countries] 

#%%

# ========== CONFIGURATION ==========
# Define all plot configurations
plot_configs_full_period = [
    {
        "name": "Hindcast_Std_Dyn_Old_vs_Reference",
        "title_suffix": "Old Standard & Dynamic vs Reference",
        "datasets": ["df_co2_int_std_old", "df_co2_int_dyn_old", "df_co2_ref"],
        "labels": ["Hindcast Standard (Old)", "Hindcast Dynamic (Old)", "Reference Data"],
        "output_dir": "comparison_01_old"
    },
    {
        "name": "Hindcast_Std_Dyn_New_vs_Reference",
        "title_suffix": "Standard & Dynamic vs Reference",
        "datasets": ["df_co2_int_std", "df_co2_int_dyn", "df_co2_ref"],
        "labels": ["Hindcast Standard", "Hindcast Dynamic", "Reference Data"],
        "output_dir": "comparison_02_new"
    },
    {
        "name": "Hindcast_Dyn_Variants_v1",
        "title_suffix": "Dynamic Variants v1 vs Reference",
        "datasets": ["df_co2_int_dyn_old", "df_co2_int_dyn", "df_co2_int_dyn_rolling", "df_co2_ref"],
        "labels": ["Hindcast Dynamic (Old)", "Hindcast Dynamic", "Hindcast Dynamic Rolling", "Reference Data"],
        "output_dir": "comparison_03_dyn_variants_v1"
    },
    {
        "name": "Hindcast_Dyn_SpecCap",
        "title_suffix": "Dynamic & Spec Cap vs Reference",
        "datasets": ["df_co2_int_dyn", "df_co2_int_dyn_spec_cap", "df_co2_ref"],
        "labels": ["Hindcast Dynamic", "Hindcast Dynamic Spec Cap", "Reference Data"],
        "output_dir": "comparison_04_dyn_spec_cap"
    },
    {
        "name": "Hindcast_Dyn_Rolling_SpecCap",
        "title_suffix": "Dynamic Rolling & Spec Cap Rolling vs Reference",
        "datasets": ["df_co2_int_dyn_rolling", "df_co2_int_dyn_spec_cap_rolling", "df_co2_ref"],
        "labels": ["Hindcast Dynamic Rolling", "Hindcast Dynamic Spec Cap Rolling", "Reference Data"],
        "output_dir": "comparison_05_dyn_rolling_spec_cap"
    }
]

plot_configs_2022_only = [
    {
        "name": "2022_Hindcast_Dyn_Variants",
        "title_suffix": "2022 Dynamic Variants vs Reference",
        "datasets": ["df_co2_int_dyn_old", "df_co2_int_dyn", "df_co2_int_dyn_rolling", "df_co2_ref"],
        "labels": ["Hindcast Dynamic (Old)", "Hindcast Dynamic", "Hindcast Dynamic Rolling", "Reference Data"],
        "output_dir": "comparison_2022_01_dyn_variants",
        "year": 2022
    }
]

# ========== COLOR PALETTE CONFIGURATION ==========
# Nature/Okabe-Ito colorblind-friendly palette
nature_orange = '#e69f00'
nature_sky_blue = '#56b4e9'
nature_bluish_green = '#009e73'
nature_yellow = '#f0e442'
nature_blue = '#0072b2'
nature_vermillion = '#d55e00'
nature_reddish_purple = '#cc79a7'
nature_ref_green_dark = '#006400'
nature_warm_brown = '#a6761d'
nature_warm_rose = '#e377c2'

# Color mapping for datasets
dataset_color_map = {
    'df_co2_ref': nature_ref_green_dark,
    # standard (cool colors)
    'df_co2_int_std_old': nature_bluish_green,
    'df_co2_int_std': nature_sky_blue,
    # dynamic (warm colors, no cycling)
    'df_co2_int_dyn_old': nature_vermillion,
    'df_co2_int_dyn': nature_orange,
    'df_co2_int_dyn_rolling': nature_reddish_purple,
    'df_co2_int_dyn_2022': nature_yellow,
    'df_co2_int_dyn_spec_cap': nature_warm_brown,
    'df_co2_int_dyn_spec_cap_rolling': nature_warm_rose
}

# Configure plot settings with Arial font
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ========== PLOTTING HELPERS ==========

def create_color_mapping(dataset_names):
    """
    Create consistent color mapping for dataset names.
    Prefer explicit dataset_color_map; fallback to matplotlib prop cycle.
    Returns dict: { dataset_name: color_hex }
    """
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mapping = {}
    for i, name in enumerate(dataset_names):
        if name in dataset_color_map:
            mapping[name] = dataset_color_map[name]
        else:
            mapping[name] = prop_cycle[i % len(prop_cycle)]
    return mapping

# ========== PLOTTING FUNCTION ==========

def plot_by_country_and_aggregated(config, all_dfs_dict, countries, years_reference, 
                                   plot_base_dir, time_period_label="full_period"):
    """
    Generate plots for each country and an aggregated plot (averaged across all countries).
    
    Parameters:
    - config: Dictionary with plot configuration
    - all_dfs_dict: Dictionary mapping dataset names to dataframes
    - countries: List of country column names
    - years_reference: List of years with reference data available
    - plot_base_dir: Base directory for saving plots (Path or string)
    - time_period_label: Label for the time period (e.g., "full_period", "2022_only")
    """
    
    plot_base_dir = Path(plot_base_dir)
    output_dir = plot_base_dir / config["output_dir"] / time_period_label
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # create color mapping keyed by dataset name
    colors = create_color_mapping(config["datasets"])
    
    year_filter = config.get("year", None)  # None means full period, otherwise specific year
    
    # layout sizing variables (keeps close to original but uses requested formula)
    x_length = 14
    y_length = 6
    figsize = (x_length * 1.2, y_length * 1.2)
    
    # ========== PLOT FOR EACH COUNTRY ==========
    print(f"  Generating individual country plots for {config['name']}...")
    
    for country in countries:
        fig, ax = plt.subplots(figsize=figsize)
        
        plotted_any = False
        for dataset_name, label in zip(config["datasets"], config["labels"]):
            if dataset_name not in all_dfs_dict:
                # dataset not available in provided dict
                continue
            df = all_dfs_dict[dataset_name]
            
            # Check if country exists in this dataset
            if country not in df.columns:
                continue
            
            # Filter by year if specified
            if year_filter:
                start_date = pd.Timestamp(f'{year_filter}-01-01')
                end_date = pd.Timestamp(f'{year_filter}-12-31 23:59:59')
                df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
            else:
                df_filtered = df
            
            if len(df_filtered) == 0:
                continue
            
            color = colors.get(dataset_name)
            ax.plot(df_filtered.index, df_filtered[country], 
                    label=label, alpha=0.8, linewidth=1.2, linestyle='-', 
                    color=color)
            plotted_any = True
        
        if not plotted_any:
            plt.close(fig)
            continue
        
        # Apply the "similar layout" styling requested while keeping original labels
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('CO₂ Emissions (g/kWh)', fontsize=9)  # kept semantic label, fontsize adjusted
        period_info = f"({year_filter})" if year_filter else "(2020-2024)"
        ax.set_title(f'CO₂ Intensity: {country} {period_info} - {config["title_suffix"]}', 
                     fontsize=10, fontweight='bold', loc='left')
        
        # visual polish per user's snippet
        ax.grid(linestyle=':', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='best', fontsize=8, frameon=True)
        
        plt.tight_layout()
        plot_filename = f"{country}_comparison.png"
        plt.savefig(output_dir / plot_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # ========== AGGREGATED PLOT (AVERAGE ACROSS ALL COUNTRIES) ==========
    print(f"  Generating aggregated plot for {config['name']}...")
    
    fig, ax = plt.subplots(figsize=figsize)
    plotted_any = False
    
    for dataset_name, label in zip(config["datasets"], config["labels"]):
        if dataset_name not in all_dfs_dict:
            continue
        df = all_dfs_dict[dataset_name]
        
        # Filter by year if specified
        if year_filter:
            start_date = pd.Timestamp(f'{year_filter}-01-01')
            end_date = pd.Timestamp(f'{year_filter}-12-31 23:59:59')
            df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
        else:
            df_filtered = df
        
        # Ensure the countries exist in df_filtered
        available_countries = [c for c in countries if c in df_filtered.columns]
        if len(available_countries) == 0:
            continue
        
        # Calculate mean across available countries at each timestep
        df_mean = df_filtered[available_countries].mean(axis=1)
        color = colors.get(dataset_name)
        ax.plot(df_mean.index, df_mean.values, 
               label=label, alpha=0.8, linewidth=1.5, linestyle='-', 
               color=color)
        plotted_any = True
    
    if plotted_any:
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('CO₂ Emissions (g/kWh)', fontsize=9)
        period_info = f"({year_filter})" if year_filter else "(2020-2024)"
        ax.set_title(f'CO₂ Intensity: Aggregated (Average all countries) {period_info} - {config["title_suffix"]}', 
                     fontsize=10, fontweight='bold', loc='left')
        
        # apply similar layout/styling
        ax.grid(linestyle=':', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='best', fontsize=8, frameon=True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "AGGREGATED_all_countries.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"    Plots saved to: {output_dir}")

# ========== MAIN PLOTTING EXECUTION ==========

def run_all_plots(plot_dir, all_dfs_dict, countries, years_reference):
    """Execute all plotting configurations."""
    
    print("\n" + "="*70)
    print("GENERATING FULL PERIOD PLOTS (2020-2024)")
    print("="*70)
    
    for config in plot_configs_full_period:
        print(f"\nProcessing: {config['name']}")
        plot_by_country_and_aggregated(config, all_dfs_dict, countries, years_reference, 
                                      plot_dir, time_period_label="full_period")
    
    print("\n" + "="*70)
    print("GENERATING 2022-ONLY PLOTS")
    print("="*70)
    
    for config in plot_configs_2022_only:
        print(f"\nProcessing: {config['name']}")
        plot_by_country_and_aggregated(config, all_dfs_dict, countries, years_reference, 
                                      plot_dir, time_period_label="2022_only")
    
    print("\n" + "="*70)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*70)

#%%
## Execute plotting

plot_dir = base_path / "simulations" / "co2_comparison_plots"
plot_dir.mkdir(parents=True, exist_ok=True)

all_dfs_dict = {
    "df_co2_int_std_old": df_co2_int_std_old,
    "df_co2_int_dyn_old": df_co2_int_dyn_old,
    "df_co2_int_std": df_co2_int_std,
    "df_co2_int_dyn": df_co2_int_dyn,
    "df_co2_int_dyn_rolling": df_co2_int_dyn_rolling,
    "df_co2_int_dyn_2022": df_co2_int_dyn_2022,
    "df_co2_int_dyn_spec_cap": df_co2_int_dyn_spec_cap,
    "df_co2_int_dyn_spec_cap_rolling": df_co2_int_dyn_spec_cap_rolling,
    "df_co2_ref": df_co2_ref,
}

countries = df_co2_int_std_old.columns.tolist()
years_reference = [2021, 2022, 2023, 2024]

run_all_plots(plot_dir, all_dfs_dict, countries, years_reference)




# %%
# # ========== PLOTTING FUnctionality (updated: consistent colors per DATASET, solid lines only)
# # Create output directory for plots
# plot_dir = base_path / "simulations" / "co2_comparison_plots"
# plot_dir.mkdir(parents=True, exist_ok=True)

# # Get all country columns (assuming they're the same across all dataframes)
# countries = df_co2_int_std_old.columns.tolist()

# # Create a consistent color mapping for each dataset so the same dataset always uses the same color
# # Uses matplotlib's default property cycle (first three colors will be used for the three datasets)
# prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# dataset_order = ['Hindcast Standard', 'Hindcast Dynamic', 'Reference Data']
# dataset_colors = {name: prop_cycle[i % len(prop_cycle)] for i, name in enumerate(dataset_order)}

# print(f"Generating plots for {len(countries)} countries...")

# # ========== 1. FULL TIME PERIOD PLOTS (2020-2024) ==========
# print("Generating full time period plots (2020-2024)...")

# for country in countries:
#     fig, ax = plt.subplots(figsize=(14, 6))
    
#     # Plot hindcast_standard co2 intensity(full 2020-2024)
#     if country in df_co2_int_std_old.columns:
#         ax.plot(df_co2_int_std_old.index, df_co2_int_std_old[country], 
#                 label='Hindcast Standard', alpha=0.8, linewidth=1.2, linestyle='-', color=dataset_colors['Hindcast Standard'])
    
#     # Plot hindcast_dynamic (full 2020-2024)
#     if country in df_co2_int_dyn_old.columns:
#         ax.plot(df_co2_int_dyn_old.index, df_co2_int_dyn_old[country], 
#                 label='Hindcast Dynamic', alpha=0.8, linewidth=1.0, linestyle='-', color=dataset_colors['Hindcast Dynamic'])
    
#     # Plot reference data (2021-2024 only)
#     if country in df_co2_ref.columns:
#         ax.plot(df_co2_ref.index, df_co2_ref[country], 
#                 label='Reference Data', alpha=0.9, linewidth=1.8, linestyle='-', color=dataset_colors['Reference Data'])
    
#     ax.set_xlabel('Time', fontsize=12)
#     ax.set_ylabel('CO₂ Emissions in g/kWh', fontsize=12)
#     ax.set_title(f'CO₂ Emissions Comparison: {country} (2020-2024)', fontsize=14, fontweight='bold')
#     ax.legend(loc='best')
#     ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(plot_dir / f"{country}_full_period.png", dpi=150, bbox_inches='tight')
#     plt.show()
#     plt.close()

# print(f"Full period plots saved to: {plot_dir}")

# # ========== 2. YEARLY PLOTS FOR EACH COUNTRY ==========
# print("Generating yearly plots for each country...")

# yearly_plot_dir = plot_dir / "yearly"
# yearly_plot_dir.mkdir(parents=True, exist_ok=True)

# for country in countries:
#     for year in years:
#         fig, ax = plt.subplots(figsize=(12, 6))
        
#         # Determine date range for the year
#         start_date = pd.Timestamp(f'{year}-01-01')
#         end_date = pd.Timestamp(f'{year}-12-31 23:59:59')
        
#         # Check if reference data exists for this year (2021-2024)
#         has_reference = year in years_reference
        
#         if has_reference:
#             # Use reference index as x-axis
#             df_ref_year = df_co2_ref[(df_co2_ref.index >= start_date) & (df_co2_ref.index <= end_date)]
            
#             if len(df_ref_year) > 0 and country in df_co2_ref.columns:
#                 # Plot reference first (solid line — no dashes)
#                 ax.plot(df_ref_year.index, df_ref_year[country], 
#                         label='Reference Data', alpha=0.9, linewidth=1.8, linestyle='-', color=dataset_colors['Reference Data'])
                
#                 # Align hindcast data to reference index
#                 if country in df_co2_int_std_old.columns:
#                     df_std_aligned = df_co2_int_std_old.reindex(df_ref_year.index)
#                     ax.plot(df_ref_year.index, df_std_aligned[country], 
#                             label='Hindcast Standard', alpha=0.8, linewidth=1.2, linestyle='-', color=dataset_colors['Hindcast Standard'])
                
#                 if country in df_co2_int_dyn_old.columns:
#                     df_dyn_aligned = df_co2_int_dyn_old.reindex(df_ref_year.index)
#                     ax.plot(df_ref_year.index, df_dyn_aligned[country], 
#                             label='Hindcast Dynamic', alpha=0.8, linewidth=1.0, linestyle='-', color=dataset_colors['Hindcast Dynamic'])
#         else:
#             # No reference for 2020 - just plot hindcast data
#             df_std_year = df_co2_int_std_old[(df_co2_int_std_old.index >= start_date) & (df_co2_int_std_old.index <= end_date)]
#             df_dyn_year = df_co2_int_dyn_old[(df_co2_int_dyn_old.index >= start_date) & (df_co2_int_dyn_old.index <= end_date)]
            
#             if country in df_co2_int_std_old.columns and len(df_std_year) > 0:
#                 ax.plot(df_std_year.index, df_std_year[country], 
#                         label='Hindcast Standard', alpha=0.8, linewidth=1.2, linestyle='-', color=dataset_colors['Hindcast Standard'])
            
#             if country in df_co2_int_dyn_old.columns and len(df_dyn_year) > 0:
#                 ax.plot(df_dyn_year.index, df_dyn_year[country], 
#                         label='Hindcast Dynamic', alpha=0.8, linewidth=1.0, linestyle='-', color=dataset_colors['Hindcast Dynamic'])
        
#         ax.set_xlabel('Time', fontsize=12)
#         ax.set_ylabel('CO₂ Emissions in g/kWh', fontsize=12)
#         ax.set_title(f'CO₂ Emissions Comparison: {country} ({year})', fontsize=14, fontweight='bold')
#         ax.legend(loc='best')
#         ax.grid(True, alpha=0.3)
        
#         # Rotate x-axis labels for better readability
#         plt.xticks(rotation=45, ha='right')
        
#         plt.tight_layout()
#         # plt.show()
#         plt.savefig(yearly_plot_dir / f"{country}_{year}.png", dpi=150, bbox_inches='tight')
#         plt.close()

# print(f"Yearly plots saved to: {yearly_plot_dir}")
# print("Plot generation complete!")

# %%