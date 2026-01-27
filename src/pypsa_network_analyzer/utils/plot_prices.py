import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

root_path = Path(__file__).parent.parent.resolve()

##############################################################################################
# Nature/Okabe-Ito colorblind-friendly palette
nature_orange = '#e69f00'
nature_sky_blue = '#56b4e9'
nature_bluish_green = '#009e73'
nature_yellow = '#f0e442'
nature_blue = '#0072b2'
nature_vermillion = '#d55e00'
nature_reddish_purple = '#cc79a7'

# Color mapping for datasets
dataset_colors = [
    nature_orange,
    nature_bluish_green,
    nature_yellow,
    nature_blue,
    nature_vermillion
]

# Country colors for overview plot
country_colors = {
    'DE': nature_vermillion,
    'FR': nature_blue,
    'IT': nature_orange,
    'ES': nature_bluish_green,
    'PL': nature_yellow,
    'DK': nature_reddish_purple,
    'SE': nature_sky_blue
}

# Configure plot settings with Arial font
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

##################################################################################################

x_length = 10
y_length = x_length / 1.618


def plot_aggregated_prices(
    df_benchmark: pd.DataFrame,
    countries: List[str],
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    resample_option: str = 'M'
):
    """
    Plot aggregated electricity prices for multiple countries.
    
    Parameters:
    -----------
    df_benchmark : pd.DataFrame
        Benchmark dataframe with datetime index and country columns
    countries : List[str]
        List of country codes to plot (e.g., ['DE', 'FR', 'IT'])
    year_start : Optional[int]
        Start year for slicing data (e.g., 2020)
    year_end : Optional[int]
        End year for slicing data (e.g., 2023)
    resample_option : str
        Resampling frequency ('M' for monthly, 'W' for weekly, 'D' for daily)
    """
    # Process benchmark data
    df_plot = df_benchmark.copy()
    df_plot = df_plot.resample(resample_option).mean()
    df_plot = df_plot[countries]
    
    # Apply year slicing if specified
    if year_start:
        df_plot = df_plot[df_plot.index.year >= year_start]
    if year_end:
        df_plot = df_plot[df_plot.index.year <= year_end]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(x_length, y_length))
    ax.grid(linestyle=':', alpha=0.8)
    
    for country in countries:
        ax.plot(
            df_plot.index,
            df_plot[country],
            linewidth=1.5,
            label=country,
            color=country_colors.get(country, nature_blue)
        )
    
    ax.set_title('Average monthly electricity prices\n(EUR/MWh)', loc='left', fontsize=12)
    ax.set_xlim(df_plot.index.min(), df_plot.index.max())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best', fontsize=10, frameon=True)
    
    fig.tight_layout()
    plt.show()


def plot_country_comparison_subplots(
    df_benchmark: pd.DataFrame,
    countries: List[str],
    datasets: List[str],
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    resample_option: str = 'M'
):
    """
    Plot comparison of benchmark vs simulations for each country (separate figure per country).
    
    Parameters:
    -----------
    df_benchmark : pd.DataFrame
        Benchmark dataframe with datetime index and country columns
    countries : List[str]
        List of country codes to plot
    datasets : List[str]
        List of dataset names to compare (e.g., ['hindcast-dyn', 'hindcast-std'])
    year_start : Optional[int]
        Start year for slicing data
    year_end : Optional[int]
        End year for slicing data
    resample_option : str
        Resampling frequency ('M' for monthly, 'W' for weekly, 'D' for daily)
    """
    # Process benchmark data
    df_bench = df_benchmark.copy()
    df_bench = df_bench.resample(resample_option).mean()
    df_bench = df_bench[countries]
    
    # Apply year slicing
    if year_start:
        df_bench = df_bench[df_bench.index.year >= year_start]
    if year_end:
        df_bench = df_bench[df_bench.index.year <= year_end]
    
    for country in countries:
        # Create subplot grid (2x3 for 6 datasets)
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(x_length * 1.5, y_length * 1.2))
        axs = axs.flatten()
        
        fig.suptitle(f'Electricity Price Comparison: {country}',  fontsize=14, fontweight='bold')
        
        for idx, dataset in enumerate(datasets):
            # Load simulation data
            df_sim = pd.read_csv(
                root_path / 'simulations' / f'{dataset}/results_concat' / 'combined_electricity_prices.csv',
                index_col=0,
                parse_dates=True
            )
            
            # Resample simulation data
            df_sim = df_sim.resample(resample_option).mean()
            
            # Apply year slicing
            if year_start:
                df_sim = df_sim[df_sim.index.year >= year_start]
            if year_end:
                df_sim = df_sim[df_sim.index.year <= year_end]
            
            # Plot benchmark (sky blue for reference standard)
            axs[idx].plot(
                df_bench.index,
                df_bench[country],
                linewidth=1,
                label='Benchmark',
                color=nature_sky_blue,
                alpha=0.9
            )
            
            # Plot simulation
            if country in df_sim.columns:
                axs[idx].plot(
                    df_sim.index,
                    df_sim[country],
                    linewidth=1,
                    label=dataset,
                    color=dataset_colors[idx % len(dataset_colors)],
                    alpha=0.8,
                    #linestyle='--'
                )
            
            axs[idx].set_title(f'Electricity prices daily avg. - {dataset} vs benchmark', fontsize=10, loc='left')
            axs[idx].grid(linestyle=':', alpha=0.6)
            axs[idx].spines['top'].set_visible(False)
            axs[idx].spines['right'].set_visible(False)
            axs[idx].legend(loc='best', fontsize=8, frameon=True)
            axs[idx].set_ylabel('EUR/MWh', fontsize=9)
        
        # Hide unused subplots
        for idx in range(len(datasets), len(axs)):
            axs[idx].set_visible(False)
        
        fig.tight_layout()
        plt.show()


def plot_dataset_with_multiple_countries(
    df_benchmark: pd.DataFrame,
    dataset: str,
    countries: List[str] = ['DE', 'IT', 'ES', 'DK'],
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    resample_option: str = 'M'
):
    """
    Plot one dataset with multiple countries in subplots (one subplot per country).
    
    Parameters:
    -----------
    df_benchmark : pd.DataFrame
        Benchmark dataframe with datetime index and country columns
    dataset : str
        Dataset name (e.g., 'hindcast-dyn')
    countries : List[str]
        List of country codes to plot (default: ['DE', 'IT', 'ES', 'DK'])
    year_start : Optional[int]
        Start year for slicing data
    year_end : Optional[int]
        End year for slicing data
    resample_option : str
        Resampling frequency ('M' for monthly, 'W' for weekly, 'D' for daily)
    """
    # Country name mapping
    country_names = {
        'DE': 'Germany',
        'IT': 'Italy',
        'ES': 'Spain',
        'DK': 'Denmark',
        'FR': 'France',
        'PL': 'Poland',
        'SE': 'Sweden'
    }
    
    # Determine frequency label
    freq_label = {
        'D': 'Daily',
        'W': 'Weekly',
        'M': 'Monthly',
        'H': 'Hourly'
    }.get(resample_option, resample_option.lower())
    
    # Process benchmark data
    df_bench = df_benchmark.copy()
    df_bench = df_bench.resample(resample_option).mean()
    df_bench = df_bench[countries]
    
    # Load simulation data
    df_sim = pd.read_csv(
        root_path / 'simulations' / f'{dataset}/results_concat' / 'combined_electricity_prices.csv',
        index_col=0,
        parse_dates=True
    )
    
    # Resample simulation data
    df_sim = df_sim.resample(resample_option).mean()
    
    # Apply year slicing
    if year_start:
        df_bench = df_bench[df_bench.index.year >= year_start]
        df_sim = df_sim[df_sim.index.year >= year_start]
    if year_end:
        df_bench = df_bench[df_bench.index.year <= year_end]
        df_sim = df_sim[df_sim.index.year <= year_end]
    
    # Create subplot grid (2x2 for 4 countries)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(x_length * 1.2, y_length * 1.2)) #######################
    axs = axs.flatten()###################################################################
    
    for idx, country in enumerate(countries):
        # Plot benchmark (sky blue for reference standard)
        axs[idx].plot(
            df_bench.index,
            df_bench[country],
            linewidth=1,
            label=f'Benchmark {country}',
            color=nature_sky_blue,
            alpha=0.9,
        )
        
        # Plot simulation
        if country in df_sim.columns:
            axs[idx].plot(
                df_sim.index,
                df_sim[country],
                linewidth=1.5,
                label=dataset,
                color=country_colors.get(country, nature_orange),
                alpha=0.8,

            )
            #axs[idx].set_xlim(df_bench.index.min(), df_bench.index.max())
            axs[idx].set_ylim(-50, 750)
        
        # Set title with country name and frequency #######################################################################
        country_full_name = country_names.get(country, country)
        axs[idx].set_title(f'{country_full_name} - {freq_label} average electricity prices', 
                          fontsize=10, loc='left')
        axs[idx].grid(linestyle=':', alpha=0.6)
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['right'].set_visible(False)
        axs[idx].legend(loc='best', fontsize=8, frameon=True)
        axs[idx].set_ylabel('EUR/MWh', fontsize=9)
    
    fig.tight_layout()
    plt.show()

def plot_dataset_with_multiple_datasets_single_graph(
    df_benchmark: pd.DataFrame,
    datasets: List[str],
    country: str = 'DE',
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    resample_option: str = 'M'
):
    """
    Plot benchmark and multiple simulation datasets for a single country in one graph.

    Parameters:
    -----------
    df_benchmark : pd.DataFrame
        Benchmark dataframe with datetime index and country columns.
    datasets : List[str]
        List of dataset names to compare (e.g., ['hindcast-dyn', 'hindcast-std']).
    country : str
        Country code to plot (e.g., 'DE').
    year_start : Optional[int]
        Start year for slicing data.
    year_end : Optional[int]
        End year for slicing data.
    resample_option : str
        Resampling frequency ('M', 'W', 'D', etc.).
    """

    # Frequency label for title
    freq_label = {
        'D': 'daily',
        'W': 'weekly',
        'M': 'monthly',
        'H': 'hourly'
    }.get(resample_option, resample_option.lower())

    # -------- Benchmark preprocessing --------
    df_bench = (
        df_benchmark[country]
        .resample(resample_option)
        .mean()
    )

    # Year slicing
    if year_start:
        df_bench = df_bench[df_bench.index.year >= year_start]
    if year_end:
        df_bench = df_bench[df_bench.index.year <= year_end]

    # -------- Plot setup --------
    fig, ax = plt.subplots(figsize=(x_length * 1.2, y_length * 1.2))

    # Benchmark line
    ax.plot(
        df_bench.index,
        df_bench.values,
        linewidth=2,
        label=f'Benchmark {country}',
        color=nature_sky_blue,
        alpha=0.9
    )

    # -------- Simulation datasets --------
    for idx, dataset in enumerate(datasets):

        df_sim = pd.read_csv(
            root_path / 'simulations' / dataset / 'results_concat' / 'combined_electricity_prices.csv',
            index_col=0,
            parse_dates=True
        )

        if country not in df_sim.columns:
            continue  # Skip datasets without this country

        df_sim = df_sim.resample(resample_option).mean()

        # Apply slicing
        if year_start:
            df_sim = df_sim[df_sim.index.year >= year_start]
        if year_end:
            df_sim = df_sim[df_sim.index.year <= year_end]

        ax.plot(
            df_sim.index,
            df_sim[country].values,
            linewidth=1.5,
            label=dataset,
            color=dataset_colors[idx % len(dataset_colors)],
            linestyle='--',
            alpha=0.8
        )

    # -------- Final formatting --------
    ax.set_title(
        f'{country} – {freq_label} average electricity prices\nBenchmark vs multiple simulations',
        fontsize=12,
        loc='left'
    )
    ax.set_ylabel('EUR/MWh', fontsize=10)
    ax.grid(linestyle=':', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=9, frameon=True)
    fig.tight_layout()

    plt.show()

# Example usage:
if __name__ == "__main__":
    
    # Load benchmark data once
    df_benchmark = pd.read_csv(
        root_path / 'simulations' / 'benchmark/electricity_prices.csv',
        index_col=0,
        parse_dates=True
    )
    
    # 1. Plot aggregated prices for selected countries
    # plot_aggregated_prices(
    #     df_benchmark=df_benchmark,
    #     countries=['DE', 'FR', 'IT', 'ES', 'PL', 'DK', 'SE'],
    #     year_start=2020,
    #     year_end=2025,
    #     resample_option='M'
    # )
    
    # 2. Plot country-by-country comparison with simulations
    datasets = [
        # 'hindcast_dyn_old',
        # 'hindcast_std_old',
        'hindcast-dyn',
        'hindcast-dyn-rolling',
        'hindcast-dyn-spec-cap',
        'hindcast-dyn-spec-cap-rolling',
        # 'hindcast-dyn-2022',
        # 'hindcast-std'
        'hindcast-dyn-spec-cap-irena',
        'hindcast-dyn-spec-cap-rolling-irena'

    ]
    
    plot_country_comparison_subplots(
        df_benchmark=df_benchmark,
        countries=['DE', 'IT', 'ES', 'DK'],
        datasets=datasets,
        year_start=2020,
        year_end=2024,
        resample_option='D'
    )
    
    
    # plot_dataset_with_multiple_datasets_single_graph(
    # df_benchmark=df_benchmark,
    # datasets=[
    #     'hindcast-dyn',
    #     'hindcast-dyn-rolling',
    #     'hindcast-dyn-spec-cap',
    #     'hindcast-dyn-spec-cap-rolling',
    #     'hindcast-dyn-2022',
    #     'hindcast-std'
    # ],
    # country='DE',
    # year_start=2020,
    # year_end=2024,
    # resample_option='D'
# )
    # 3. Plot one dataset with multiple countries
    
    
    # for dataset in datasets:
        
    #     plot_dataset_with_multiple_countries(
    #         df_benchmark=df_benchmark,
    #         dataset=dataset,
    #         countries=['DE', 'IT', 'ES', 'DK'],
    #         year_start=2020,
    #         year_end=2024,
    #         resample_option='D'
    #     )