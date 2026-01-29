from pathlib import Path
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
import logging


class NetworkAnalyzer:
    """
    A class to analyze CO₂ emissions and generation dispatch from a PyPSA network.
    """

    def __init__(self, 
                 simulation_folder: str, 
                 weather_year: str,
                 network_file: str = 'base_s_39_elec_Ept.nc',
                 plots_format_export: str = 'pdf',
                 logger: Optional[logging.Logger] = None
                 ):
        
        # Use default logger if none provided
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.logger.info(f"STARTING ANALISYS - {simulation_folder} | {weather_year}")
        
        # Read export file format of plot
        self.file_format = plots_format_export.lower()
        if self.file_format not in {"png", "pdf", "svg"}:
            raise ValueError("plots_format_export must be one of: 'png', 'pdf', 'svg'")

        # Set up directories and read network file
        self._set_directories(simulation_folder, network_file, weather_year)
        
        # Get network components
        self._get_network_components()
        
        # Generation dispatch
        self.gen_t_bus, self.gen_filtered_t_bus = self.compute_dispatch()
        
        # Plot settings
        self._plot_settings()
        
    def _set_directories(self, simulation_folder: str, network_file: str, weather_year: str):
        """Configure key directories using pathlib."""
        # Get the directory of the current script file
        self.file_dir = Path(__file__).resolve()
        # Navigate to parent directory (scripts folder) then to root project directory
        self.base_dir = self.file_dir.parent.parent.resolve()
        
        # Set path to simulations directory
        self.base_sim_dir = self.base_dir / 'simulations'
        
        # Set path to the specific simulation folder
        self.sim_dir = self.base_sim_dir / simulation_folder
        
        # Create results directory at simulation level
        self.base_res_dir = self.sim_dir / "results_concat"
        self.base_res_dir.mkdir(parents=True, exist_ok=True)
        
        # Search for the first available weather year directory

        sim_wy_dir = self.sim_dir / weather_year
            
        # If weather year directory exists, set it as active and break loop
        if sim_wy_dir.exists():
                self.sim_wy_dir = sim_wy_dir
                
                # Set path to networks subdirectory
                self.network_folder_dir = sim_wy_dir / "networks"
                
                # Set path to specific network file
                self.network_file_dir =  self.network_folder_dir / network_file
                
                # Set path to results directory for this weather year
                self.sim_res_dir = self.sim_wy_dir / "results"
                # Create results directory if it doesn't exist
                self.sim_res_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise Warning(f"Weather year directory '{weather_year}' not found in simulation '{simulation_folder}'.")
    
    def _get_network_components(self) -> None:
        
        # Load the PyPSA network
        self.n = pypsa.Network(self.network_file_dir)
        
        # Extract buses from network file
        self.buses = self.n.generators.bus.unique()
        
        # Clean carriers without valid color information
        carriers = self.n.carriers
        carriers = carriers[carriers.index != '']
        carriers = carriers[~carriers.color.isna()]
        carriers = carriers[carriers.color != ""]
        self.n.carriers = carriers
        
    def _save_plot(self, fig, filename: str, bus: str = "") -> None:
        """
        Save a matplotlib figure in the plot directory.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        filename : str
            File name without extension.
        bus : str, optional
            Subfolder per bus if specified.
        """
        folder = self.sim_res_dir / bus if bus else self.sim_res_dir
        folder.mkdir(parents=True, exist_ok=True)
        filepath = folder / f"{filename}.{self.file_format}"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved: {filepath}")

    def _plot_settings(self):
        """Configure colorblind-friendly plot settings and global plot style."""
        # Colorblind-friendly palette from Okabe-Ito
        self.nature_orange = '#e69f00'
        self.nature_sky_blue = '#56b4e9'
        self.nature_bluish_green = '#009e73'
        self.nature_yellow = '#f0e442'
        self.nature_blue = '#0072b2'
        self.nature_vermillion = '#d55e00'
        self.nature_reddish_purple = '#cc79a7'

        # Set Seaborn theme (whitegrid, version 8)
        plt.style.use("seaborn-v0_8-whitegrid")

        # Set global Matplotlib font to Arial using plt.rcParams
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['axes.titleweight'] = 'bold'
        #plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        
        self.x_length = 10
        self.y_length = self.x_length / 1.618
        
        self.title_fontsize = 14
        self.subtitle_fontsize = 12
        
        self.title_position = 1.06
        self.subtitle_position = 1.02

    def compute_emissions(self):
        """Compute CO2 emissions per generator and per bus."""
        n = self.n
        co2_gen = pd.DataFrame(index=n.generators_t.p.index, columns=n.generators.index)

        for g in n.generators.index:
            p_primary = n.generators_t.p[g] / n.generators.efficiency[g]
            co2_emi = n.carriers.co2_emissions.get(n.generators.carrier[g], 0)
            co2_gen[g] = p_primary * co2_emi

        gen_bus = n.generators.bus
        co2_bus = pd.DataFrame(index=co2_gen.index, columns=gen_bus.unique())

        for b in co2_bus.columns:
            gens_on_bus = gen_bus[gen_bus == b].index
            co2_bus[b] = co2_gen[gens_on_bus].sum(axis=1)

        return co2_gen, co2_bus

    def compute_dispatch(self):
        """Compute total and fossil-based dispatch by bus."""
        n = self.n
        gen_bus = n.generators.bus
        gen_t_bus = n.generators_t.p.T.groupby(gen_bus).sum().T

        fossil_carriers = ["CCGT", "oil", "lignite", "coal", "OCGT"]
        gens_filtered = n.generators.index[n.generators.carrier.isin(fossil_carriers)]
        gen_bus_filtered = n.generators.bus.reindex(gens_filtered)
        gen_filtered_t_bus = n.generators_t.p.T.groupby(gen_bus_filtered).sum().T

        return gen_t_bus, gen_filtered_t_bus
      
    def extract_summary(self):
        folder = self.sim_res_dir / "summary"
        folder.mkdir(parents=True, exist_ok=True)
        
        # Electricity prices
        self.electricity_prices_df = self.n.buses_t.marginal_price
        self.electricity_prices_df.to_csv(folder / "electricity_prices.csv")
        
        # Buses
        self.buses_df = self.n.buses
        self.buses_df.to_csv(folder / "buses.csv")
        
        # Electric load
        self.load_df = self.n.loads_t.p_set
        self.load_df.to_csv(folder / "electric_load.csv")
        
        # Generators
        self.generators_df = self.n.generators_t['p']  # Non-null keys --> "p_max_pu", "marginal_cost", "p"
        self.marginal_cost_df = self.n.generators_t['marginal_cost']
        self.generators_df.to_csv(folder / "generators_dispatch.csv")
        self.marginal_cost_df.to_csv(folder / "generators_marginal_cost.csv")
        
        

        # CO2 emissions
        self.co2_gen, self.co2_bus = self.compute_emissions()
        
        self.co2_gen.to_csv(folder / "co2_emissions_by_generator_t.csv")
        self.co2_bus.to_csv(folder / "co2_emissions_by_country.csv")
        
        
        ''' FEGU '''
        self.p_max_pu_df = self.n.generators_t['p_max_pu']
        self.p_max_pu_df.to_csv(folder / 'p_max_pu.csv')
        
        
        
        # FEGU Changes
        # Installed capacities per bus
        installed_capacities = []
        
        for bus in self.buses:
            gens_on_bus = self.n.generators[self.n.generators.bus == bus].index.copy()
        
            # Remove synthetic load if present
            load_row = f"{bus} load"
            if load_row in gens_on_bus:
                gens_on_bus = gens_on_bus.drop(load_row)
        
            # Retrieve optimized nominal capacities
            p_nom_on_bus = self.n.generators.p_nom_opt.reindex(gens_on_bus)
        
            for gen, cap in p_nom_on_bus.items():
                installed_capacities.append({
                    "bus": bus,
                    "generator": gen,
                    "p_nom_opt": cap
                })
        
        installed_capacities_df = pd.DataFrame(installed_capacities)
        installed_capacities_df.to_csv(folder / "installed_capacities.csv", index=False)
        
        
        
        


        
        # Energy mix
        energy_mix_dict ={}
        for bus in self.buses:
            
            gens_in = self.n.generators.index[self.n.generators.bus == bus]
            if len(gens_in) == 0:
                print(f"No generators on bus: {bus}")
                return

            # Annual energy per generator then grouped by carrier
            gen_sum = self.n.generators_t.p[gens_in].sum()
            gen_carriers = self.n.generators.carrier[gens_in]
            gen_sum_by_carrier = gen_sum.groupby(gen_carriers).sum()
        
            energy_mix_dict[bus] = gen_sum_by_carrier
        
        energy_mix_df = pd.DataFrame(energy_mix_dict).fillna(0).T
        energy_mix_df = energy_mix_df / 1e6 # Convert to TWh
        energy_mix_df.to_csv(folder / "energy_mix_by_bus_TWh.csv")
        
        

        
    def plot_installed_capacity(self, bus):
        """Plot installed capacities per bus as a horizontal bar chart."""

        load_row = f'{bus} load'
        gens_on_bus = self.n.generators[self.n.generators.bus == bus].index.copy()
        if load_row in gens_on_bus:
            gens_on_bus = gens_on_bus.drop(load_row)
        
        p_nom_on_bus = self.n.generators.p_nom_opt.reindex(gens_on_bus).sort_values()

        fig, ax = plt.subplots(figsize=(self.x_length, self.y_length))
        ax.barh(p_nom_on_bus.index, p_nom_on_bus.values, color=self.nature_sky_blue, edgecolor='black', linewidth=0.5)

        # Plot title and subtitle
        
        # Titles
        ax.text(0, self.subtitle_position,
            f"{bus} Installed Capacity - (MW-e)",
            fontsize=13,
            weight="bold",
            transform=ax.transAxes,
            ha="left",
        )
            
        #title = f"{bus} Installed Capacity - (MW-e)"
        #ax.set_title(title, loc="left", fontsize=13, pad=10)
        # Insert grid
        ax.grid(True, axis="x", linestyle="dashed", alpha=0.5)

        # remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        self._save_plot(fig, f"{bus}_installed_capacities", bus=bus)
        
    def plot_load(self, bus):
        """Plot load time series for a specific bus."""

        load = self.n.loads_t.p_set[bus]

        fig, ax = plt.subplots(figsize=(self.x_length, self.y_length))
        ax.plot(load.index, load.values, color=self.nature_sky_blue, linewidth=1)

        # Titles
        ax.text(0, self.title_position,
            f"{bus} Electric load",
            fontsize=self.title_fontsize,
            weight="bold",
            transform=ax.transAxes,
            ha="left",
        )
        
        ax.text(0, self.subtitle_position,
            "(MWh-e)",
            fontsize=self.subtitle_fontsize,
            #weight="bold",
            transform=ax.transAxes,
            ha="left",
        )

        ax.grid(True, linestyle="dashed", alpha=0.5)

        # remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # remove x-axis whitespace
        ax.set_xlim(load.index.min(), load.index.max())

        # adjust figure to prevent title/subtitle overlap
        fig.subplots_adjust(top=0.88)

        plt.tight_layout()
        self._save_plot(fig, f"{bus}_load", bus=bus)
        
    def plot_electricity_price(self, bus):
                    
        # Electricity price
        price = self.n.buses_t.marginal_price[bus]
        
        
        fig, ax = plt.subplots(figsize=(self.x_length, self.y_length))
        ax.plot(price.index, price.values, color=self.nature_sky_blue, linewidth=1)

        # Titles
        ax.text(0, self.title_position,
            f"{bus} Electricity prices",
            fontsize=self.title_fontsize,
            weight="bold",
            transform=ax.transAxes,
            ha="left",
        )
        
        ax.text(0, self.subtitle_position,
            "(€/MWh-e)",
            fontsize=self.subtitle_fontsize,
            #weight="bold",
            transform=ax.transAxes,
            ha="left",
        )

        #ax.set_ylabel("Price [€/MWh]")
        ax.grid(True, linestyle="dashed", alpha=0.5)

        # remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # remove x-axis whitespace
        ax.set_xlim(price.index.min(), price.index.max())

        plt.tight_layout()
        self._save_plot(fig, f"{bus}_el_prices", bus=bus)

    def plot_total_dispatch(self, bus):
            
        fig, ax = plt.subplots(figsize=(self.x_length, self.y_length))
        ax.plot(self.gen_t_bus.index, self.gen_t_bus[bus].values, color=self.nature_bluish_green)
        
        # Titles
        ax.text(0, self.title_position,
            f"{bus} Total Dispatch",
            fontsize=self.title_fontsize,
            weight="bold",
            transform=ax.transAxes,
            ha="left",
        )
        
        ax.text(0, self.subtitle_position,
            "Energy (MWh-el)",
            fontsize=self.subtitle_fontsize,
            #weight="bold",
            transform=ax.transAxes,
            ha="left",
        )
        
        #ax.set_title(f"{bus} Total Dispatch", loc='left')
        #ax.set_ylabel("Energy [MWh]")
        
        ax.grid(True, linestyle="dashed", alpha=0.5)
        
        # remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # remove x-axis whitespace
        ax.set_xlim(self.gen_t_bus.index.min(), self.gen_t_bus.index.max())
        
        plt.tight_layout()
        self._save_plot(fig, f"{bus}_total_dispatch", bus=bus)

    def plot_generation_dispatch_annual(self, bus):
        # Generation by carrier over time
        
        gens_in = self.n.generators.index[self.n.generators.bus == bus]
        gen_sum = self.n.generators_t.p[gens_in].sum()
        gen_carriers = self.n.generators.carrier[gens_in]
            
        p_gen = self.n.generators_t.p[gens_in]
        carriers = self.n.generators.carrier[gens_in]
        p_by_carrier = p_gen.groupby(carriers, axis=1).sum()
        
        fig, ax = plt.subplots(figsize=(12, 5))

        p_by_carrier.plot.area(ax=ax, linewidth=0)
        ax.set_title(f"{bus} Generation by Carrier", loc='left')
        ax.set_ylabel("Power [MW]")

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(True, linestyle="dashed", alpha=0.5)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Remove x-axis whitespace
        ax.set_xlim(p_by_carrier.index.min(), p_by_carrier.index.max())

        ax.legend(title="Carrier", bbox_to_anchor=(1.05, 1), loc="best",  frameon=True)
        plt.tight_layout()
        self._save_plot(fig, f"{bus}_generation_dispatch_annual", bus=bus)        
        
    def plot_generation_dispatch_fossil(self, bus):
        """Plot fossil-only dispatch time series for a bus (styled like other plots)."""
        # Retrieve series
        series = self.gen_filtered_t_bus[bus]

        fig, ax = plt.subplots(figsize=(self.x_length, self.y_length))
        ax.plot(series.index, series.values, color=self.nature_reddish_purple, linewidth=1)

        # Titles
        ax.text(0, self.title_position,
            f"{bus} Fossil Dispatch",
            fontsize=self.title_fontsize,
            weight="bold",
            transform=ax.transAxes,
            ha="left",
        )
        
        ax.text(0, self.subtitle_position,
            "Energy (MWh-el)",
            fontsize=self.subtitle_fontsize,
            #weight="bold",
            transform=ax.transAxes,
            ha="left",
        )
        
        #ax.set_title(f"{bus} Fossil Dispatch", loc="left", fontsize=14, pad=20)
        #ax.set_ylabel("Energy [MWh]")
        #ax.set_xlabel("Time")
        ax.grid(True, linestyle="dashed", alpha=0.5)

        # remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # x-axis range
        ax.set_xlim(series.index.min(), series.index.max())

        plt.tight_layout()
        self._save_plot(fig, f"{bus}_dispatch_fossil", bus=bus)

    def plot_co2_emissions(self, bus):
        """Plot CO₂ emissions time series for a bus (styled like other plots)."""
        # Retrieve series
        series = self.co2_bus[bus]

        fig, ax = plt.subplots(figsize=(self.x_length, self.y_length))
        ax.plot(series.index, series.values, color=self.nature_vermillion, linewidth=1)

        # Titles (use proper Unicode subscript for 2)
        ax.text(0, self.title_position,
            f"{bus} CO$_2$ Emissions",
            fontsize=self.title_fontsize,
            weight="bold",
            transform=ax.transAxes,
            ha="left",
        )
        
        ax.text(0, self.subtitle_position,
            "Emissions (tCO$_2$)",
            fontsize=self.subtitle_fontsize,
            transform=ax.transAxes,
            ha="left",
        )
        
        
        #ax.set_title(f"{bus} CO_{2} Emissions", loc="left", fontsize=14, pad=20)
        #ax.set_ylabel("Emissions [tCO₂]")
        #ax.set_xlabel("Time")
        ax.grid(True, linestyle="dashed", alpha=0.5)

        # remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # x-axis range
        ax.set_xlim(series.index.min(), series.index.max())

        plt.tight_layout()
        self._save_plot(fig, f"{bus}_co2_emissions", bus=bus)

    def plot_generation_mix_annual(self, bus):
        """Plot annual generation mix as a pie chart, using carrier colors where available."""
        gens_in = self.n.generators.index[self.n.generators.bus == bus]
        if len(gens_in) == 0:
            self.logger.info(f"No generators on bus: {bus}")
            return

        # Annual energy per generator then grouped by carrier
        gen_sum = self.n.generators_t.p[gens_in].sum()
        gen_carriers = self.n.generators.carrier[gens_in]
        gen_sum_by_carrier = gen_sum.groupby(gen_carriers).sum()

        if gen_sum_by_carrier.sum() == 0:
            self.logger.info(f"No generation energy for bus: {bus}")
            return

        # Try to get carrier colors from network carriers table, fallback to default cmap
        carrier_colors = {}
        try:
            carrier_colors = self.n.carriers.color.to_dict()
        except Exception:
            carrier_colors = {}

        colors = [carrier_colors.get(c, None) for c in gen_sum_by_carrier.index]
        # matplotlib will handle None entries by assigning default colors

        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax.pie(
            gen_sum_by_carrier.values,
            labels=gen_sum_by_carrier.index,
            colors=colors,
            autopct="%.1f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="w"),
            textprops=dict(color="k")
        )
        ax.set_title(f"{bus} Annual Generation Mix", loc="left", fontsize=14, pad=20)
        ax.axis("equal")  # keep pie circular

        # legend on the right if many carriers
        if len(gen_sum_by_carrier) > 6:
            ax.legend(wedges, 
                      gen_sum_by_carrier.index,
                      title="Carrier",
                      bbox_to_anchor=(1.05, 1),
                      loc="upper left",
                      frameon=True,
                      )

        plt.tight_layout()
        self._save_plot(fig, f"{bus}_generation_mix_annual", bus=bus)
    
    # TODO     
    def plot_generation_comparison_all(self, bus):
    
        pypsa_gen = self.n.generators_t['p']
    
        # Remove timezone info from PyPSA generation index if present
        if pypsa_gen.index.tz is not None:
            pypsa_gen.index = pypsa_gen.index.tz_convert(None)
    
        # Filter PyPSA columns for the bus (country)
        columns_bus = [col for col in pypsa_gen.columns if col.startswith(f"{bus} ") and "load" not in col.lower()]
        if not columns_bus:
            print(f"No PyPSA generation columns found for bus '{bus}'")
            return
    
        # Map generation technologies to standard names
        mapping = {
            'biomass': 'Biomass',
            'CCGT': 'Fossil Gas',
            'OCGT': 'Fossil Gas',
            'coal': 'Fossil Hard coal',
            'lignite': 'Fossil Hard coal',
            'oil': 'Fossil Oil',
            'geothermal': 'Geothermal',
            'ror': 'Hydro Run-of-river and poundage',
            '0 offwind-ac': 'Wind Offshore',
            '0 offwind-dc': 'Wind Offshore',
            '0 onwind': 'Wind Onshore',
            '0 solar': 'Solar',
            '0 solar-hsat': 'Solar',
            'load': None  # ignore load columns
        }
        mapping_lower = {k.lower(): v for k, v in mapping.items()}
    
        # Select and map tech
        filtered_df = pypsa_gen[columns_bus].copy()
        raw_tech = [col.split(" ", 1)[1].strip().lower() for col in filtered_df.columns]
        mapped_tech = [mapping_lower.get(tech) for tech in raw_tech]
    
        # Keep only columns with a valid mapping
        valid_cols = [col for col, tech in zip(filtered_df.columns, mapped_tech) if tech is not None]
        valid_techs = [tech for tech in mapped_tech if tech is not None]
    
        filtered_df = filtered_df[valid_cols]
        filtered_df.columns = valid_techs
    
        # Group by technology
        pypsa_grouped = filtered_df.groupby(filtered_df.columns, axis=1).sum()
    
        # Load external generation data for this bus
        external_file = self.base_dir / "data" / "generation" / "generation_hourly_data" / f"generation_{bus}_hourly_data.csv"
        if not external_file.exists():
            print(f"External generation file not found for bus '{bus}': {external_file}")
            return
    
        generation_external = pd.read_csv(external_file, quotechar='"', sep=",")
        generation_external.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
        generation_external["Date"] = pd.to_datetime(generation_external["Date"])
        generation_external.set_index("Date", inplace=True)
    
        # Remove timezone info from external data index if present
        if generation_external.index.tz is not None:
            generation_external.index = generation_external.index.tz_convert(None)
    
        # Extract all years from PyPSA data index
        if not hasattr(pypsa_grouped.index, 'year'):
            print("PyPSA generation index does not support .year attribute")
            return
        years = sorted(set(pypsa_grouped.index.year))
    
        for year in years:
            start_date = pd.Timestamp(year, 1, 1)
            end_date = start_date + pd.DateOffset(years=1) - pd.Timedelta(hours=1)
    
            # Filter PyPSA and external data by year
            pypsa_year = pypsa_grouped.loc[(pypsa_grouped.index.year == year)]
            external_year = generation_external.loc[start_date:end_date]
    
            # Find common technologies available both in PyPSA and external data
            common_cols = pypsa_year.columns.intersection(external_year.columns)
            if len(common_cols) == 0:
                print(f"No common generation technologies to compare for bus '{bus}' in year {year}")
                continue
    
            # Prepare data for plotting
            pypsa_plot = pypsa_year[common_cols].copy()
            external_plot = external_year[common_cols].copy()
    
            pypsa_plot.index = pd.to_datetime(pypsa_plot.index).to_pydatetime()
            external_plot.index = pd.to_datetime(external_plot.index).to_pydatetime()
    
            # Plot each technology comparison
            fig, axes = plt.subplots(len(common_cols), 1, figsize=(12, 4 * len(common_cols)), sharex=True)
            if len(common_cols) == 1:
                axes = [axes]
    
            for ax, tech in zip(axes, common_cols):
                ax.plot(external_plot.index, external_plot[tech], label='External Generation Data')
                ax.plot(pypsa_plot.index, pypsa_plot[tech], label='PyPSA Generation Dispatch', linestyle='--')
                ax.set_title(f"{tech} — {bus} ({year})")
                ax.set_ylabel('Power [MW]')
                ax.legend()
                ax.grid(True, linestyle='dashed', alpha=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
    
            plt.xlabel('Time')
            plt.tight_layout()
    
            self._save_plot(fig, f"{bus}_generation_comparison", bus=bus)
    
            
    def plot_all_figures(self):
        """Generate and save standard analysis plots per bus."""

        for bus in self.buses:
            bus_folder = self.sim_res_dir / bus
            bus_folder.mkdir(parents=True, exist_ok=True)

            # Plot installed capacity
            self.plot_installed_capacity(bus)         # Plot installed capacity
            self.plot_load(bus)                       # Plot electric load
            self.plot_generation_dispatch_annual(bus) # Plot generation dispatch annual
            self.plot_electricity_price(bus)          # Plot electricity price
            self.plot_total_dispatch(bus)             # Plot total dispatch
            self.plot_generation_dispatch_fossil(bus) # Plot fossil dispatch
            self.plot_co2_emissions(bus)              # Plot CO2 emissions
            self.plot_generation_mix_annual(bus)      # Plot annual generation mix
            self.plot_generation_comparison_all(bus)  # Plot generation comparison with external data
            
            
    