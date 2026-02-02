from pathlib import Path
import pypsa
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from omegaconf import DictConfig

import logging


class NetworkAnalyzer:
    """
    A class to analyze main figures from a PyPSA network file
    """

    def __init__(self, config: DictConfig, network_file: str, logger: Optional[logging.Logger]):
        self.logger = logger
        self.config = config
        self.network_file = network_file

        self._plot_settings()
        self._set_directories(network_file)
        self._get_network_components()
        self.gen_t_bus, self.gen_filtered_t_bus = self.compute_dispatch()

    def _set_directories(self, network_file: str):
        """Configure key directories using pathlib."""

        # Set root and file directories
        self.file_dir = Path(__file__).resolve().parent
        self.root_dir = Path(self.config.paths.root)
        self.network_file_path = Path(network_file)

        # Set input data directory
        self.data_dir = self.root_dir / "data"
        network_files_data_dir = self.data_dir / "network_files"
        self.network_file_dir = network_files_data_dir / self.network_file_path

        # Check if network file exists
        if not self.network_file_dir.exists():
            msg = (
                f"Network file {self.network_file_dir} does not exist in data folder"
            )
            self.logger.error(f"Network file {self.network_file_dir} does not exist in data folder"
            )
            raise FileNotFoundError(msg)

        if self.network_file_dir.suffix != ".nc":
            msg = (
                f"Invalid network file extension: "
                f"{self.network_file_dir.suffix}. Expected .nc"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Directories creation
        self.results_dir = self.root_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        network_stem = self.network_file_path.stem
        self.network_file_res_dir = self.results_dir / network_stem
        self.network_file_res_dir.mkdir(parents=True, exist_ok=True)

        self.res_concat_dir = self.root_dir / "results_concat"
        self.res_concat_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Directories configured for network: {network_stem}")


    def _get_network_components(self) -> None:
        # Load the PyPSA network
        self.n = pypsa.Network(self.network_file_dir)
        self.logger.info("Correctly read network file")
        
        # Extract buses from network file
        self.buses = self.n.generators.bus.unique()

        # Extract hydro and pumped hydro storage units
        self.hydro_units = self.n.storage_units[self.n.storage_units["carrier"] == "hydro"]
        self.phs_units = self.n.storage_units[self.n.storage_units["carrier"] == "PHS"]

        # Clean carriers without valid color information
        carriers = self.n.carriers
        carriers = carriers[carriers.index != ""]
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
        folder = self.network_file_res_dir / bus if bus else self.network_file_res_dir
        folder.mkdir(parents=True, exist_ok=True)
        
        filepath = folder / f"{filename}.{self.config.plot_export_format}"
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Plot saved: {filepath}")

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
        plt.rcParams['font.family'] = 'DejaVu Sans'
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
        """Compute total and conventional+hydro dispatch by bus."""

        n = self.n

        # Total generator dispatch by bus
        gen_bus = n.generators.bus
        gen_t_bus = n.generators_t.p.T.groupby(gen_bus).sum().T

        # Conventional generators
        conventional_carriers = [
            "nuclear", "oil", "OCGT", "CCGT",
            "coal", "lignite", "geothermal", "biomass"
        ]
        gens_filtered = n.generators.index[n.generators.carrier.isin(conventional_carriers)]
        gen_bus_filtered = n.generators.bus.reindex(gens_filtered)
        gen_filtered_t_bus = n.generators_t.p.T.groupby(gen_bus_filtered).sum().T

        # Add hydro + pumped hydro from storage units
        hydro_gen_bus = self.hydro_units.bus
        phs_gen_bus = self.phs_units.bus

        # Get discharge power time series (p_dispatch) and sum by bus
        hydro_dispatch_t_bus = n.storage_units_t.p_dispatch.T.groupby(hydro_gen_bus).sum().T
        phs_net_t_bus = n.storage_units_t.p_dispatch.T.groupby(phs_gen_bus).sum().T

        # Add hydro + PHS discharge to conventional generators dispatch and gen_t_bus
        gen_filtered_t_bus = gen_filtered_t_bus.add(hydro_dispatch_t_bus, fill_value=0)
        gen_filtered_t_bus = gen_filtered_t_bus.add(phs_net_t_bus, fill_value=0)
        gen_t_bus = gen_t_bus.add(hydro_dispatch_t_bus, fill_value=0)
        gen_t_bus = gen_t_bus.add(phs_net_t_bus, fill_value=0)

        return gen_t_bus, gen_filtered_t_bus
      
    def extract_summary(self):
        folder = self.network_file_res_dir / "summary"
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
        self.generators_df = self.n.generators_t["p"]  # Non-null keys --> "p_max_pu", "marginal_cost", "p"
        self.marginal_cost_df = self.n.generators_t["marginal_cost"]
        self.generators_df.to_csv(folder / "generators_dispatch.csv")
        self.marginal_cost_df.to_csv(folder / "generators_marginal_cost.csv")
        self.hydro_all_df = self.n.storage_units_t["p_dispatch"]
        
        # Concat generators_df with hydro and phs dispatch
        self.total_dispatch = self.generators_df.add(self.hydro_all_df, fill_value=0)
        self.total_dispatch.to_csv(folder / "total_generators_dispatch_with_hydro_phs.csv")

        carrier_gen = self.n.generators.carrier
        carrier_storage = self.n.storage_units.carrier

        self.carrier_total = pd.concat([carrier_gen, carrier_storage])
        self.total_dispatch_by_carrier = self.total_dispatch.T.groupby(self.carrier_total).sum().T
        self.total_dispatch_by_carrier.to_csv(folder / "total_dispatch_by_carrier.csv")

        # Get total capacities
        self.capacities_gen = self.n.generators.p_nom_opt
        self.capacities_storage = self.n.storage_units.p_nom_opt
        self.capacities = pd.concat([self.capacities_gen, self.capacities_storage])
        # Retrieve the bus for each generator and storage unit
        # Directly create aligned dataframe
        capacities_bus = pd.DataFrame(
            {
                "bus": pd.concat([self.n.generators.bus, self.n.storage_units.bus]),
                "p_nom_opt": pd.concat([self.capacities_gen, self.capacities_storage]),
            }
        )
        # Add name generator/storage unit as index name
        capacities_bus.index.name = "generator"
        self.capacities_bus = capacities_bus
        self.capacities_bus.to_csv(folder / "installed_capacities_MW.csv")
        # Total capacities by carrier
        self.capacities_by_carrier = self.capacities.groupby(self.carrier_total).sum()
        self.capacities_by_carrier.to_csv(folder / "installed_capacities_by_carrier_MW.csv")

        # CO2 emissions
        self.co2_gen, self.co2_bus = self.compute_emissions()

        self.co2_gen.to_csv(folder / "co2_emissions_by_generator_t.csv")
        self.co2_bus.to_csv(folder / "co2_emissions_by_country.csv")
        self.p_max_pu_df = self.n.generators_t["p_max_pu"]
        self.p_max_pu_df.to_csv(folder / "p_max_pu.csv")

        # Energy mix
        energy_mix_dict = {}
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
        energy_mix_df = energy_mix_df / 1e6  # Convert to TWh
        energy_mix_df.to_csv(folder / "energy_mix_by_bus_TWh.csv")

    def plot_installed_capacity(self, bus):
        """
        Plot installed capacities per bus as a horizontal bar chart.
        Adds reservoir hydro and pumped hydro as separate pseudo-generators.
        """
        load_row = f'{bus} load'
        gens_on_bus = self.n.generators[self.n.generators.bus == bus].index.copy()
        if load_row in gens_on_bus:
            gens_on_bus = gens_on_bus.drop(load_row)
        # Add hydro and PHS as single bar (sum their p_nom per bus)
        hydro_units = self.hydro_units[self.hydro_units.bus == bus]
        phs_units = self.phs_units[self.phs_units.bus == bus]
        extras = {}
        if not hydro_units.empty:
            extras[f'{bus} Hydro_Reservoir'] = hydro_units["p_nom_opt"].sum()
        if not phs_units.empty:
            extras[f'{bus} Pumped_Hydro'] = phs_units["p_nom_opt"].sum()

        p_nom_on_bus = self.n.generators.p_nom_opt.reindex(gens_on_bus)
        extras_s = pd.Series(extras, name=p_nom_on_bus.name)
        extras_s.index.name = p_nom_on_bus.index.name   
        full_p_nom = pd.concat([p_nom_on_bus, extras_s])
        full_p_nom = full_p_nom.sort_values()
        

        fig, ax = plt.subplots(figsize=(self.x_length, self.y_length))
        ax.barh(full_p_nom.index, full_p_nom.values, color=self.nature_sky_blue, edgecolor='black', linewidth=0.5)
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
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        self._save_plot(fig, f"{bus}_installed_capacities", bus=bus)
        
    def plot_load(self, bus):
        """Plot load time series for a specific bus."""

        load = self.n.loads_t.p_set[bus]

        self.total_load_bus_TWh = round(load.sum() / 1e6, 1)
        fig, ax = plt.subplots(figsize=(self.x_length, self.y_length))
        ax.plot(load.index, load.values, color=self.nature_sky_blue, linewidth=1)

        # Titles
        ax.text(0, self.title_position,
            f"{bus} Electric load - Total load: {self.total_load_bus_TWh} TWh",
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
        
        self.total_bus_gen_dispatch_TWh = round(self.gen_t_bus[bus].sum() / 1e6, 1)
        # Titles
        ax.text(0, self.title_position,
            f"{bus} Dispatch - Total generation: {self.total_bus_gen_dispatch_TWh} TWh",
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
        # Add hydro and PHS storage units as generators
        hydro_units = self.hydro_units.index[self.hydro_units.bus == bus]
        phs_units = self.phs_units.index[self.phs_units.bus == bus]


        p_gen = self.n.generators_t.p[gens_in]
        hydro_gen = self.n.storage_units_t.p_dispatch[hydro_units]
        phs_gen = self.n.storage_units_t.p_dispatch[phs_units]
        carriers = self.n.generators.carrier[gens_in]
        carrier_hydro = self.n.storage_units.carrier[hydro_units]
        carrier_phs = self.n.storage_units.carrier[phs_units]
        all_generation = pd.concat([p_gen, hydro_gen, phs_gen], axis=1)
        all_carriers = pd.concat([carriers, carrier_hydro, carrier_phs])
        p_by_carrier = all_generation.T.groupby(all_carriers).sum().T
        
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
        
    def plot_generation_dispatch_conventional(self, bus):
        """Plot fossil-only dispatch time series for a bus (styled like other plots)."""
        # Retrieve series
        series = self.gen_filtered_t_bus[bus]

        fig, ax = plt.subplots(figsize=(self.x_length, self.y_length))
        ax.plot(series.index, series.values, color=self.nature_reddish_purple, linewidth=1)

        # Titles
        ax.text(0, self.title_position,
            f"{bus} Conventional Dispatch",
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
        self._save_plot(fig, f"{bus}_dispatch_conventional", bus=bus)

    def plot_co2_emissions_per_bus(self, bus):
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

    #TODO 
    def plot_CO2_intensity_comparison_all(self, bus):
        weather_year = self.weather_year
        resample_rule = "3h"   

        # Retrieve the CO2 emissions time series from PyPSA network (tCO2/MWh)
        pypsa_co2 = self.co2_bus[bus]

        # Load in MWh
        load = self.load_df[bus]

        # Compute CO₂ intensity (tCO₂/MWh ÷ → gCO₂/kWh)
        df_co2_intensity = (pypsa_co2 / load) * 1000

        # --- Set up plot ---
        fig, ax = plt.subplots(figsize=(self.x_length, self.y_length))

        # Plot PyPSA CO₂ intensity
        pypsa_plot = df_co2_intensity.resample(resample_rule).mean()
        ax.plot(
            pypsa_plot.index,
            pypsa_plot.values,
            label='PyPSA CO$_2$ Intensity',
            color=getattr(self, "nature_vermillion", None),
            linewidth=1
        )

        # --- decide whether to load & plot reference data ---
        if weather_year != "2020":
            ref_file_path = (
                self.base_dir / 'data' / 'co2_emissions' /
                'aggregated_data' / f'co2_intensity_{weather_year}.csv'
            )

            if ref_file_path.exists():
                try:
                    df_ref = pd.read_csv(ref_file_path, parse_dates=['DateTime (UTC)'])
                    df_ref = df_ref.set_index('DateTime (UTC)')

                    if bus in df_ref.columns:
                        ref_series = df_ref[bus].astype(float)
                        ref_plot = ref_series.resample(resample_rule).mean()

                        ax.plot(
                            ref_plot.index,
                            ref_plot.values,
                            label='Reference CO$_2$ Intensity',
                            color=getattr(self, "nature_blue", None),
                            linewidth=1
                        )
                    else:
                        print(
                            f"No reference CO2 data found for bus '{bus}' "
                            f"in {ref_file_path.name} — plotting only PyPSA."
                        )

                except Exception as e:
                    print(f"Failed to read reference file {ref_file_path}: {e} — plotting only PyPSA.")
            else:
                print(f"Reference file not found for weather_year={weather_year}. Plotting only PyPSA.")
        else:
            print("weather_year == '2020' → no reference data available; plotting only PyPSA.")

        # --- Labels & styling ---
        ax.set_xlabel('Time')
        ax.set_ylabel(r'CO$_2$ Emissions in g/kWh')
        ax.set_title(
            rf'CO$_2$ Emissions Comparison: {bus} {weather_year}',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_plot(fig, f"{bus}_co2_emissions_comparison", bus=bus)

    def plot_generation_mix_annual(self, bus):
        """Plot annual generation mix as a pie chart, using carrier colors where available."""
        gens_in = self.n.generators.index[self.n.generators.bus == bus]
        # Add hydro and PHS storage units as generators
        hydro_units = self.hydro_units[self.hydro_units.bus == bus]
        phs_units = self.phs_units[self.phs_units.bus == bus]
        if len(gens_in) == 0:
            self.logger.info(f"No generators on bus: {bus}")
            return

        # Annual energy per generator then grouped by carrier
        gen_sum = self.n.generators_t.p[gens_in].sum()
        hydro_sum = self.n.storage_units_t.p_dispatch[hydro_units.index].sum()
        phs_sum = self.n.storage_units_t.p_dispatch[phs_units.index].sum()
        gen_carriers = self.n.generators.carrier[gens_in]
        hydro_carriers =self.n.storage_units.carrier[hydro_units.index]
        phs_carriers = self.n.storage_units.carrier[phs_units.index]
        gen_sum_by_carrier = gen_sum.groupby(gen_carriers).sum()
        gen_hydro_by_carrier = hydro_sum.groupby(hydro_carriers).sum()
        gen_phs_by_carrier = phs_sum.groupby(phs_carriers).sum()

        # Combine all generation by carrier
        gen_sum_by_carrier = gen_sum_by_carrier.add(gen_hydro_by_carrier, fill_value=0)
        gen_sum_by_carrier = gen_sum_by_carrier.add(gen_phs_by_carrier, fill_value=0)

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
        
        
    def plot_total_dispatch_all_buses(self):
        self.total_dispatch_by_carrier_sum = self.total_dispatch_by_carrier.sum() / 1e6
        series = self.total_dispatch_by_carrier_sum.sort_values(ascending=True)
        folder = self.sim_res_dir / "summary"

        # ===============================================================
        # 1) HORIZONTAL BAR CHART
        # ===============================================================
        fig, ax = plt.subplots(figsize=(self.x_length, self.y_length))
        ax.barh(
            series.index,
            series.values,
            color=self.nature_sky_blue,
            edgecolor="black",
            linewidth=0.5
        )

        ax.text(
            0,
            self.subtitle_position,
            f"All buses Dispatch by Carrier (MWh-e) - Total: {round(series.sum(), 1)} TWh",
            fontsize=13,
            weight="bold",
            transform=ax.transAxes,
            ha="left",
        )

        ax.grid(True, axis="x", linestyle="dashed", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(folder / "total_dispatch_all_buses.pdf")
        plt.close(fig)

        # ===============================================================
        # 2) PIE CHART
        # ===============================================================
        # Carrier colors (fallback to default)
        try:
            carrier_colors = self.n.carriers.color.to_dict()
        except Exception:
            carrier_colors = {}
        colors = [carrier_colors.get(c, None) for c in series.index]

        fig, ax = plt.subplots(figsize=(self.x_length, self.y_length))
        wedges, _, _ = ax.pie(
            series.values,
            labels=series.index,
            colors=colors,
            autopct="%.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "w"},
            textprops={"color": "k"},
        )

        ax.set_title(
            f"All Buses Dispatch Share by Carrier – Total: {round(series.sum(), 1)} TWh",
            loc="left",
            fontsize=14,
            pad=15
        )
        ax.axis("equal")

        if len(series) > 6:
            ax.legend(
                wedges,
                series.index,
                title="Carrier",
                bbox_to_anchor=(1.05, 1),
                loc="upper left"
            )

        plt.tight_layout()
        fig.savefig(folder / "total_dispatch_all_buses_pie.pdf")
        plt.close(fig)

        
        
        
        
        
    def plot_generation_comparison_all(self, bus):
        pypsa_gen = self.n.generators_t.p
        # Add storage unit discharges (hydro + PHS) to generation dispatch
        hydro_dispatch = self.n.storage_units_t.p_dispatch[self.hydro_units.index]
        phs_net = self.n.storage_units_t.p[self.phs_units.index]
        pypsa_gen = pypsa_gen.join(hydro_dispatch).join(phs_net)
        pypsa_gen.fillna(0, inplace=True)
    
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
            'hydro': 'Hydro Water Reservoir',
            'PHS': 'Hydro Pumped Storage Net',
            'nuclear': 'Nuclear',
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
        pypsa_grouped = filtered_df.T.groupby(filtered_df.columns).sum().T
    
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

    def plot_hydro_analysis(self, bus):
        """
        Plots hydro inflow, state of charge, and power dispatch.
        Each output is two subplots: left is PHS, right is reservoir hydro.
        """
        
        # Hydro inflow
        hydro_inflow_res = self.n.storage_units_t.inflow.loc[:, self.n.storage_units_t.inflow.columns.str.contains(bus, na=False) & self.n.storage_units.carrier[self.n.storage_units_t.inflow.columns].eq('hydro').values]
        hydro_inflow_phs = self.n.storage_units_t.inflow.loc[:, self.n.storage_units_t.inflow.columns.str.contains(bus, na=False) & self.n.storage_units.carrier[self.n.storage_units_t.inflow.columns].eq('PHS').values]

        # 1. Hydro inflow plot
        fig, axs = plt.subplots(1,2, figsize=(14,5), constrained_layout=True)
        hydro_inflow_phs.sum(axis=1).plot(ax=axs[0], title="Pumped Hydro Inflow", color=self.nature_sky_blue)
        hydro_inflow_res.sum(axis=1).plot(ax=axs[1], title="Hydro Reservoir Inflow", color=self.nature_bluish_green)
        self._save_plot(fig, f"{bus}_hydro_inflow", bus=bus)
        
        # 2. State of charge (SOC)
        soc_phs = self.n.storage_units_t.state_of_charge.loc[:, self.n.storage_units_t.state_of_charge.columns.str.contains(bus, na=False) & self.n.storage_units.carrier[self.n.storage_units_t.state_of_charge.columns].eq('PHS').values]
        soc_hydro = self.n.storage_units_t.state_of_charge.loc[:, self.n.storage_units_t.state_of_charge.columns.str.contains(bus, na=False) & self.n.storage_units.carrier[self.n.storage_units_t.state_of_charge.columns].eq('hydro').values]
        fig, axs = plt.subplots(1,2, figsize=(14,5), constrained_layout=True)
        soc_phs.sum(axis=1).plot(ax=axs[0], title="PHS State of Charge (TWh)", color=self.nature_sky_blue)
        soc_hydro.sum(axis=1).plot(ax=axs[1], title="Hydro Reservoir State of Charge (TWh)", color=self.nature_bluish_green)
        self._save_plot(fig, f"{bus}_hydro_soc", bus=bus)
        
        # 3. Power "p"
        phs_p = self.n.storage_units_t.p.loc[:, self.n.storage_units_t.p.columns.str.contains(bus, na=False) & self.n.storage_units.carrier[self.n.storage_units_t.p.columns].eq('PHS').values]
        hydro_p = self.n.storage_units_t.p.loc[:, self.n.storage_units_t.p.columns.str.contains(bus, na=False) & self.n.storage_units.carrier[self.n.storage_units_t.p.columns].eq('hydro').values]
        # Resample to daily 
        phs_p = phs_p.resample('D').mean()
        hydro_p = hydro_p.resample('D').mean()
        fig, axs = plt.subplots(1,2, figsize=(14,5), constrained_layout=True)
        phs_p.sum(axis=1).plot(ax=axs[0], title="PHS Daily avg Power", color=self.nature_sky_blue)
        hydro_p.sum(axis=1).plot(ax=axs[1], title="Hydro Reservoir Daily avg Power", color=self.nature_bluish_green)
        self._save_plot(fig, f"{bus}_hydro_power", bus=bus)
          
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
            self.plot_generation_dispatch_conventional(bus) # Plot fossil dispatch
            self.plot_co2_emissions_per_bus(bus)              # Plot CO2 emissions
            self.plot_generation_mix_annual(bus)      # Plot annual generation mix
            self.plot_generation_comparison_all(bus)  # Plot generation comparison with external data
            self.plot_hydro_analysis(bus)             # Plot hydro analysis
            self.plot_CO2_intensity_comparison_all(bus) # Plot CO2 emissions comparison with external data
            self.plot_total_dispatch_all_buses()
            # Break for now
            # break
            