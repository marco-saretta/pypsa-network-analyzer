#%%
import pandas as pd
import pypsa
from pathlib import Path
#%%
# Define the base directory
base_dir = Path(__file__).resolve().parent.parent

# Read in the network file 
network_file = base_dir / "simulations" / "hindcast_dyn_old" / "weather_year_2022" / "networks" / "base_s_39_elec_Ept.nc"
n = pypsa.Network(network_file)
#%%
# Get overall statistics static
phs_statistics = n.storage_units[n.storage_units['carrier'] == 'PHS']
hydro_statistics = n.storage_units[n.storage_units['carrier'] == 'hydro']
#%%

# Get pumpped hydro storage units per bus
phs_gen = n.storage_units['carrier'] == 'PHS'
hydro_gen = n.storage_units['carrier'] == 'hydro'
# Extract relevant time series data for pumped hydro storage units
hydro_inflow = n.storage_units_t["inflow"]
hp_p = n.storage_units_t["p"] # Both hydro and pumped hydro
hp_p_dispatch = n.storage_units_t["p_dispatch"]  # Both hydro and pumped hydro
phs_p_store = n.storage_units_t["p_store"] # 0 for hydro, values for pumped hydro
hp_p_soc = n.storage_units_t["state_of_charge"] # For both hydro and pumped hydro

# %%
# Filter the data frames wih both hydro and pumped hydro to seperate (columns name is either ... PHS or ... hydro)
phs_p = hp_p.loc[:, phs_gen]
hydro_p = hp_p.loc[:, hydro_gen]
phs_p_dispatch = hp_p_dispatch.loc[:, phs_gen]
hydro_p_dispatch = hp_p_dispatch.loc[:, hydro_gen]
phs_p_store = phs_p_store.loc[:, phs_gen]
phs_p_soc = hp_p_soc.loc[:, phs_gen]
hydro_p_soc = hp_p_soc.loc[:, hydro_gen]

# Convert SOC from MWh into TWh for better readability
phs_p_soc /= 1e6
hydro_p_soc /= 1e6
# %%
# Choose Norway as bus "NO"
# Plot all time series for Norway filter by bus (make sure column inclufes NO)
bus_name = "NO"
phs_p_no = phs_p.loc[:, phs_p.columns.str.contains(bus_name)]
hydro_p_no = hydro_p.loc[:, hydro_p.columns.str.contains(bus_name)]
phs_p_dispatch_no = phs_p_dispatch.loc[:, phs_p_dispatch.columns.str.contains(bus_name)]
hydro_p_dispatch_no = hydro_p_dispatch.loc[:, hydro_p_dispatch.columns.str.contains(bus_name)]
phs_p_store_no = phs_p_store.loc[:, phs_p_store.columns.str.contains(bus_name)]
phs_p_soc_no = phs_p_soc.loc[:, phs_p_soc.columns.str.contains(bus_name)]
hydro_p_soc_no = hydro_p_soc.loc[:, hydro_p_soc.columns.str.contains(bus_name)]

# %%
# Plot the time series for all the datasets
import matplotlib.pyplot as plt
fig, axs = plt.subplots(4, 2, figsize=(15, 10), constrained_layout=True)
phs_p_no.plot(ax=axs[0, 0], title="Pumped Hydro Power (PHS) at Bus NO", ylabel="Power (MW)")
hydro_p_no.plot(ax=axs[0, 1], title="Hydro Power at Bus NO", ylabel="Power (MW)")
phs_p_dispatch_no.plot(ax=axs[1, 0], title="Pumped Hydro Power Dispatch (PHS) at Bus NO", ylabel="Power (MW)")
hydro_p_dispatch_no.plot(ax=axs[1, 1], title="Hydro Power Dispatch at Bus NO", ylabel="Power (MW)")
phs_p_store_no.plot(ax=axs[2, 0], title="Pumped Hydro Power Store (PHS) at Bus NO", ylabel="Power (MW)")
phs_p_soc_no.plot(ax=axs[3, 0], title="Pumped Hydro State of Charge (PHS) at Bus NO", ylabel="State of Charge (TWh)")
hydro_p_soc_no.plot(ax=axs[3, 1], title="Hydro State of Charge at Bus NO", ylabel="State of Charge (TWh)")
plt.show()
# %%
# Plot capacities of hydro and pumped hydro storage units
fig, ax = plt.subplots(figsize=(10, 6))
phs_statistics['p_nom'].plot(kind='bar', ax=ax, color='blue', label='Pumped Hydro Capacity (PHS)')
hydro_statistics['p_nom'].plot(kind='bar', ax=ax, color='green', label='Hydro Capacity', alpha=0.7)
ax.set_title('Capacities of Hydro and Pumped Hydro Storage Units')
ax.set_ylabel('Capacity (MW)')
ax.legend()
plt.show()



# %%
