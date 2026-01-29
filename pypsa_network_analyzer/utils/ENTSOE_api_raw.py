"""
Code to query data from ENTSOe RES API
Code sources:
    https://thesmartinsights.com/how-to-query-data-from-the-entso-e-transparency-platform-using-python/
    https://github.com/EnergieID/entsoe-py
"""

# =============================================================================
# import requests
# import urllib
# import datetime
# import json
# import pandas as pd
# 
# response = requests.get('https://apidatos.ree.es/en/datos/generacion/estructura-generacion?start_date=2020-01-01T00:00&end_date=2020-12-31T00:00&time_trunc=day')
# 
# data = response.json()
# 
# with open('data.json', 'w') as f:
#     json.dump(data, f)
# 
# for d in data['included']:
#     print(d['type'])
# 
# df_values = pd.DataFrame(columns=gentypes, index = days)
# df_percentages = pd.DataFrame(columns=gentypes, index = days)
# 
# gentypes = list(range(len(data['included'])))
# days = list(range(len(data['included'][0]['attributes']['values'])))
# dates = []
# values = []
# percentages = []
# 
# for j in days:
#     dates.append(data['included'][0]['attributes']['values'][j]['datetime'])
# 
# for i in gentypes:
#     for j in days:
#         df_values.loc[j,i] = data['included'][i]['attributes']['values'][j]['value']
#         df_percentages.loc[j,i] = data['included'][i]['attributes']['values'][j]['percentage']
# =============================================================================


import pandas as pd
from entsoe import EntsoePandasClient
from tqdm import tqdm

client = EntsoePandasClient(api_key='1ec78127-e12b-4cb2-a9fb-1258e4d5622a')

start = pd.Timestamp('20250501', tz ='UTC')

end = pd.Timestamp('20300101', tz ='UTC')


countries_dict = {
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
    'XK': 'Kosovo'
}


demand = {}

for c in tqdm(countries_dict.keys()):
    try:
        demand[c] = client.query_load(c, start=start,end=end)
        demand[c] = demand[c].resample('h').mean()
        print(f'{c} correctly fecthed')
    except:
        print(f'{c} gave problems')
        pass

demand_df = pd.DataFrame(columns=list(countries_dict))

for c in countries_dict:
    try:
        demand_df[c] = demand[c][('Actual Load')]    
    except:
        pass
            
demand_df['year'] = demand_df.index.year
demand_df['month'] = demand_df.index.month
demand_df['week'] = demand_df.index.isocalendar().week
demand_df['day'] = demand_df.index.day
demand_df['hour'] = demand_df.index.hour
demand_df.to_csv('demand.csv')

DA_prices = {}
 
for c in countries_dict:
    try:
        DA_prices[c] = client.query_day_ahead_prices(c, start=start,end=end)
        DA_prices[c] = pd.DataFrame(DA_prices[c],columns=[c])
        print(f'{c} correctly fecthed')
    except:
        print(f'{c} gave problems')
        pass

prices_df = pd.DataFrame(columns=list(countries_dict))

for c in countries_dict:
    try:
        prices_df[c] = prices_df[c][('Actual Load')]    
    except:
        pass
 
prices_df.to_csv('DAprices.csv')

#generation = {}

# =============================================================================
# country_codes = ['AL','AT','BG','CZ','DE','ES','FI','FR','GR','HR','HU','IE',
#                 'IT','LT','LV','ME','MK','NL','PL','PT','RO','RS','SI','SK']
# =============================================================================

# =============================================================================
# country_code = 'UK'
# country_code_1 = 'AT'
# country_code_2 = 'DE_LU'
# country_code_3 = 'CZ'
# country_code_4 = 'NO_5'
# country_code_5 = 'CH'
# country_code_6 = 'BE'
# =============================================================================

# =============================================================================
# Day-ahead prices (EUR/MWh)
# =============================================================================

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

# =============================================================================
# countries_list = ['AT','DE','IT','NL','CH']
# 
# generation = {}
# for i in countries_list:
#     generation[i] = client.query_generation(i, start=start,end=end)
# 
# for i in countries_list:
#     generation[i].to_csv('generation_{}.csv'.format(i))
# =============================================================================


#generation = client.query_generation(country_code, start=start,end=end)
#generation_per_plant = client.query_generation_per_plant(country_code_1, start=start,end=end)
#generation_forecast = client.query_generation_forecast(country_code_1, start=start,end=end)
#wind_solar_forecast = client.query_wind_and_solar_forecast(country_code_1, start=start,end=end, psr_type=None)
#installed_generation_capacity = client.query_installed_generation_capacity(country_code_1, start=start,end=end)
#installed_generation_capacity_per_unit = client.query_installed_generation_capacity_per_unit(country_code_1, start=start,end=end)


# =============================================================================
# countries_list = ['AT','DE','IT','NL','CH']
# 
# Load = {}
# # Loop of demand queries for all regions
# for c in countries_list:
#     Load[c] = client.query_load(c, start=start,end=end)
#     Load[c] = pd.DataFrame(Load[c],columns=[c])
# for c in countries_list:
#     Load[c].to_csv('{} - Load.csv'.format(c))    
# 
# =============================================================================

#=============================================================================
# load and load forecast (MW)
start = pd.Timestamp('20160101', tz ='UTC')
end = pd.Timestamp('20211031', tz ='UTC')

Load = {} # dictionary to store dataframes for each region
countries_list=['BE','BG','CZ','DK_1','DK_2','DE','EE','IE','GR','ES','FR','HR','IT',
           'LV','LT','LU','HU','NL','AT','PL','PT','RO','SI','SK','FI','SE_1',
           'SE_2','SE_3','SE_4','GB','GB_NIR','NO_1','NO_2','NO_3','NO_4','NO_5',
           'CH','ME','MK','AL','RS','BA', 'XK'
           ] # List of regions

# Loop of demand queries for all regions
for c in countries_list:
    Load[c] = client.query_load(c, start=start,end=end)
    Load[c] = pd.DataFrame(Load[c],columns=[c])

# Join all region dataframes into one single dataframe
Load_df = Load[countries_list[0]]
for c in countries_list[1:]:
    Load_df = pd.concat([Load_df, Load[c]], axis=1)

# Filter datframe to hourly values (15 minute values are removed) and export to csv
Load_df = Load_df.reset_index()
old_columns=list(Load_df.columns)
new_columns=old_columns.copy()
new_columns[0]='Date'
Load_df.columns=new_columns
Load_df['Minute'] = Load_df.Date.apply(lambda x: x.minute)
Load_df = Load_df[Load_df.Minute==0]
Load_df = Load_df.drop(['Minute'], axis=1)
Load_df = Load_df.set_index('Date')
Load_df.to_csv('Load.csv')
# =============================================================================

#load = client.query_load(country_code_1, start=start,end=end)
#load_forecast = client.query_load_forecast(country_code_1, start=start,end=end)

#day-ahead scheduled (commercial) exchanges (MW)
#scheduled_exchanges = client.query_scheduled_exchanges(country_code_1, country_code_2, start=start,end=end)

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
