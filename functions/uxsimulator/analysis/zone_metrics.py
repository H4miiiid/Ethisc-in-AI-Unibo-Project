import pandas as pd
from collections import defaultdict
import os.path

import uxsimulator.analysis.utils
from io_utils import vprint


def _calculate_zone_inflow(
    vehicles: pd.DataFrame
) -> dict:
    """
    Calculates passenger inflows and outflows for each zone by AV position.
    Processes vehicle trajectories to determine how many passengers enter and exit
    each zone, grouped by the AV position (origin of the vehicle).

    Args:
        vehicles (pd.DataFrame): DataFrame containing vehicle trajectory data with
                                columns for vehicle_id, AV_position, dn (passengers),
                                and id_zone_link.
                                
    Returns:
        dict: Nested dictionary with structure {zone: {av_position: {'inflow': n, 'outflow': m}}}
    """
    # Initialize dictionary variable
    zone_group_counts = defaultdict(lambda: defaultdict(lambda: {'inflow': 0, 'outflow': 0}))

    # Fill dictionary with counts
    for _, vehicle_df in vehicles.groupby('vehicle_id'):
        orig, pax = vehicle_df[['AV_position', 'dn']].iloc[0]
        for i, zone in enumerate(vehicle_df['id_zone_link']):
            if i != 0:
                zone_group_counts[zone][orig]['inflow'] += pax
            if i != vehicle_df.shape[0] - 1:
                zone_group_counts[zone][orig]['outflow'] += pax

    return zone_group_counts


def _transform_inflow_dict_to_dataframe(
    res_dict: dict
) -> pd.DataFrame:
    """
    Transforms the nested dictionary of zone flows into a wide-format DataFrame.
    Converts the nested dictionary structure from _compute_zone_passenger_flows into
    a pivot table with zones as rows and columns for inflow/outflow by AV position.

    Args:
        res_dict (dict): Nested dictionary with structure {zone: {av_position: {'inflow': n, 'outflow': m}}}
        
    Returns:
        pd.DataFrame: Wide-format DataFrame with zone flows pivoted by AV position.
    """
    res_df = pd.DataFrame([
        {'id_zone': zone, 'AV_position': av, 'inflow': counts['inflow'], 'outflow': counts['outflow']}
        for zone, av_pos in res_dict.items()
        for av, counts in av_pos.items()
    ])
    
    res_df = res_df.pivot(index='id_zone', columns='AV_position', values=['inflow', 'outflow'])

    res_df.columns = [f"{col[0]}_from_{col[1]}" for col in res_df.columns]
    res_df = res_df.reset_index()
    return res_df


def _average_seed_results(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes the average metrics across all simulation seeds for each zone and hour.

    Args:
        df (pd.DataFrame): DataFrame containing simulation results with columns for
                        id_zone, hour, seed, and various metrics.
                        
    Returns:
        pd.DataFrame: DataFrame with metrics averaged across seeds, grouped by zone and hour.
    """
    return (
        df
        .groupby(['id_zone', 'hour'])
        .agg(['mean', 'std'])
        .reset_index()
        .drop(columns=['seed'], level=0)
        .pipe(lambda x: x.set_axis([f'{col[0]}_{col[1]}' if col[1] else col[0] 
                                for col in x.columns], axis=1))
    )


def _compute_hourly_inflow(
    hour: int,
    datapath: str,
    namefile_nodes: str,
    namefile_edges: str,
    namefile_vehicles: str,
    namefile_save: str|None = None,
    seed: int = 0
) -> pd.DataFrame:
    """
    Creates an hourly inflow-outflow (IO) table for vehicles and zones from simulation output.
    
    Args:
        hour (int): The hour for which the IO table is created.
        datapath (str): The base path to the simulation output files.
        save (bool, optional): Whether to save the IO table to a file.
        
    Returns:
        pd.DataFrame: The resulting IO table DataFrame for the specified hour.
    """
    # Load initial dataframes
    name_iter = f"from_{hour}_to_{hour+1}_seed_{seed}"
    nodes = uxsimulator.analysis.utils.read_output(namefile=f"{namefile_nodes}.csv", datapath=datapath,  namecols=['node_id', 'x', 'y', 'AV_position'])
    links = uxsimulator.analysis.utils.read_output(namefile=f"{namefile_edges}.csv", datapath=datapath, namecols=['link_id', 'u','v', 'length', 'maxspeed', 'lanes', 'id_zone'])
    vehicles = uxsimulator.analysis.utils.read_output(namefile=f"{namefile_vehicles}_{name_iter}.csv", datapath=datapath, dtype={0: str}).rename(columns={'name':'vehicle_id'})
    # Elaborate datasets
    vehicles = vehicles.iloc[:-1]
    vehicles = (
        vehicles[vehicles['link'].apply(uxsimulator.analysis.utils.is_valid_string)]
        .merge(links[['link_id', 'id_zone']], left_on='link', right_on='link_id', how='left')
        .drop(columns=['link_id'])
        .rename(columns={'id_zone':'id_zone_link'})
        .merge(nodes[['node_id', 'AV_position']], left_on='orig', right_on='node_id', how='left')
        .drop(columns=['node_id'])
        .groupby(['vehicle_id'])
        .apply(lambda group: group[(group['id_zone_link'] != group['id_zone_link'].shift(1)) | (group.index == group.index[0])])
        .reset_index(drop=True)
    )

    # From vehicles to count dictionary to count dataframe
    zone_group_counts = _calculate_zone_inflow(vehicles)
    df_pivot = _transform_inflow_dict_to_dataframe(zone_group_counts)
    
    # Save and return
    if namefile_save is not None:
        df_pivot.to_parquet(f"{datapath}/{namefile_save}_{name_iter}.parquet", index=False)
    return df_pivot


def _compute_hourly_starting(
    hour: int,
    datapath: str,
    namefile_edges: str,
    namefile_vehicles: str,
    namefile_save: str|None = None,
    seed: int = 0
) -> pd.DataFrame:
    """
    Creates an hourly table of trip starting per zone from the simulation output.
    This function counts how many vehicles start their trips in each zone during 
    the specified hour.

    Args:
        hour (int): The hour for which the origins table is created.
        datapath (str): The base path to the simulation output files.
        version (str): Version identifier for saved output files.
        namefile_edges (str): Base filename for edge/link data.
        namefile_vehicles (str): Base filename for vehicle trajectory data.
        save (bool): Whether to save the origins table to a file.
        seed (int): Simulation seed number.
        
    Returns:
        pd.DataFrame: DataFrame with zones and their respective trip origin counts.
    """
    # Load initial dataframes
    name_iter = f"from_{hour}_to_{hour+1}_seed_{seed}"
    links = uxsimulator.analysis.utils.read_output(namefile=f"{namefile_edges}.csv", datapath=datapath, namecols=['link_id', 'u','v', 'length', 'maxspeed', 'lanes', 'id_zone'])
    vehicles = uxsimulator.analysis.utils.read_output(namefile=f"{namefile_vehicles}_{name_iter}.csv", datapath=datapath, dtype={0: str}).rename(columns={'name':'vehicle_id'})

    # Elaborate datasets
    vehicles = vehicles.iloc[:-1]
    df_pivot = (
        vehicles[vehicles['link'].apply(uxsimulator.analysis.utils.is_valid_string)]
        .groupby(['vehicle_id']).head(1).reset_index()
        .merge(links[['link_id', 'id_zone']], left_on='link', right_on='link_id', how='left')
        .drop(columns=['link_id'])
        .rename({'id_zone_link': 'id_zone'}, axis=1)
        .groupby('id_zone')
        .agg(starting_from_zone=('vehicle_id', 'nunique'))
        .reset_index()
    )
    
    if namefile_save is not None:
        df_pivot.to_parquet(f"{datapath}/{namefile_save}_{name_iter}.parquet", index=False)
    return df_pivot


def _compute_hourly_traffic(
    hour: int,
    datapath: str,
    namefile_edges: str,
    namefile_vehicles: str,
    namefile_save: str|None = None,
    seed: int = 0
) -> pd.DataFrame:
    """
    Creates an hourly table of traffic density per zone from simulation output.

    This function counts the number of vehicle observations in each zone during
    the specified hour, providing a measure of traffic intensity.

    Args:
        hour (int): The hour for which the traffic density table is created.
        datapath (str): The base path to the simulation output files.
        version (str): Version identifier for saved output files.
        namefile_edges (str): Base filename for edge/link data.
        namefile_vehicles (str): Base filename for vehicle trajectory data.
        save (bool): Whether to save the traffic density table to a file.
        seed (int): Simulation seed number.
        
    Returns:
        pd.DataFrame: DataFrame with zones and their respective traffic density values.
    """
    # Load initial dataframes
    name_iter = f"from_{hour}_to_{hour+1}_seed_{seed}"
    links = uxsimulator.analysis.utils.read_output(namefile=f"{namefile_edges}.csv", datapath=datapath, namecols=['link_id', 'u','v', 'length', 'maxspeed', 'lanes', 'id_zone'])
    vehicles = uxsimulator.analysis.utils.read_output(namefile=f"{namefile_vehicles}_{name_iter}.csv", datapath=datapath, dtype={0: str}).rename(columns={'name':'vehicle_id'})

    # Elaborate datasets
    vehicles = vehicles.iloc[:-1]
    df_pivot = (
        vehicles[vehicles['link'].apply(uxsimulator.analysis.utils.is_valid_string)].reset_index(drop=True)
        .merge(links[['link_id', 'id_zone']], left_on='link', right_on='link_id', how='left').reset_index(drop=True)
        .drop(columns=['link_id'])
        .rename({'id_zone_link': 'id_zone'}, axis=1)
        .groupby('id_zone')
        .agg(traffic_in_zone=('vehicle_id', 'nunique'))
        .reset_index()
    )
    
    # Save and return
    if namefile_save is not None:
        df_pivot.to_parquet(f"{datapath}/{namefile_save}_{name_iter}.parquet", index=False)
    return df_pivot


def compute_full_table(
    table_type: str,
    datapath: str,
    namefile_edges: str,
    namefile_vehicles: str,
    namefile_nodes: str|None = None,
    namefile_save: str|None = None,
    list_hours: list = [i for i in range(24)],
    list_seeds: list = [0],
    verbose: bool = False
) -> pd.DataFrame:
    """
    Creates a comprehensive traffic analysis table for all hours and seeds by computing either 
    starting points (S), inflow-outflow (IO), or traffic density (T) metrics.

    Args:
        table_type (str): Analysis type to perform - "S" (starting points per area), 
                        "IO" (inflow-outflow between zones), or "T" (traffic density per zone).
        datapath (str): Base path to the simulation output files.
        version_save (str): Version identifier for saved output files.
        total_hours (int): Number of hours to analyze (default: 24).
        total_seed (int): Number of random seeds/simulation runs to process (default: 1).
        full_save (bool): Whether to save the aggregated table of all hours and seeds.
        local_save (bool): Whether to save individual hourly tables per seed.
        namefile_nodes (str): Base filename for node data.
        namefile_edges (str): Base filename for edge/link data.
        namefile_vehicles (str): Base filename for vehicle trajectory data.
        
    Returns:
        pd.DataFrame: The complete analysis table with averaged results across all seeds.
    """
    # Iterate over the seeds and hours
    for seed in list_seeds:
        for hour in list_hours:

            # If the file already exists, load it
            if os.path.exists(f"{datapath}/{namefile_save}_from_{hour}_to_{hour+1}_seed_{seed}.parquet"):
                vprint(text=f"...Loading hour {hour}, seed {seed}...", verbose=verbose)
                df_hourly = pd.read_parquet(f"{datapath}/{namefile_save}_from_{hour}_to_{hour+1}_seed_{seed}.parquet")
            
            # If the file does not exist, create it
            else:
                vprint(text=f"...Computing hour {hour}, seed {seed}...", verbose=verbose)
                if table_type == "IO":
                    df_hourly = _compute_hourly_inflow(hour=hour, datapath=datapath, 
                                                    namefile_nodes=namefile_nodes, namefile_edges=namefile_edges, namefile_vehicles=namefile_vehicles,
                                                    namefile_save=namefile_save, seed=seed)
                elif table_type == "S":
                    df_hourly = _compute_hourly_starting(hour=hour, datapath=datapath,
                                                    namefile_edges=namefile_edges, namefile_vehicles=namefile_vehicles,
                                                    namefile_save=namefile_save, seed=seed)
                elif table_type == "T":
                    df_hourly = _compute_hourly_traffic(hour=hour, datapath=datapath,
                                                    namefile_edges=namefile_edges, namefile_vehicles=namefile_vehicles,
                                                    namefile_save=namefile_save, seed=seed)
            
            # Elaborate output dataset
            if df_hourly.isna().sum().sum() > 0:
                df_hourly.fillna(0, inplace=True)
            df_hourly['seed'] = seed
            df_hourly['hour'] = hour
            df_all = df_hourly if ((seed == list_seeds[0])&(hour==list_hours[0])) else pd.concat([df_all, df_hourly])

    # Evaluate results of different simulations (seeds)
    df_all = df_all.sort_values(by=['id_zone', 'hour', 'seed']).reset_index(drop=True)
    if len(list_seeds) > 1:
        df_all = _average_seed_results(df_all)

    # Save and return
    if namefile_save is not None:
        df_all.to_parquet(f"{datapath}/{namefile_save}.parquet", index=False)
    vprint(text=f"Done!", verbose=verbose)
    return df_all