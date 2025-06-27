import pandas as pd
import uxsimulator.analysis.utils


def _completed_vehicles(
    df: pd.DataFrame
):
    """
    Identifies and counts vehicles which completed their trip.

    Args:
        `df` (`pd.DataFrame`): DataFrame containing vehicle trajectory data with
                           columns `vehicle_id`, `link`, `s`, and `x`.

    Returns:
        Id of the vehicles identified as completed.
    """
    return df[df['link'] == 'trip_end']['vehicle_id'].tolist()


def _are_stuck_vehicles(
    df: pd.DataFrame,
    n: int
):
    """
    Identifies vehicles that are stuck, i.e. those that haven't moved for the n minutes, at the end of the simulation.

    Args:
        `df` (`pd.DataFrame`): DataFrame containing vehicle trajectory data with
                           columns `vehicle_id`, `link`, `t`.
        `n` (`int`): Number of consecutive minutes while stuck.

    Returns:
        Id of the vehicles identified as stuck.
    """
    df_ended = _completed_vehicles(df)
    df = df[~df['vehicle_id'].isin(df_ended)]
    stuck_vehicles = []
    grouped = df.groupby('vehicle_id')

    for vehicle_id, group in grouped:
        group = (
            group
            .sort_values(by='t', ascending=False)
            .reset_index(drop=True)
        )
        if group.loc[0, 'link'] == "waiting_at_origin_node" or group.loc[0, 'link'] == "trip_aborted":
            stuck_vehicles.append(vehicle_id)
        elif len(group) == 1:
            stuck_vehicles.append(vehicle_id)
        else:
            t1 = group.loc[0, 't']
            t0 = group.loc[1, 't']
            if t1-t0 > n * 60:
                stuck_vehicles.append(vehicle_id)
    return stuck_vehicles


def _have_been_stuck_vehicles(
    df: pd.DataFrame,
    n: int
):
    """
    Identifies vehicles that has been stuck, i.e. those that haven't moved for the n minutes, 
    at least one time during the entire simulation.

    Args:
        `df` (`pd.DataFrame`): DataFrame containing vehicle trajectory data with
                        columns `vehicle_id` and `t`.
        `n` (`int`): Number of consecutive minutes while stuck.

    Returns:
        Id of the vehicles identified as stuck.
    """
    df['duration_on_link'] = -df.groupby('vehicle_id')['t'].diff(-1)
    df['is_stuck'] = df['duration_on_link'] > 60 * n
    tmp = (df.groupby('vehicle_id')['is_stuck'].sum()>0).reset_index()
    return list(tmp.loc[tmp['is_stuck'],'vehicle_id'].unique())      


def identify_hourly_vehicles(
    type_veh: str,
    hour: int,
    seed: int = 0,
    namefile_vehicles: str = "UXsim_vehicles/AreaVerde_vehicles_ALL_v7",
    datapath: str = "data/results",
    n_stuck: int|None = 5
) -> list:
    """
    For a specific hour and seed in the simulation
        - If `type_veh == "stuck"` : find the vehicles that are stuck at the end of the simulation.
        - If `type_veh == "full_stuck"` : find vehicles that have been stuck during the simulation.
        - If `type_veh == "completed"` : find the vehicles that completed the trips. 

    Args:
        `hour` (`int`): The hour for which vehicles are counted.
        `seed` (`int`, optional): Simulation randomization seed. Defaults to 0.
        `namefile_vehicles` (`str`, optional): Base filename for vehicle data. Defaults to "UXsim_vehicles/AreaVerde_vehicles_ALL_v7".
        `datapath` (`str`, optional): Path to the simulation output data files. Defaults to "data/results".
        `n_stuck` (`int`, optional): Number of recent positions to consider for detecting stuck vehicles. Defaults to 5.

    Returns:
         Ids int: Total number of vehicles identified as stuck during the hour.
    """
    name_iter = f"from_{hour}_to_{hour+1}_seed_{seed}"
    vehicles = uxsimulator.analysis.utils.read_output(
        namefile=f"{namefile_vehicles}_{name_iter}.csv",
        datapath=datapath).rename(columns={'name': 'vehicle_id'})

    if type_veh == "stuck":
        return _are_stuck_vehicles(vehicles, n_stuck)
    elif type_veh == "full_stuck":
        return _have_been_stuck_vehicles(vehicles, n_stuck)
    elif type_veh == "completed":
        return _completed_vehicles(vehicles)
    else:
        return 0


def count_hourly_vehicles(
    type_veh: str,
    hour: int,
    seed: int = 0,
    namefile_vehicles: str = "UXsim_vehicles/AreaVerde_vehicles_ALL_v7",
    datapath: str = "data/results",
    n_stuck: int|None = 5
) -> int:
    """
    For a specific hour and seed in the simulation
        - If `type_veh == "stuck"` : count the vehicles that are stuck at the end of the simulation.
        - If `type_veh == "full_stuck"` : count vehicles that have been stuck during the simulation.
        - If `type_veh == "completed"` : count the vehicles that completed the trips. 

    Args:
        `hour` (`int`): The hour for which vehicles are counted.
        `seed` (`int`, optional): Simulation randomization seed. Defaults to 0.
        `namefile_vehicles` (`str`, optional): Base filename for vehicle data. Defaults to "UXsim_vehicles/AreaVerde_vehicles_ALL_v7".
        `datapath` (`str`, optional): Path to the simulation output data files. Defaults to "data/results".
        `n_stuck` (`int`, optional): Number of recent positions to consider for detecting stuck vehicles. Defaults to 5.

    Returns:
        `int`: Total number of vehicles identified as stuck during the hour.
    """
    name_iter = f"from_{hour}_to_{hour+1}_seed_{seed}"
    vehicles = uxsimulator.analysis.utils.read_output(
        namefile=f"{namefile_vehicles}_{name_iter}.csv",
        datapath=datapath,
        dtype={0: str}).rename(columns={'name': 'vehicle_id'})

    if type_veh == "stuck":
        return len(_are_stuck_vehicles(vehicles, n_stuck))
    elif type_veh == "full_stuck":
        return len(_have_been_stuck_vehicles(vehicles, n_stuck))
    elif type_veh == "completed":
        return len(_completed_vehicles(vehicles))
    else:
        return 0


def _stucking_links(
    df: pd.DataFrame, 
    n: int = 5
) -> pd.DataFrame:
    """
    Analyzes which links contain stuck vehicles and calculates relevant statistics.

    Args:
        `df` (`pd.DataFrame`): DataFrame containing vehicle trajectory data with columns
                        `vehicle_id`, `link`, and `t`.
        `n` (`int`, optional): Number of consecutive positions to check for immobility. Default to 5.

    Returns:
        `pd.DataFrame`: DataFrame with links statistics including stuck vehicle counts and percentages.
                    Columns include 'link', 'stuck_count', 'total_count', and 'stuck_percentage'.
    """
    df['duration_on_link'] = -df.groupby('vehicle_id')['t'].diff(-1)
    df['is_stuck'] = (df['duration_on_link'] > 60 * n)
    stuck_counts = (
        df
        [df['is_stuck']]
        ['link']
        .value_counts()
        .reset_index(name='stuck_count')
    )
    
    total_counts = df.groupby('link').size().reset_index(name='total_count')
    
    link_stats = pd.merge(stuck_counts, total_counts, on='link', how='right')
    link_stats['stuck_count'] = link_stats['stuck_count'].fillna(0)
    link_stats['stuck_percentage'] = (link_stats['stuck_count'] / link_stats['total_count']) * 100
    
    link_stats = link_stats[link_stats['link'].apply(uxsimulator.analysis.utils.is_valid_string)]
    return link_stats


def identify_hourly_stucking_links(
    stuck_hour: int,
    stuck_seed: int = 0,
    namefile_vehicles: str = "UXsim_vehicles/AreaVerde_vehicles_ALL_v7",
    datapath: str = "data/results",
    n_stuck: int = 5,
) -> pd.DataFrame:
    """
    Identifies and analyzes links with stuck vehicles for a specific hour of simulation.

    Args:
        stuck_hour (int): The hour for which the stuck links are analyzed.
        namefile_vehicles (str, optional): Base filename for vehicle data. Defaults to "UXsim_vehicles/AreaVerde_vehicles_ALL_v7".
        datapath (str, optional): Path to the simulation output data files. Defaults to "data/results".
        n_stuck (int, optional): Number of recent positions to consider for detecting stuck vehicles. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame containing link statistics, including stuck counts, total vehicle counts,
                    and the percentage of vehicles stuck on each link.
    """
    name_iter = f"from_{stuck_hour}_to_{stuck_hour+1}_seed_{stuck_seed}"
    vehicles = uxsimulator.analysis.utils.read_output(namefile=f"{namefile_vehicles}_{name_iter}.csv",
                                 datapath=datapath, dtype={0: str}).rename(columns={'name':'vehicle_id'})
    return _stucking_links(vehicles, n_stuck)
