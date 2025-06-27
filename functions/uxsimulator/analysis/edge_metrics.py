import pandas as pd
import uxsimulator.analysis.utils
import numpy as np


def _calculate_hourly_traffic(
    hour: int,
    seed: int,
    datapath: str,
    namefile_nodes: str,
    namefile_traffic: str,
) -> pd.DataFrame:
    """
    Creates a DataFrame containing traffic information for a specific hour, including 
    origin and destination node coordinates for each link.

    Args:
        hour (int): The hour for which the link traffic data is generated.
        datapath (str, optional): Path to the simulation output data files. Defaults to "data/results".

    Returns:
        pd.DataFrame: A DataFrame with traffic information, including link-specific traffic 
                      data and corresponding geographical coordinates.
    """
    name_iter = f"from_{hour}_to_{hour+1}_seed_{seed}"
    traffic = uxsimulator.analysis.utils.read_output(namefile=f"{namefile_traffic}_{name_iter}.csv", datapath=datapath)
    nodes = uxsimulator.analysis.utils.read_output(namefile=f"{namefile_nodes}.csv", datapath=datapath, namecols=['node_id', 'x', 'y', 'AV_position'])
    
    traffic[traffic['average_travel_time'] < 0] = None
    return (
        traffic
        .merge(nodes.rename(columns={'node_id': 'start_node', 'x': 'x_origin', 'y': 'y_origin'}), on='start_node')
        .merge(nodes.rename(columns={'node_id': 'end_node', 'x': 'x_dest', 'y': 'y_dest'}), on='end_node')
    )


def calculate_traffic(
    namefile_nodes: str,
    namefile_traffic: str,
    list_hours: list = [i for i in range(24)],
    list_seeds: list = [0],
    datapath: str = "data/results"
) -> pd.DataFrame:
    for h in list_hours:
        for s in list_seeds:
            tmp_traffic = _calculate_hourly_traffic(hour=h, seed=s, datapath=datapath, namefile_traffic=namefile_traffic, namefile_nodes=namefile_nodes)
            tmp_traffic['hour'] = h
            tmp_traffic['seed'] = s
            traffic = tmp_traffic if ((h==list_hours[0])&(s==list_seeds[0])) else pd.concat([traffic, tmp_traffic])
    cols = ['link','start_node', 'end_node','length', 'x_origin', 'y_origin', 'AV_position_x', 'x_dest', 'y_dest',
       'AV_position_y', 'hour']
    if len(list_seeds) > 1:
        traffic['var_travel_time'] = traffic['stddiv_travel_time']**2
        traffic = traffic.groupby(cols).mean(numeric_only=True).reset_index()
        traffic['stddiv_travel_time'] = np.sqrt(traffic['var_travel_time'])
        traffic.drop(columns=['var_travel_time'], inplace=True)
    return traffic
