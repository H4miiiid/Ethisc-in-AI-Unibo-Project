import pandas as pd
import networkx as nx
import math

import road_network


def find(
    df_nodes,
    df_edges, 
    df_spiras
)-> pd.DataFrame:
    """
    Finds the catchment area for each spira by calculating time distances on the road network at free flow speed.

    Args:
        `df_nodes` (`pd.DataFrame`): DataFrame containing road network nodes.
        `df_edges` (`pd.DataFrame`): DataFrame containing road network edges.
        `df_spiras` (`pd.DataFrame`): DataFrame containing spira information.

    Returns:
        `pd.DataFrame`: DataFrame containing the catchment area of each spira with road details and durations.
    """
    G = road_network.create_road_graph(df_edges=df_edges, df_nodes=df_nodes, weight_col='free_flow_time')
    all_shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    for iter in range(df_spiras.shape[0]): # tqdm(range(spira_close_roads.shape[0]), desc='Computing time distance between spiras and roads',bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'):
        ref_edge = (
            df_spiras['u'][iter], 
            df_spiras['v'][iter], 
            df_spiras['key'][iter]
        )
        result = _find_edges_distance(graph=G, edge1=ref_edge, all_shortest_paths=all_shortest_paths)
        result['spira_unique_id'] = df_spiras['spira_unique_id'][iter]
        spira_catchment_area = result if iter == 0 else pd.concat([spira_catchment_area, result])
    
    spira_catchment_area = pd.merge(
        spira_catchment_area[['u', 'v', 'key', 'spira_unique_id', 'duration']],
        df_edges[['u', 'v', 'key', 'free_flow_time', 'id_zone']],
        on=['u','v','key']
    )

    return spira_catchment_area


def _find_edges_distance(
    graph, 
    edge1, 
    all_shortest_paths
):
    edges_within_threshold = []

    # Add edges to the set based on shortest path distance
    for edge2 in graph.edges(keys=True):
        if edge1 != edge2:
            # Calculate the shortest path distance between the two edges
            distances = []
            for src in [edge1[0], edge1[1]]:
                for tgt in [edge2[0], edge2[1]]:
                    distance = all_shortest_paths.get(src, {}).get(tgt, math.inf)
                    distances.append(distance)
            d1 = min(distances)
            # Calculate the shortest path distance between the two edges (inverse order)
            distances = []
            for src in [edge2[0], edge2[1]]:
                for tgt in [edge1[0], edge1[1]]:
                    distance = all_shortest_paths.get(src, {}).get(tgt, math.inf)
                    distances.append(distance)
            d2 = min(distances)
            edges_within_threshold.append((edge2[0], edge2[1], edge2[2], min(d1,d2)))
        else:
            edges_within_threshold.append((edge2[0], edge2[1], edge2[2], 0))
    
    df = pd.DataFrame(edges_within_threshold, columns=['u', 'v', 'key', 'duration']) #time = time_distance
    return df


def filter(
    df_catch: pd.DataFrame,
    time_threshold: float,
) -> pd.DataFrame:
    """
    Filters the spira catchment area based on a time threshold and calculates time-proportional weights.

    Args:
        `df_catch` (`pd.DataFrame`): DataFrame containing spira catchment area data.
        `time_threshold` (`float`): Maximum duration threshold for filtering roads.

    Returns:
        `pd.DataFrame`: Filtered and updated catchment area DataFrame with proportional time weights.
    """
    # Filter close roads only
    df_catch = df_catch[df_catch['duration'] < time_threshold]
    # Compute time-proportional weight
    df_catch = (
        df_catch
        .assign(
            tot_t_imp = lambda df: df.groupby('spira_unique_id')['free_flow_time'].transform('sum'),
            prop_t_imp = lambda df: df['free_flow_time'] / df['tot_t_imp']
        )
        .drop(columns=['free_flow_time', 'tot_t_imp'], axis=1)
    )
    return df_catch

