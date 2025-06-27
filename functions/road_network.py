import geopandas as gpd
import pandas as pd
import osmnx as ox
import numpy as np
from shapely.geometry import Point, LineString
from constants import CRS_LATLONG, CRS_PROJECTED
from warnings import catch_warnings, simplefilter
import networkx as nx
import ast


def create_road_data(
    df_BAV: gpd.GeoDataFrame,
    relevant_highway: bool,
    connected_network: bool,
    edges_namefile: str = None, 
    nodes_namefile: str = None, 
    datapath: str = None,
):
    """
    Extracts and preprocesses road data using OSMNX. Optionally filters the largest connected component and saves the data.

    Args:
        `df_BAV` (`gpd.GeoDataFrame`): GeoDataFrame representing the area of interest.
        `relevant_highway` (`bool`): Whether to filter for relevant highway types.
        `connected_network` (`bool`): Whether to filter the largest connected network.
        `edges_namefile` (`str`, optional): Name of the file to save edges data. Defaults to `None`.
        `nodes_namefile` (`str`, optional): Name of the file to save nodes data. Defaults to `None`.
        `datapath` (`str`, optional): Directory path to save the files. Defaults to `None`.

    Returns:
        `tuple`: Two GeoDataFrames, one for edges and one for nodes.
    """
    # Extract from OSMNX and preprocess the data
    edges, nodes = get_osmxn_data(df_bav=df_BAV, relevant_highway=relevant_highway)

    # Fix the network manually
    if relevant_highway:
        edges, nodes = fix_manually(edges, nodes)

    # Filter connected component
    if connected_network:
        edges, nodes = get_largest_component(edges, nodes)

    if edges_namefile and nodes_namefile and datapath:
        save_road_data(edges, nodes, edges_namefile, nodes_namefile)
    return edges, nodes


def save_road_data(
    edges: gpd.GeoDataFrame, 
    nodes: gpd.GeoDataFrame,
    edges_namefile: str, 
    nodes_namefile: str
):
    """
    Saves road network edges and nodes to GeoJSON files in a subfolder of the current one named results.

    Args:
        `edges` (`gpd.GeoDataFrame`): GeoDataFrame containing edges data.
        `nodes` (`gpd.GeoDataFrame`): GeoDataFrame containing nodes data.
        `edges_namefile` (`str`): Name of the file to save edges data.
        `nodes_namefile` (`str`): Name of the file to save nodes data.

    Returns:
        `None`
    """
    edges.to_file(f"results/{edges_namefile}", driver="GeoJSON")
    nodes.to_file(f"results/{nodes_namefile}", driver="GeoJSON")


def get_osmxn_data(
    df_bav: gpd.GeoDataFrame,
    relevant_highway: bool,
) -> tuple:
    """
    Extracts road data using OSMNX and then preprocesses the nodes and edges.

    Args:
        `df_bav` (`gpd.GeoDataFrame`): GeoDataFrame representing the area of interest.
        `relevant_highway` (`bool`): Whether to filter for relevant highway types.

    Returns:
        `tuple`: Two GeoDataFrames, one for edges and one for nodes.
    """

    edges, nodes = _extract_osmnx_data(df_bav=df_bav, relevant_highway=relevant_highway)

    nodes = _fix_osmnx_nodes(nodes)
    edges = _fix_osmnx_edges(edges)

    return edges, nodes


def _extract_osmnx_data(
    df_bav: gpd.GeoDataFrame,
    relevant_highway: bool,
) -> tuple:
    """
    Extracts road network data from OSMNX using a custom filter.

    Args:
        `df_bav` (`gpd.GeoDataFrame`): GeoDataFrame representing the area of interest.
        `relevant_highway` (`bool`): Whether to filter for relevant highway types.

    Returns:
        `tuple`: Two GeoDataFrames, one for edges and one for nodes.
    """
    if relevant_highway:
        highway_ok = ['motorway', 'motorway_link', 'primary', 'secondary', 'tertiary', 'unclassified', 
                'trunk', 'trunk_link','secondary_link','primary_link', 'tertiary_link']
        pattern = '|'.join(highway_ok)
        custom_filter=f"['highway'~'{pattern}']"
    else:
        custom_filter=None

    G = ox.graph_from_polygon(
            polygon=df_bav.to_crs(CRS_LATLONG).geometry.iloc[0],
            network_type='drive',
            simplify=True, 
            custom_filter=custom_filter
        )
    nodes, edges = ox.graph_to_gdfs(G)
    return edges, nodes


def _fix_osmnx_nodes(
    ox_nodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Fixes and preprocesses OSMNX node data.

    Args:
        `ox_nodes` (`gpd.GeoDataFrame`): GeoDataFrame containing raw OSMNX node data.

    Returns:
        `gpd.GeoDataFrame`: Processed GeoDataFrame containing nodes data.
    """
    ox_nodes = (
        ox_nodes
        .reset_index()
        .rename(columns={'osmid': 'node_id'})
        .set_geometry('geometry', crs=CRS_LATLONG)
        .to_crs(CRS_PROJECTED)
    )
    ox_nodes['coord1'] = ox_nodes.geometry.x
    ox_nodes['coord2'] = ox_nodes.geometry.y
    ox_nodes['node_id'] = ox_nodes['node_id'].astype(str)
    return ox_nodes[['node_id', 'coord1', 'coord2', 'geometry']]


def _fix_osmnx_edges(
    ox_edges: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Fixes and preprocesses OSMNX edge data.

    Args:
        `ox_edges` (`gpd.GeoDataFrame`): GeoDataFrame containing raw OSMNX edge data.

    Returns:
        `gpd.GeoDataFrame`: Processed GeoDataFrame containing edges data.
    """
    ox_edges = (
        ox_edges
        .reset_index()
        .to_crs(CRS_PROJECTED)
        .astype({'u': str, 'v': str, 'key': str})
        .query('u != v')
    )
    
    ox_edges['highway'] = ox_edges['highway'].astype(str).apply(lambda x: _process_columns(x, expected_type="str"))
    ox_edges['highway_ok'] = _check_highway_ok(ox_edges['highway'])
    ox_edges['maxspeed_imputed'] = _impute_missing_maxspeed(ox_edges) / 3.6 #moving to m/s
    ox_edges['lanes_imputed'] = _impute_missing_lane(ox_edges)
    ox_edges['free_flow_time'] = ox_edges['length'] / ox_edges['maxspeed_imputed'] #measured in seconds
    ox_edges['link_id'] = ox_edges['u'] + '_' + ox_edges['v'] + '_' + ox_edges['key']

    return ox_edges[['u', 'v', 'key', 'highway', 'length', 'geometry', 'maxspeed_imputed', 
                     'lanes_imputed', 'free_flow_time', 'link_id', 'highway_ok', 'oneway']]


def _impute_missing_maxspeed(
    df: gpd.GeoDataFrame,
    na_fill: float = 30
):
    """
    Imputes missing maxspeed values for edges.

    Args:
        `df` (`gpd.GeoDataFrame`): GeoDataFrame containing edge data with potential missing maxspeed values.
        `na_fill` (`float`): remaining missing values are filled with this quantity. Default to `30` (for speed in kmh)
    Returns:
        `pd.Series`: Series of imputed maxspeed values.
    """
    df['maxspeed'] = df['maxspeed'].astype(str).apply(_process_columns, expected_type="int")
    mean_maxspeed_by_highway = df.groupby('highway')['maxspeed'].transform('mean')
    values_imp = df['maxspeed'].fillna(mean_maxspeed_by_highway, inplace=False)
    values_imp = values_imp.fillna(na_fill)

    return values_imp


def _impute_missing_lane(
    df: gpd.GeoDataFrame
):
    """
    Imputes missing lane values for edges.

    Args:
        `df` (`gpd.GeoDataFrame`): GeoDataFrame containing edge data with potential missing lane values.

    Returns:
        `np.array`: Array of imputed lane values.
    """
    df['lanes'] = df['lanes'].astype(str).apply(_process_columns, expected_type="int")
    missing_lanes_mask = df['lanes'].isna()

    with catch_warnings():
        simplefilter("ignore", category=RuntimeWarning)
        grouped = df.groupby(['oneway', 'highway'])['lanes']
        group_medians = grouped.median()
        group_medians = group_medians.reindex(pd.MultiIndex.from_product([df['oneway'].unique(), df['highway'].unique()]), fill_value=np.nan)
        group_medians = group_medians.to_dict()

    group_medians_series = pd.Series(group_medians)

    lanes_imputed = df['lanes'].values
    lanes_imputed[missing_lanes_mask] = df[missing_lanes_mask].apply(
        lambda row: group_medians_series.get((row['oneway'], row['highway']), 1.0 if row['oneway'] else 2.0), axis=1
    )
    lanes_imputed[np.isnan(lanes_imputed)] = np.where(df['oneway'].values[np.isnan(lanes_imputed)], 1.0, 2.0)

    return lanes_imputed


def _check_highway_ok(
    highway: list
) -> list:
    """
    Checks if the highway type is within the acceptable list.

    Args:
        `highway` (`list`): List of highway types.

    Returns:
        `list`: Boolean list indicating whether each highway type is acceptable.
    """
    good_highways = {'motorway', 'motorway_link', 'primary', 'secondary', 'tertiary', 'unclassified', 
                'trunk', 'trunk_link','secondary_link','primary_link', 'tertiary_link'}
    return [any(hw in entry for hw in good_highways) for entry in highway]
        

def fix_manually(
    edges_tbf: gpd.GeoDataFrame, 
    nodes_tbf: gpd.GeoDataFrame, 
):
    """
    Manually fixes road network edges and nodes by adding missing edges and nodes.

    Args:
        `edges_tbf` (`gpd.GeoDataFrame`): GeoDataFrame containing edges to be fixed.
        `nodes_tbf` (`gpd.GeoDataFrame`): GeoDataFrame containing nodes to be fixed.

    Returns:
        `tuple`: Updated GeoDataFrames for edges and nodes.
    """
    new_edge = _add_edge(edges_tbf, nodes_tbf, id1='252169510', id2='252169103', coords1=[6947613.340770261, 4921671.587873447],
                      coords2=[6947665.422735082, 4921660.3822595235])
    edges_new = pd.DataFrame([new_edge])
    new_edge = _add_edge(edges_tbf, nodes_tbf, id2='252169510', id1='252169103', coords2=[6947613.340770261, 4921671.587873447],
                        coords1=[6947665.422735082, 4921660.3822595235])
    edges_new = pd.concat([edges_new, pd.DataFrame([new_edge])], ignore_index=True)
    new_edge = _add_edge(edges_tbf, nodes_tbf, '252169103', '4317179527', coords1=[6947665.422735082, 4921660.3822595235],
                        coords2=[6947790.316, 4921629.073])
    edges_new = pd.concat([edges_new, pd.DataFrame([new_edge])], ignore_index=True)
    new_edge = _add_edge(edges_tbf, nodes_tbf, id2='252169103', id1='4317179527', coords2=[6947665.422735082, 4921660.3822595235],
                        coords1=[6947790.316, 4921629.073])
    edges_new = pd.concat([edges_new, pd.DataFrame([new_edge])], ignore_index=True)
    new_edge = _add_edge(edges_tbf, nodes_tbf, '279205194', '4317179527', coords1=[6947806.345302537, 4921704.342970123],
                        coords2=[6947790.316, 4921629.073])
    edges_new = pd.concat([edges_new, pd.DataFrame([new_edge])], ignore_index=True)
    new_edge = _add_edge(edges_tbf, nodes_tbf, '245980299', '8992868196')
    edges_new = pd.concat([edges_new, pd.DataFrame([new_edge])], ignore_index=True)
    new_edge = _add_edge(edges_tbf, nodes_tbf, '8992868196', '279205194', coords2=[6947806.345302537, 4921704.342970123])
    edges_new = pd.concat([edges_new, pd.DataFrame([new_edge])], ignore_index=True)
    new_edge = _add_edge(edges_tbf, nodes_tbf, '4317179527', '82550591', coords1=[6947790.316, 4921629.073])
    edges_new = pd.concat([edges_new, pd.DataFrame([new_edge])], ignore_index=True)
    new_edge = _add_edge(edges_tbf, nodes_tbf, '250734900', '7610346791')
    edges_new = pd.concat([edges_new, pd.DataFrame([new_edge])], ignore_index=True)
    new_edge = _add_edge(edges_tbf, nodes_tbf, '251845817', '566327047')
    edges_new = pd.concat([edges_new, pd.DataFrame([new_edge])], ignore_index=True)
    new_edge = _add_edge(edges_tbf, nodes_tbf, '251844955','251846404')
    edges_new = pd.concat([edges_new, pd.DataFrame([new_edge])], ignore_index=True)

    edges = pd.concat([edges_tbf, edges_new], ignore_index=True)
    edges = gpd.GeoDataFrame(edges, geometry='geometry', crs=CRS_PROJECTED)

    nodes = {}
    for idx, row in edges.iterrows():
        line = row['geometry']
        u, v = row['u'], row['v']
        nodes[u] = line.coords[0]
        nodes[v] = line.coords[-1]
    nodes = pd.DataFrame({
        'node_id': list(nodes.keys()),
        'coord1': [coord[0] for coord in nodes.values()],
        'coord2': [coord[1] for coord in nodes.values()]
    })
    nodes['node_id'] = nodes['node_id'].astype(str)
    nodes['geometry'] = nodes.apply(lambda row: Point(row['coord1'], row['coord2']), axis=1)
    nodes = gpd.GeoDataFrame(nodes, geometry='geometry', crs=CRS_PROJECTED)

    return edges, nodes
 

def _add_edge(
    edges_tbf: gpd.GeoDataFrame, 
    nodes_tbf: gpd.GeoDataFrame, 
    id1: int|str, 
    id2: int|str, 
    coords1: float = None, 
    coords2: float = None,
    speed_mps: float = 30/3.6, 
    lanes: int = 2
):
    """
    Adds a new edge to the road network.

    Args:
        `edges_tbf` (`gpd.GeoDataFrame`): GeoDataFrame containing existing edges.
        `nodes_tbf` (`gpd.GeoDataFrame`): GeoDataFrame containing existing nodes.
        `id1` (`int`|`str`): ID of the first node.
        `id2` (`int`|`str`): ID of the second node.
        `coords1` (`float`, optional): Coordinates of the first node. Defaults to `None`.
        `coords2` (`float`, optional): Coordinates of the second node. Defaults to `None`.
        `speed_mps` (`float`, optional): Speed in meters per second. Defaults to `30/3.6` (for speed in ms).
        `lanes` (`int`, optional): Number of lanes. Defaults to `2`.

    Returns:
        `pd.Series`: Series representing the new edge.
    """
    try:
        p1 = nodes_tbf[nodes_tbf['node_id'] == id1]['geometry'].values[0]
    except IndexError:
        p1 = Point(coords1)
    try:
        p2 = nodes_tbf[nodes_tbf['node_id'] == id2]['geometry'].values[0]
    except IndexError:
        p2 = Point(coords2)
    line = LineString([p1, p2])
    line_gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=CRS_PROJECTED)
    length = line_gdf['geometry'].length.values[0]

    new_edge = [id1, id2, '0', "tertiary", length, line, speed_mps, lanes, length/speed_mps , f"{id1}_{id2}_0", True, True]
    new_edge_series = pd.Series(new_edge, index=edges_tbf.columns)

    return new_edge_series


def get_largest_component(
    edges_tbf: gpd.GeoDataFrame,
    nodes_tbf: gpd.GeoDataFrame
) -> tuple:
    """
    Filters the largest connected component of the road network.

    Args:
        `edges_tbf` (`gpd.GeoDataFrame`): GeoDataFrame containing edges.
        `nodes_tbf` (`gpd.GeoDataFrame`): GeoDataFrame containing nodes.

    Returns:
        `tuple`: Filtered GeoDataFrames for edges and nodes.
    """
    G1 = create_road_graph(df_edges=edges_tbf, df_nodes=nodes_tbf)
    strong_components = list(nx.strongly_connected_components(G1))
    largest_component = max(strong_components, key=len)
    large_component_nodes =  list(largest_component)

    nodes_tbf = nodes_tbf[nodes_tbf['node_id'].isin(large_component_nodes)]
    edges_tbf = edges_tbf[edges_tbf['u'].isin(large_component_nodes) & edges_tbf['v'].isin(large_component_nodes)]

    return edges_tbf, nodes_tbf


def create_road_graph(
    df_edges: pd.DataFrame,
    df_nodes: pd.DataFrame,
    weight_col: str|None = None
) -> nx.MultiDiGraph:
    """
    Creates a directed graph representation of a road network using NetworkX.

    Args:
        `df_edges` (`pd.DataFrame`): DataFrame containing the edges of the road network. 
            Must include columns `u`, `v`, and `key`.
        `df_nodes` (`pd.DataFrame`): DataFrame containing the nodes of the road network. 
            Must include a column `node_id`.
        `weight_col` (`str`|`None`, optional): Column in `df_edges` to use as edge weights. Defaults to `None`.

    Returns:
        `nx.MultiDiGraph`: A directed graph of the road network with nodes and edges.
    """
    G = nx.MultiDiGraph()
    for idx, row in df_nodes.iterrows():
        G.add_node(row['node_id'])
    for idx, row in df_edges.iterrows():
        if weight_col is None:
            G.add_edge(row['u'], row['v'], key=row['key'])
        else:
            G.add_edge(row['u'], row['v'], key=row['key'], weight=row[weight_col])
    return G


def _process_columns(
    value: str,
    expected_type: str = "int"):
    """
    Process mixed integer values:
    - For string integers (e.g., "60"), returns the integer
    - For string lists (e.g., "['70','90']"), returns the average of the integers
    - For other strings (e.g., "signals"), returns np.nan
    
    Process mixed string values:
    - For signle string (e.g., "motorway"), returns the string
    - For string lists (e.g., "['motorway','trunk']"), returns the first string

    Args:
        value (str or object) : The value to be processed
        expected_type (str) : The expected type of the value, either 'int' or 'str'. Default to 'int'.
    
    Returns:
        Processed value or np.nan for non-numeric values
    """
    if pd.isna(value):
        return np.nan
    
    value_str = str(value)
    
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            parsed_list = ast.literal_eval(value_str)
            if expected_type=="int":
                integers = [int(x) for x in parsed_list]
                return sum(integers) / len(integers)
            elif expected_type=="str":
                return parsed_list[0]
        except (ValueError, SyntaxError, TypeError):
            return np.nan
        
    if expected_type=="int" and value_str.isdigit():
        return int(value_str)
    elif expected_type=="str":
        return value_str    
    
    return np.nan