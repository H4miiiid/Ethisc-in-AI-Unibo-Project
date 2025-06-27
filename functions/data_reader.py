import pandas as pd
import geopandas as gpd

from constants import CRS_LATLONG, CRS_PROJECTED, P2V
import data_cleaner


def OD_shapes(
    namefile_polygons: str = "Shape_zone.shp",
    namefile_centers: str|None = None,
    datapath: str = "data"
) -> gpd.GeoDataFrame:
    """
    Reads OD (origin-destination) shapes from a local file.

    Sets its coordinate reference system (CRS) correctly to `CRS_PROJECTED`. The geometry of the shapes is fixed to remove 
    data inconsistencies. If the center file is provided, the shapes are merged with the centroids buffered of a symbolic amount of 1000.

    Args:
        `namefile_polygons (str)`: Name of the file to load the areas. Defaults to `"PROGETTO-AREA.shp"`.
        `namefile_centers (str, optional)`: Name of the file to load the points. Defaults to `None`.
        `datapath (str, optional)`: Directory path where the files are located. Defaults to `"data"`.

    Returns:
        `gpd.GeoDataFrame`: GeoDataFrame containing the processed OD shapes.
    """
    df = (
        gpd.read_file(f"{datapath}/{namefile_polygons}")
        .set_crs("EPSG:23032")
        .to_crs(CRS_PROJECTED)
        .rename(columns={"NO": "id"})
        .astype({"id": int, "TYPENO": int, "NAME": pd.StringDtype()})
    )
    df.columns = df.columns.str.lower()
    #df['geometry'] = df.geometry.buffer(0)
    #df = df.dissolve(by='id').reset_index() #fix the multiple polygons to single multipolygon
    df['type2'] = 'internal'

    if namefile_centers is not None:
        df_centers = OD_centers(namefile_centers=namefile_centers, datapath=datapath)
        df_centers = df_centers[~(df_centers['id'].isin(df['id']))]
        df_centers['geometry'] = df_centers.geometry.buffer(1000)
        df_centers['type2'] = 'external'
        df = pd.concat([df, df_centers], ignore_index=True)
    return df


def OD_centers(
    namefile_centers: str = "Shape_zone_centriod.shp",
    datapath: str = "data"
) -> gpd.GeoDataFrame:
    """
    Reads OD (origin-destination) centroid points from a local file.
    Sets its coordinate reference system (CRS) correctly to `CRS_PROJECTED`. The geometry of the shapes is fixed to remove data inconsistencies.

    Args:
        `namefile_centers (str, optional)`: Name of the file to load the OD centroids. Defaults to `"PROGETTO-CENTER.shp"`.
        `datapath (str, optional)`: Directory path where the file is located. Defaults to `"data"`.

    Returns:
        `gpd.GeoDataFrame`: GeoDataFrame containing the processed OD centroids.
"""
    df_centers = (
        gpd.read_file(f"{datapath}/{namefile_centers}")
        .set_crs("EPSG:23032")
        .to_crs(CRS_PROJECTED)
        .rename(columns={"NO": "id"})
        .astype({"id": int, "TYPENO": int, "NAME": pd.StringDtype()})
    )
    df_centers.columns = df_centers.columns.str.lower()
    return df_centers


def AV_shape(
    namefile: str = "area_verde_manual_v1.geojson",
    datapath: str = "data"
) -> gpd.GeoDataFrame:
    """
    Reads the Area Verde (AV) shape from a local file and sets its coordinate reference system (CRS) correctly to `CRS_PROJECTED`.

    Args:
        `namefile (str)`: Name of the file to load. Defaults to `"area_verde_manual_v1.geojson"`.
        `datapath (str)`: Directory path where the file is located. Defaults to `"data"`.

    Returns:
        `gpd.GeoDataFrame`: GeoDataFrame containing the processed AV shape.
    """
    df = (
        gpd.read_file(f"{datapath}/{namefile}")
        .set_crs(CRS_LATLONG)
        .to_crs(CRS_PROJECTED)
    )
    return df


def AOI_shapes(
    namefile: str,
    datapath: str = "data",
    aoi_type: str = "od",
    df_around: gpd.GeoDataFrame = None
) -> gpd.GeoDataFrame:
    """
    Loads shapes of Areas of Interest (AOI) based on the specified type: `"od"` for OD zones, `"census"` for statistical areas, or `"av"` for the entire AreaVerde.
    The coordinate reference system (CRS) is set to `CRS_PROJECTED`.

    Args:
        `namefile (str)`: Name of the file to load.
        `datapath (str)`: Directory path where the file is located. Defaults to `"data"`.
        `aoi_type (str)`: Type of AOI (`"od"`, `"census"`, or `"av"`). Defaults to `"od"`.
        `df_around (gpd.GeoDataFrame, optional)`: AV GeoDataFrame for intersection. If provided, the AOI shapes are cropped with the AV. Defaults to `None`.

    Returns:
        `gpd.GeoDataFrame: GeoDataFrame containing the loaded AOI shapes.
    """
    if aoi_type == "od":
        df = OD_shapes(namefile_polygons=namefile, datapath=datapath)[['id', 'geometry']]
        df = data_cleaner.AOI_OD_shapes(df).rename({'id': 'id_zone'}, axis=1)
    elif aoi_type == "census":
        df = (
            gpd.read_file(f"{datapath}/{namefile}") 
            .set_crs(CRS_LATLONG)
            .to_crs(CRS_PROJECTED)
            .dissolve(by='zona')
            .reset_index()
            .rename({'zona': 'id_zone'}, axis=1)
            [['id_zone', 'geometry']]
            .drop_duplicates()
        )
    elif aoi_type == "av":
        df = AV_shape(namefile, datapath)
        df['id_zone'] = "0"
    
    df['id_zone'] = df['id_zone'].astype(str)
    if df_around is not None and not df_around.empty:
        df = gpd.overlay(df, df_around, how='intersection')

    return df


def OD_flows(
    namefile: str = "PROGETTO-OD.xlsx",
    datapath: str = "data",
    df_shapes: pd.DataFrame|None = None
) -> pd.DataFrame:
    """
    Loads OD matrix of traffic flows [unit: people]. Filters flows contained from/to selected shapes, if a `df_shapes` DataFrame is provided.

    Args:
        `namefile (str)`: Name of the file to load. Defaults to "PROGETTO-OD.xlsx".
        `datapath (str)`: Directory path where the file is located. Defaults to "data".
        `df_shapes (pd.DataFrame, optional)`: DataFrame for filtering flows. Defaults to None.

    Returns:
        `pd.DataFrame`: DataFrame containing the OD matrix flows.
    """
    df_od = (
        pd.read_excel(f"{datapath}/{namefile}")
        .rename(columns={'NumDaZona': 'from', 'NumAZona': 'to', 'ValMatr(1 Totale_H24)': 'flow'})
        # .assign(flow = lambda x: x['flow'] * P2V)  #TODO REMOVE
        [['from', 'to', 'flow']]
    )

    if df_shapes is None or df_shapes.empty:
        return df_od
    else:
        return data_cleaner.OD_flows(df_od, df_shapes)


def road_data(
    edges_namefile: str = "geo_edges_v4.geojson",
    nodes_namefile: str = "geo_nodes_v4.geojson",
    datapath: str = "data/results"
) -> list:
    """
    Loads road network data as a DataFrame of edges and a DataFrame of nodes.
 
    Args:
        `edges_namefile (str)`: Name of the edges file. Defaults to `"geo_edges_v4.geojson"`.
        `nodes_namefile (str)`: Name of the nodes file. Defaults to `"geo_nodes_v4.geojson"`.
        `datapath (str)`: Directory path where the files are located. Defaults to `"data/results"`.

    Returns:
        `list`: List containing two GeoDataFrames - edges and nodes.
    """
    edges = gpd.read_file(f"{datapath}/{edges_namefile}")
    nodes = gpd.read_file(f"{datapath}/{nodes_namefile}")
    return [edges, nodes]


def spira_shapes(
    namefile: str = "SpiraFlowData.parquet",
    datapath: str = "data",
    project = None,
    df_around : gpd.GeoDataFrame|None = None
) -> gpd.GeoDataFrame:
    """
    Reads Spira shapes either from a data file, if present, or downloading it from the platform. If `df_around` is provided, shapes are cropped to intersect it.

    Args:
        `namefile (str)`: Name of the file to load. Defaults to `"SpiraFlowData.parquet"`.
        `datapath (str)`: Directory path where the file is located. Defaults to `"data"`.
        `project (optional)`: Project instance for data download. Defaults to `None`.
        `df_around (gpd.GeoDataFrame, optional)`: GeoDataFrame for filtering shapes. Defaults to `None`.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing Spira shapes.
    """
    if project is None:
        spira = pd.read_parquet(f"{datapath}/{namefile}")
    else:
        spira = project.get_dataitem("spira_flow_data_2024")
        spira.download(f"{datapath}/{namefile}", overwrite=True)
        spira = pd.read_parquet(f"{datapath}/{namefile}", engine="pyarrow")
    
    spira = (
        spira[['spira_unique_id','longitudine', 'latitudine']]
        .drop_duplicates()
        .groupby('spira_unique_id')
        .first()
        .reset_index()   
    )
    spira = gpd.GeoDataFrame(spira, geometry=gpd.points_from_xy(spira.longitudine, spira.latitudine), crs=CRS_LATLONG).to_crs(CRS_PROJECTED)

    if df_around is not None and not df_around.empty:
        spira = gpd.overlay(
            spira.set_crs(CRS_PROJECTED), 
            df_around.set_crs(CRS_PROJECTED), 
            how='intersection'
        )
    
    return spira[['spira_unique_id', 'longitudine', 'latitudine', 'geometry']]


def spira_codes(
    namefile: str = "SpiraFlowData.parquet",
    datapath: str = "data",
    project = None
) -> gpd.GeoDataFrame:
    """
    Reads Spira codes either from a local file, if present, or downloading it from the platform.

    Args:
        `namefile (str)`: Name of the file to load. Defaults to `"SpiraFlowData.parquet"`.
        `datapath (str)`: Directory path where the file is located. Defaults to `"data"`.
        `project (optional)`: Project instance for data download. Defaults to `None`.

    Returns:
        `gpd.GeoDataFrame`: DataFrame with Spira unique IDs `spira_unique_id` and sensor codes `spira_code`.
    """
    if project is None:
        spira = pd.read_parquet(f"{datapath}/{namefile}")
    else:
        spira = project.get_dataitem("spira_flow_data_2024")
        spira.download(f"{datapath}/{namefile}", overwrite=True)
        spira = pd.read_parquet(f"{datapath}/{namefile}", engine="pyarrow")

    return spira[['spira_unique_id', 'spira_code']].drop_duplicates()


def spira_flows(
    namefile: str = "SpiraFlowData5m.parquet",
    datapath: str = "data",
    project = None,
    filters: list|None = None,
    clean: bool = False
)-> pd.DataFrame:
    """
    Reads Spira flows data from a local file, if present, or downloading it from the platform. Filters can be applied to the reading by specifying the argument `filters`.

    Args:
        `namefile (str)`: Name of the file to load. Defaults to `"SpiraFlowData5m.parquet"`.
        `datapath (str)`: Directory path where the file is located. Defaults to `"data"`.
        `project (optional)`: Project instance for data download. Defaults to `None`.
        `filters (list, optional)`: List of filtering conditions for reading. Defaults to `None`.
        `clean` (`bool`): whether to apply a cleaning filter to the data or not.

    Returns:
        `pd.DataFrame`: DataFrame with Spira sensor flow data.
    """
    if project is None:
        spira = pd.read_parquet(f"{datapath}/{namefile}", filters=filters, engine="pyarrow")
    else:
        spira = project.get_dataitem("spire_flow5m_2024")
        spira.download(f"{datapath}/{namefile}", overwrite=True)
        spira = pd.read_parquet(f"{datapath}/{namefile}", filters=filters, engine="pyarrow")

    spira = spira[['sensor_id', 'value', 'start']].rename(columns={'value': 'count', 'start': 'DateTime'})
    spira['DateTime'] = pd.to_datetime(spira['DateTime'], format='%Y-%m-%d %H:%M')
    spira = spira.sort_values('DateTime')

    if clean:
        spira = data_cleaner.spira_flows(df_spiras=spira)

    return spira


def spira_accuracy(
    namefile: str = "SpiraAccuracy.parquet",
    datapath: str = "data",
    project = None,
    filters: list|None = None
)-> pd.DataFrame:
    """
    Reads Spira accuracy data from a local file, if present, or downloading it from the platform. Filters can be applied to the reading by specifying the argument `filters`.

    Args:
        `namefile (str)`: Name of the file to load. Defaults to `"SpiraAccuracy.parquet"`.
        `datapath (str)`: Directory path where the file is located. Defaults to `"data"`.
        `project (optional)`: Project instance for data download. Defaults to `None`.
        `filters (list, optional)`: List of filtering conditions for reading. Defaults to `None`.
        `clean` (`bool`): whether to apply a cleaning filter to the data or not.

    Returns:
        `pd.DataFrame`: DataFrame with Spira sensor accuracy data.
    """
    if project is None:
        spira = pd.read_parquet(f"{datapath}/{namefile}", filters=filters, engine="pyarrow")
    else:
        spira = project.get_dataitem("spira_accur_data_2024")
        spira.download(f"{datapath}/{namefile}", overwrite=True)
        spira = pd.read_parquet(f"{datapath}/{namefile}", filters=filters, engine="pyarrow")
        print
    spira['DateTime'] = pd.to_datetime(spira['DateTime'])
    return spira