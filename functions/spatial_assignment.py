import geopandas as gpd
from constants import CRS_PROJECTED
from spatial_utils import is_left, nearest_lines, closest_point


def OD_to_AV(
    df_od : gpd.GeoDataFrame, 
    df_av : gpd.GeoDataFrame, 
    overlap_threshold: float = 0.5
) -> gpd.GeoDataFrame:
    """
    Links the OD zones with the Area Verde (AV) by calculating the overlap and discovering which are mostly inside.

    Args:
        `df_od` (`gpd.GeoDataFrame`): GeoDataFrame of OD zones with their geometries.
        `df_av` (`gpd.GeoDataFrame`): GeoDataFrame of Area Verde geometries.
        `overlap_threshold` (`float`): Minimum overlap ratio required for linking. Defaults to `0.5`.

    Returns:
        `gpd.GeoDataFrame`: The updated OD GeoDataFrame with a new column `mostly_within_area_verde` indicating overlap.
    """
    df_od["area"] = df_od.area
    areas_intersects = (
        df_od
        .reset_index(drop=True)
        .overlay(df_av[["geometry"]], how="intersection")
    )
    ratio_overlap = areas_intersects.area / areas_intersects['area']

    id_ok = areas_intersects[ratio_overlap > overlap_threshold]['id'].values
    df_od['mostly_within_area_verde'] = False
    df_od.loc[df_od['id'].isin(id_ok), 'mostly_within_area_verde'] = True

    return df_od


def AOI_to_AV(
    df_od : gpd.GeoDataFrame, 
    df_av : gpd.GeoDataFrame, 
    overlap_threshold: float = 0.5
) -> gpd.GeoDataFrame:
    """
    Links the AOI zones with the Area Verde (AV) by calculating the overlap and discovering which are mostly inside.

    Args:
        `df_od` (`gpd.GeoDataFrame`): GeoDataFrame of AOI zones with their geometries.
        `df_av` (`gpd.GeoDataFrame`): GeoDataFrame of Area Verde geometries.
        `overlap_threshold` (`float`): Minimum overlap ratio required for linking. Defaults to `0.5`.

    Returns:
        `gpd.GeoDataFrame`: The updated OD GeoDataFrame with a new column `mostly_within_area_verde` indicating overlap.
    """
    df_od["area"] = df_od.area
    areas_intersects = (
        df_od
        .reset_index(drop=True)
        .overlay(df_av[["geometry"]], how="intersection")
    )
    ratio_overlap = areas_intersects.area / areas_intersects['area']

    id_ok = areas_intersects[ratio_overlap > overlap_threshold]['id_zone'].values
    df_od['mostly_within_area_verde'] = False
    df_od.loc[df_od['id_zone'].isin(id_ok), 'mostly_within_area_verde'] = True

    return df_od


def roads_to_AOI(
    df_edges: gpd.GeoDataFrame, 
    df_aoi: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Assigns roads (edges) to Areas of Interest (AOI) by checking intersections.

    Args:
        `df_edges` (`gpd.GeoDataFrame`): GeoDataFrame of road network edges.
        `df_aoi` (`gpd.GeoDataFrame`): GeoDataFrame of AOI geometries with `id_zone` attributes.

    Returns:
        `gpd.GeoDataFrame`: Updated GeoDataFrame of edges with AOI zone assignments.
    """
    df_edges.set_crs(CRS_PROJECTED, inplace=True)
    joined = gpd.sjoin(df_edges, df_aoi, how="left", predicate="intersects")
    joined['id_zone'] = joined['id_zone'].fillna('0')
    joined = joined.merge(df_aoi[['id_zone', 'geometry']], on='id_zone', how='left')
    joined["intersection_length"] = joined.apply(
        lambda row: row["geometry_x"].intersection(row["geometry_y"]).length if row["geometry_x"].intersects(row["geometry_y"]) else 0,
        axis=1
    )
    joined_sorted = joined.sort_values(by="intersection_length", ascending=False)
    unique_matches = joined_sorted.drop_duplicates(subset=["geometry_x"], keep="first")

    df_edges = df_edges.merge(unique_matches[['u','v','key','id_zone']], on=['u','v','key'])

    return df_edges


def nodes_to_AV(
    df_nodes: gpd.GeoDataFrame, 
    df_av: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Assigns nodes to the Area Verde (AV) by spatial intersection.

    Args:
        `df_nodes` (`gpd.GeoDataFrame`): GeoDataFrame of road network nodes.
        `df_av` (`gpd.GeoDataFrame`): GeoDataFrame of Area Verde geometries.

    Returns:
        `gpd.GeoDataFrame`: Updated GeoDataFrame of nodes with their AV position (`INSIDE`, `OUTSIDE`).
    """
    df_nodes = gpd.sjoin(df_nodes, df_av[['geometry']], how="left", predicate="intersects")

    df_nodes['index_right'] = df_nodes['index_right'].fillna('OUTSIDE')
    df_nodes.loc[df_nodes['index_right'] == 0, 'index_right'] = 'INSIDE'
    df_nodes.rename(columns={'index_right': 'AV_position'}, inplace=True)

    return df_nodes


def spira_to_road(
    df_spira: gpd.GeoDataFrame,
    df_edge: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Assigns spira sensors to the closest road segments.

    Args:
        `df_spira` (`gpd.GeoDataFrame`): GeoDataFrame of spira sensor locations.
        `df_edge` (`gpd.GeoDataFrame`): GeoDataFrame of road network edges.

    Returns:
        `gpd.GeoDataFrame`: GeoDataFrame of spira sensors with road assignments.
    """
    #Find the closest road to each spira
    spira_close_roads = nearest_lines(df_spira, df_edge)
    spira_close_roads = spira_close_roads.merge(
        df_edge[['u', 'v', 'key', 'geometry', 'oneway', 'highway_ok', 'id_zone']], 
        on=['u','v','key']
    )
    # Apply the function to understand if the spira is on the left of the road
    pt_series = gpd.GeoSeries(spira_close_roads['geometry_x'])
    ln_series = gpd.GeoSeries(spira_close_roads['geometry_y'])
    spira_close_roads['spira_left_location'] = [is_left(linestring=ln, point=pt) for ln, pt in zip(ln_series, pt_series)]
    # Assign the real road
    spira_close_roads = spira_close_roads[
        (spira_close_roads['oneway']==True) | 
        ((spira_close_roads['oneway']==False) & (spira_close_roads['spira_left_location']==False) )
    ]
    spira_close_roads = (
        spira_close_roads
        .drop_duplicates(subset='spira_unique_id')
        .reset_index(drop=True)
        .drop(columns=['spira_left_location'])
    )
    return spira_close_roads


def OD_to_circles(
    df_od: gpd.GeoDataFrame,
    df_circles: gpd.GeoDataFrame
) -> dict:
    """
    Assigns OD zones to the closest circles (inside or outside).

    Args:
        `df_od` (`gpd.GeoDataFrame`): GeoDataFrame of OD zones.
        `df_circles` (`gpd.GeoDataFrame`): GeoDataFrame of circles.

    Returns:
        `dict`: A dictionary mapping OD zone IDs to circle IDs.
    """
    out_id_series = df_od['geometry'].apply(
        lambda x: closest_point(
            x, 
            df_circles[['id']].assign(centroid=df_circles.geometry.centroid).set_geometry('centroid').set_crs(CRS_PROJECTED)
        )
    )
    out_id_series[df_od['mostly_within_area_verde']] = df_od.loc[df_od['mostly_within_area_verde'], 'id']

    id_to_out_id_dict = dict(zip(df_od['id'], out_id_series))
    return id_to_out_id_dict
