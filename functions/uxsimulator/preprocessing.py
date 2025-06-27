import geopandas as gpd
import pandas as pd
from constants import CRS_PROJECTED
from spatial_utils import equivalent_circle, boundary_points


def _inside_circles(
    df_od: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Creates circles for OD zones that are mostly within the Area Verde.

    Args:
        `df_od` (`gpd.GeoDataFrame`): GeoDataFrame of OD zones with their geometries and overlap status.

    Returns:
        `gpd.GeoDataFrame`: GeoDataFrame of circles with their attributes (`coord1`, `coord2`, `geometry`, `radius`).
    """
    gdf_circles = df_od.copy()
    gdf_circles = gdf_circles[gdf_circles['mostly_within_area_verde']]
    gdf_circles[['coord1', 'coord2','geometry', 'radius']] = gdf_circles['geometry'].apply(equivalent_circle).apply(pd.Series)
    gdf_circles.set_crs(CRS_PROJECTED, inplace=True)
    return gdf_circles


def _outside_circles(
    df_od: gpd.GeoDataFrame,
    df_bav: gpd.GeoDataFrame,
    outside_radius: float = 3000.0,
    n_outside: int = 8
) -> gpd.GeoDataFrame:
    """
    Creates circles outside the OD zones, based on the surrounding Area Verde.

    Args:
        `df_od` (`gpd.GeoDataFrame`): GeoDataFrame of OD zones.
        `df_bav` (`gpd.GeoDataFrame`): GeoDataFrame of surrounding Area Verde geometries.
        `outside_radius` (`float`): Radius of the outside circles. Default to 3000 (meters).
        `n_outside` (`int`): Number of outside circles to generate. Default to 8.

    Returns:
        `gpd.GeoDataFrame`: GeoDataFrame of circles outside the OD zones with attributes like geometry, coordinates, and radius.
    """

    points_gseries = gpd.GeoSeries([pt for geom in df_bav for pt in boundary_points(geom, n_outside)])
    points_gseries.set_crs(CRS_PROJECTED, inplace=True)

    gdf_circles_outside = gpd.GeoDataFrame({
        'geometry': points_gseries.buffer(outside_radius),
        'coord1': points_gseries.x,
        'coord2': points_gseries.y,
        'radius': outside_radius,
        'id': range(1, n_outside + 1),
        'point_geometry': points_gseries
    }, crs=CRS_PROJECTED)
    gdf_circles_outside['id'] += df_od['id'].max()

    return gdf_circles_outside


def create_OD_circles(
    df_od: gpd.GeoDataFrame,
    df_bav: gpd.GeoDataFrame,
    outside_radius: float = 3000.0,
    n_outside: int = 8
):
    """
    Transforms OD zones into inside and outside circles based on spatial data.

    Args:
        `df_od` (`gpd.GeoDataFrame`): GeoDataFrame of OD zones with their geometries.
        `df_bav` (`gpd.GeoDataFrame`): GeoDataFrame of Area Verde boundaries.
        `outside_radius` (`float`, optional): Radius for outside circles. Defaults to `3000.0`.
        `n_outside` (`int`, optional): Number of outside points to generate. Defaults to `8`.

    Returns:
        `pd.DataFrame`: A concatenated DataFrame of inside and outside circles with selected columns.
    """
    circles_inside = _inside_circles(df_od)
    circles_outside = _outside_circles(df_od, df_bav, outside_radius, n_outside)

    columns = ['id', 'coord1', 'coord2', 'radius', 'geometry']
    return pd.concat([circles_inside[columns], circles_outside[columns]]).reset_index(drop=True) ## REMEMBER THAT THE GEOM IS ADDED


def _map_OD_to_circles(
    od_matrix: pd.DataFrame,
    id_dict: dict, 
    out_ids,
) -> pd.DataFrame:
    """
    Transforms the OD matrix by mapping IDs using a dictionary and filtering unwanted flows.

    Args:
        `od_matrix` (`pd.DataFrame`): OD matrix DataFrame with `from`, `to`, and `flow` columns.
        `id_dict` (`dict`): Dictionary mapping old IDs to new IDs.
        `out_ids` (`list`|`set`): List or set of new IDs outside AV.

    Returns:
        `pd.DataFrame`: Transformed OD matrix.
    """
    od_matrix = (
        od_matrix
        .rename(columns={'from':'from_id', 'to':'to_id', 'flow':'flow'})
        .astype({'from_id': 'int', 'to_id': 'int'})
    )
    od_matrix['from_id'] = od_matrix['from_id'].apply(lambda x: id_dict.get(x,x))
    od_matrix['to_id'] = od_matrix['to_id'].apply(lambda x: id_dict.get(x,x))
    od_matrix = od_matrix.groupby(['from_id', 'to_id']).sum().reset_index()

    od_matrix = od_matrix[((~od_matrix['from_id'].isin(out_ids)) | (~od_matrix['to_id'].isin(out_ids))) & (od_matrix['flow']>0)]

    return od_matrix


def create_demand(
    od_matrix: pd.DataFrame, 
    circle_od_zones: pd.DataFrame,
    id_dict: dict, 
    out_ids,
) -> pd.DataFrame:
    """
    Generates demand between circle OD zones based on the OD matrix.

    Args:
        `od_matrix` (`pd.DataFrame`): DataFrame containing the OD matrix.
        `circle_od_zones` (`pd.DataFrame`): DataFrame of circle zones with attributes like coordinates and radius.
        `id_dict` (`dict`): Dictionary mapping OD zone IDs to circle IDs.
        `out_ids` (`list`|`set`): List or set of IDs to exclude from demand generation.

    Returns:
        `pd.DataFrame`: DataFrame containing demand data with coordinates, radius, and flow.
    """
    od_matrix = _map_OD_to_circles(od_matrix=od_matrix, id_dict=id_dict, out_ids=out_ids)
    return (
        od_matrix
        .merge(circle_od_zones, left_on='from_id', right_on='id', how='left')
        .drop(columns=['id'])
        .rename(columns={
            'coord1': 'coord1_from',
            'coord2': 'coord2_from',
            'radius': 'radius_from'
        })
        .merge(circle_od_zones, left_on='to_id', right_on='id', how='left')
        .drop(columns=['id'])
        .rename(columns={
            'coord1': 'coord1_to',
            'coord2': 'coord2_to',
            'radius': 'radius_to'
        })
        [['coord1_from', 'coord2_from', 'radius_from', 'coord1_to', 'coord2_to', 'radius_to', 'flow']]
    )

