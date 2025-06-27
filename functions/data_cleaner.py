import pandas as pd
import geopandas as gpd


def OD_flows(
    df_flows: pd.DataFrame,
    df_shapes: pd.DataFrame
) -> pd.DataFrame:
    """
    Filters the OD flows by merging them with shape data and removing mismatched flows.

    Args:
        `df_flows (pd.DataFrame)`: DataFrame containing the OD flows.
        `df_shapes (pd.DataFrame)`: DataFrame containing shape information.

    Returns:
        `pd.DataFrame`: Processed DataFrame with updated OD flows.
    """
    df_flows = df_flows.merge(
            df_shapes.rename({'id': 'from'}, axis=1),
            on="from",
            how='inner', # if an area has no geometry, then its flow is removed
        ).merge(
            df_shapes.rename({'id': 'to'}, axis=1),
            on="to",
            how='inner', # if an area has no geometry, then its flow is removed
        )
    return df_flows


def AOI_OD_shapes(
    df_shapes: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    When considering OD zones as AOI, it is important to remove the overlapping zones,
    that were added to better modeling the Visum simulation, but that have not a physical
    meaning. The function identifies the overlapping zones and removes them as not AOI.

    Args:
        `df_shapes (pd.DataFrame)`: DataFrame containing shape information.

    Returns:
        `gpd.GeoDataFrame`: Processed DataFrame with updated shape information.
    """
    dictionary = dict()
    df_shapes = df_shapes.set_index('id')

    for this_id in df_shapes.index:
        new_1zone = df_shapes[df_shapes.index==this_id]

        inters = df_shapes.apply(lambda x: new_1zone.geometry.iloc[0].contains(x.geometry), axis=1).reset_index(name='overlap')
        idz_overlaps = inters.loc[inters['overlap'], 'id']
        idz_overlaps = idz_overlaps[idz_overlaps!=this_id]
        
        if len(idz_overlaps)>0:
            dictionary[this_id] = idz_overlaps.tolist()

    idx_remove = [i for c in [val for val in dictionary.values()] for i in c]
    df_shapes = df_shapes.reset_index()

    return df_shapes[~df_shapes['id'].isin(idx_remove)]


def spira_flows(
    df_spiras: pd.DataFrame
)-> pd.DataFrame:
    """
    Removes all records of days where the count was always 0 for the entire day.

    Args:
        `df_spiras` (`pd.DataFrame`): The input DataFrame containing 'spira_unique_id', 'DateTime', and 'count' columns.

    Returns:
        `pd.DataFrame`: A new DataFrame with the records of such days removed.
    """
    df_spiras = df_spiras.assign(Date=lambda x: x['DateTime'].dt.date)
    grouped = df_spiras.groupby(['spira_unique_id', 'Date'])

    all_zero_days = grouped['count'].apply(lambda x: (x == 0).all())
    days_to_drop = all_zero_days[all_zero_days].index

    records_to_keep = ~df_spiras.set_index(['spira_unique_id', 'Date']).index.isin(days_to_drop)
    cleaned_dataframe = df_spiras[records_to_keep].reset_index(drop=True)

    return cleaned_dataframe
