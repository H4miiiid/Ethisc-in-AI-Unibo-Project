import pandas as pd
import numpy as np
import datetime as dt
from scipy.signal import savgol_filter
from scipy.interpolate import make_smoothing_spline

import temporal_utils
from io_utils import vprint


def _per_road_deltaTime(
    spira_data: pd.DataFrame,
    df_catch: pd.DataFrame
):
    spira_data = spira_data.groupby(['spira_unique_id']) \
        .sum(numeric_only=True) \
        .reset_index()

    counts_roads = pd.merge(df_catch, spira_data, on='spira_unique_id', how='inner')
    counts_roads['count_distributed'] = counts_roads['count'] * counts_roads['prop_t_imp'] 

    counts_roads = counts_roads \
        .drop(['prop_t_imp', 'count', 'spira_unique_id'], axis=1) \
        .groupby(['u', 'v', 'key','id_zone']) \
        .median(numeric_only=True)\
        .reset_index()
    
    return counts_roads
    

def _per_area_deltaTime(
    spira_data: pd.DataFrame,
    df_catch: pd.DataFrame
):
    counts_roads = _per_road_deltaTime(spira_data=spira_data, df_catch=df_catch)
    
    counts_areas = counts_roads \
        .drop(['u','v','key'], axis=1) \
        .groupby(['id_zone']) \
        .sum(numeric_only=True) \
        .reset_index()

    return counts_areas


def per_road(
    spira_data: pd.DataFrame,
    df_catch: pd.DataFrame,
    first_datetime: dt.datetime,
    last_datetime: dt.datetime,
    deltaTime: int = 5,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Computes distributed traffic counts per road for specified time intervals.

    Args:
        `spira_data` (`pd.DataFrame`): DataFrame containing spira traffic data with counts and timestamps.
        `df_catch` (`pd.DataFrame`): DataFrame containing spira catchment area data with time-proportional weights.
        `first_datetime` (`datetime`): The start of the time range for traffic computation.
        `last_datetime` (`datetime`): The end of the time range for traffic computation.
        `deltaTime` (`int`, optional): Duration of each time interval in minutes. Defaults to `5` minutes.

    Returns:
        `pd.DataFrame`: DataFrame containing distributed traffic counts per road over time intervals.
    """
    traffic = pd.DataFrame()
    last_datetime = last_datetime + dt.timedelta(minutes=deltaTime)
    this_datetime = first_datetime
    
    while this_datetime < last_datetime:
        next_datetime = this_datetime + dt.timedelta(minutes=deltaTime)
        spira_data_red = spira_data[spira_data['DateTime'].between(this_datetime, next_datetime, inclusive='left')] 
        spira_data_red = spira_data_red[['spira_unique_id', 'count']]

        traffic_road = _per_road_deltaTime(spira_data=spira_data_red, df_catch=df_catch)
        traffic_road['DateTime'] = this_datetime

        traffic = pd.concat([traffic, traffic_road], ignore_index=True)

        if this_datetime.day != next_datetime.day:
            vprint(text=f"Traffic in day {this_datetime.day} computed", verbose=verbose)
        
        this_datetime = this_datetime + dt.timedelta(minutes=deltaTime)
    return traffic


def per_area(
    spira_data: pd.DataFrame,
    df_catch: pd.DataFrame,
    first_datetime: dt.datetime,
    last_datetime: dt.datetime,
    deltaTime: int = 5,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Computes aggregated traffic counts per area for specified time intervals.

    Args:
        `spira_data` (`pd.DataFrame`): DataFrame containing spira traffic data with counts and timestamps.
        `df_catch` (`pd.DataFrame`): DataFrame containing spira catchment area data with time-proportional weights.
        `first_datetime` (`datetime`): The start of the time range for traffic computation.
        `last_datetime` (`datetime`): The end of the time range for traffic computation.
        `deltaTime` (`int`, optional): Duration of each time interval in minutes. Defaults to `5` minutes.

    Returns:
        `pd.DataFrame`: DataFrame containing aggregated traffic counts per area over time intervals.
    """
    traffic = pd.DataFrame()
    last_datetime = last_datetime + dt.timedelta(minutes=deltaTime)
    this_datetime = first_datetime
    
    vprint(text="Starting the traffic computation", verbose=verbose)
    while this_datetime < last_datetime:
        next_datetime = this_datetime + dt.timedelta(minutes=deltaTime)
        spira_data_red = spira_data[spira_data['DateTime'].between(this_datetime, next_datetime, inclusive='left')] 
        spira_data_red = spira_data_red[['spira_unique_id', 'count']]

        traffic_area = _per_area_deltaTime(spira_data=spira_data_red, df_catch=df_catch)
        traffic_area['DateTime'] = this_datetime

        traffic = pd.concat([traffic, traffic_area], ignore_index=True)

        if this_datetime.day != next_datetime.day:
            vprint(text=f"Traffic in day {this_datetime.day}/{this_datetime.month}/{this_datetime.year} computed", verbose=verbose)

        this_datetime = this_datetime + dt.timedelta(minutes=deltaTime)
    return traffic


def average(
    dataset: pd.DataFrame, 
    use_daytype: bool = False, 
    holiday_namefile: str|None = None,
    use_zonegroup: bool = False
) -> pd.DataFrame:
    """
    Calculates the average traffic index in 5-minute intervals, optionally grouping by day type and zone group.

    Args:
        `dataset` (`pd.DataFrame`): DataFrame containing traffic data with columns `DateTime` and `traffic_index`.
        `use_daytype` (`bool`, optional): Whether to group by day type (e.g., weekday/weekend). Defaults to `False`.
        `use_zonegroup` (`bool`, optional): Whether to group by zone. Defaults to `False`.

    Returns:
        `pd.DataFrame`: DataFrame with average traffic index grouped by time, day type, and zone.
    """
    dataset['DateTime'] = pd.to_datetime(dataset['DateTime'])
    dataset['Time'] = dataset['DateTime'].dt.time
    
    if use_daytype:
        if holiday_namefile is None:
            print('No holiday list provided. Returning only Sundays as holidays.')
            holiday_list = []
        else: 
            holiday_list = pd.read_csv(holiday_namefile, header=None)[0].to_list()
        dataset['DayType'] = dataset.apply(lambda x: temporal_utils.add_daytype(date=x['DateTime'], holiday_list=holiday_list), axis=1)
    else:
        dataset['DayType'] = 'All'
    if not use_zonegroup:
        dataset['id_zone'] = 'All'
    traffic = dataset.groupby(['Time', 'DayType', 'id_zone'])['traffic_index'].mean().reset_index()        

    return traffic


def smoothing (
    dataset: pd.DataFrame, 
    use_daytype: bool = False, 
    use_zonegroup: bool = False,
    method: str = "savgol",
) -> pd.DataFrame:
    """
    Smoothens the traffic data using specified methods, optionally grouping by day type and zone group.

    Args:
        `dataset` (`pd.DataFrame`): DataFrame containing traffic data with a `traffic_index` column.
        `use_daytype` (`bool`, optional): Whether to group by day type before smoothing. Defaults to `False`.
        `use_zonegroup` (`bool`, optional): Whether to group by zone before smoothing. Defaults to `False`.
        `method` (`str`, optional): Smoothing method to apply. Options include `'splines'` and `'savgol'`. Defaults to `'savgol'`.

    Returns:
        `pd.DataFrame`: DataFrame with a new column `smooth_traffic` containing the smoothed traffic index.
    """
    def smoothing_function(noisy, method=['splines', 'savgol']):
        if method=='splines':
            xx = np.array(range(len(noisy)))
            spl = make_smoothing_spline(
                x=xx, 
                y=np.array(noisy), 
                lam=50)
            denoised = spl(xx)
        else:
            denoised = savgol_filter(noisy, 12 * 2 + 1, 2)  #window of 65min
        return denoised

    dataset = dataset.sort_values(['id_zone', 'DayType', 'Time'])
    group_vars = ['DayType'] if use_daytype else []
    group_vars += ['id_zone'] if use_zonegroup else []

    if group_vars:
        dataset['smooth_traffic'] = dataset.groupby(group_vars)['traffic_index'].transform(lambda x: smoothing_function(x, method))
    else:
        dataset['smooth_traffic'] = smoothing_function(dataset['traffic_index'], method)
    return dataset

