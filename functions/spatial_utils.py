import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, LineString, Point
import math


def buffer_around(
    gdf : gpd.GeoDataFrame,
    buffer_size : float = 3000.0
):
    """
    Creates a buffer around the geometries in the provided GeoDataFrame.

    Args:
        `gdf` (`gpd.GeoDataFrame`): GeoDataFrame containing geometries to buffer.
        `buffer_size` (`float`): Size of the buffer in map units (e.g., meters). Defaults to `3000.0`.

    Returns:
        : A GeoSeries with buffered geometries.
    """
    return gdf.buffer(buffer_size)


def equivalent_circle(
    polygon
):
    """
    Given a geolocated Polygon, returns a circle with the same centroid and the same area.

    Args:
        `polygon` XXX

    Returns:
        XXXX
    """
    centroid = polygon.centroid
    x = centroid.coords[0][0]
    y = centroid.coords[0][1]
    radius = (polygon.area / 3.14159) ** 0.5  # Radius from area of circle: A = πr²
    return x, y, centroid.buffer(radius), radius


def boundary_points(
    geom, 
    n: int = 8
):
    """
    Given a geolocated Polygon, returns the n equally-spaced Points in its exterior border

    Args:
        `geom`: The Polygon geometry.
        `n` (`int`): The number of Points to be sampled. Default to 8.

    Returns:
        XXXXXX
    """
    return [geom.exterior.interpolate(d) for d in np.linspace(0, geom.exterior.length, n, endpoint=False)]


def closest_point(
    polygon: Polygon, 
    points_gdf: gpd.GeoDataFrame
):
    """
    Finds the ID of the closest point in a GeoDataFrame to a given polygon.    
    Calculates the distance from each point in the GeoDataFrame to the input polygon
    and returns the ID of the point with the minimum distance.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon geometry to measure distances from.
        points_gdf (gpd.GeoDataFrame): GeoDataFrame containing point geometries and an 'id' column for identifiers.
        
    Returns:
        The ID of the closest point to the polygon.
    """
    distances = points_gdf.distance(polygon)
    min_distance_index = distances.idxmin()
    closest = points_gdf.iloc[min_distance_index]['id']
    return closest


def nearest_lines(
    pts : gpd.GeoDataFrame, 
    lns : gpd.GeoDataFrame
) -> pd.DataFrame:  
    """
    Finds the closest linestring(s) in a GeoDataFrame for each point in another GeoDataFrame.    
    For each point, identifies all linestrings at the minimum distance and creates paired records
    containing both the point data and the matching linestring data.
    
    Args:
        `pts` (`gpd.GeoDataFrame`): GeoDataFrame containing point geometries.
        `lns` (`gpd.GeoDataFrame`): GeoDataFrame containing linestring geometries
                                           with 'u', 'v', and 'key' columns.
    
    Returns:
        `pd.DataFrame`: DataFrame containing combined data from points and their
                      matching nearest linestrings. Each row represents a point-linestring pair,
                      with multiple rows possible for a single point if multiple linestrings
                      are at the minimum distance.
    """  
    full_closest_linestrings = []
    for idx, point in pts.iterrows():
        distances = lns.geometry.distance(point.geometry)
        min_distance = distances.min()
        min_distance_indices = distances[distances == min_distance].index.tolist()

        for closest_index in min_distance_indices:
            closest_linestring = lns[['u','v','key']].loc[closest_index]
            combined_row = pd.concat([point, closest_linestring])
            full_closest_linestrings.append(combined_row)

    return pd.DataFrame(full_closest_linestrings)


def is_left(
    linestring: LineString, 
    point: Point
) -> bool:    
    """
    Determines if a point is to the left of the closest segment of a linestring.

    This function finds the closest segment of the linestring to the given point,
    then uses coordinate translation and rotation to determine if the point lies
    to the left of that segment when looking from the first vertex toward the second.

    The algorithm works by:
    1. Finding the closest segment of the linestring to the point
    2. Translating the coordinate system to make the first vertex the origin
    3. Rotating the coordinate system to align the segment with the x-axis
    4. Checking if the transformed point's y-coordinate is positive (left side)

    Args:
        linestring (LineString): A shapely LineString object to test against.
        point (Point): A shapely Point object to test.
        
    Returns:
        bool: True if the point is to the left of the closest segment,
                False if the point is to the right or on the segment.
                None if no closest segment is found (should not occur).
    """
    def translate_point(p, p0):
        p[0] = p[0] - p0[0]
        p[1] = p[1] - p0[1]
        return p
    def rotate_point(p, angle):
        p = [p[0]*math.cos(angle)-p[1]*math.sin(angle),
          p[0]*math.sin(angle)+p[1]*math.cos(angle) ]
        return p
    def convert_angle (angle, yy):
        if yy < 0:
            angle = - angle
        return angle

    dist = float('inf')
    closest_segment_index = -1
    for i in range(len(linestring.coords) - 1):
        segment = LineString([linestring.coords[i], linestring.coords[i + 1]])
        dist_temp = point.distance(segment)
        if dist_temp < dist:
            dist = dist_temp
            closest_segment_index = i
    if closest_segment_index == -1:
        return None  # This should not happen

    p1 = linestring.coords[closest_segment_index]
    p2 = linestring.coords[closest_segment_index + 1]

    p2 = translate_point([p2[0],p2[1]], [p1[0],p1[1]])
    pp = translate_point([point.x,point.y],  [p1[0],p1[1]])

    angle_p1p2 = math.acos(p2[0] / math.sqrt(p2[1]**2 + p2[0]**2))
    angle_p1p2 = convert_angle(angle_p1p2, yy=p2[1])      

    angle_p1p2 = -angle_p1p2
    p2 = rotate_point(p2, angle_p1p2)
    pp = rotate_point(pp, angle_p1p2)
    
    angle_p1point = math.acos((pp[0]) / math.sqrt(pp[1]**2 + pp[0]**2))
    angle_p1point = convert_angle(angle_p1point, yy=pp[1])

    left_condition = angle_p1point > 0
    return left_condition  # True if the point is to the left of the segment, False otherwise


def direction_wrt_polygon(
    line: gpd.GeoDataFrame,
    polygon: Polygon
) -> str:
    """
    Determines the direction of a line with respect to a polygon.
    
    This function evaluates whether a line is 'incoming' to or 'outgoing' from a polygon
    based on the positions of its start and end points relative to the polygon.
    
    Args:
        row (gpd.GeoDataFrame) : A row from a GeoDataFrame containing a LineString geometry.
        polygon (shapely.Polygon) : A polygon geometry object to check against.
        
    Returns:
        str : 'incoming' if the line is moving toward the polygon, or 'outgoing' if the line is moving away from the polygon.
    """
    start_point = line.coords[0]
    end_point = line.coords[-1]
    
    start_point_geom = Point(start_point)
    end_point_geom = Point(end_point)
    
    start_inside = polygon.contains(start_point_geom)
    end_inside = polygon.contains(end_point_geom)
    
    if start_inside and not end_inside:
        return 'outgoing'
    elif not start_inside and end_inside:
        return 'incoming'
    elif start_inside and end_inside:
        distance_start = start_point_geom.distance(polygon.exterior)
        distance_end = end_point_geom.distance(polygon.exterior)
        return 'incoming' if distance_start < distance_end else 'outgoing'
    else:
        distance_start = start_point_geom.distance(polygon)
        distance_end = end_point_geom.distance(polygon)
        return 'outgoing' if distance_start < distance_end else 'incoming'