import folium
import geojson
import webbrowser
import pandas as pd
import geopandas as gpd

# Carica il file GeoJSON
gdf_area_verde = gpd.read_file("/Users/venere/Documents/GitHub/AreaVerde/Simulation/area_verde_manual_v1.geojson")
aree_gdf_inside = gpd.read_file("/Users/venere/Documents/GitHub/AreaVerde/Simulation/aree_gdf_inside_v2.geojson")
aree_gdf_outside = gpd.read_file("/Users/venere/Documents/GitHub/AreaVerde/Simulation/aree_gdf_outside_v2.geojson")
aree_gdf = pd.concat([aree_gdf_inside, aree_gdf_outside], axis=0)
aree_gdf_AV = gpd.overlay(aree_gdf, gdf_area_verde, how='intersection')
aree_statistiche = gpd.read_file("/Users/venere/Documents/GitHub/AreaVerde/Simulation/aree-statistiche.geojson")
aree_e_areastatistica_gdf_AV = gpd.overlay(aree_gdf_AV, aree_statistiche, how='intersection')
df_mappatura = aree_e_areastatistica_gdf_AV.copy()
aree_e_areastatistica_gdf_AV.explore(column='name', legend=True)
df_mappatura.to_csv('df_mappatura.csv')
