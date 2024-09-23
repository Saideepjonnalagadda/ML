# Creating a world map with countries colored by population
import geopandas as gpd
import folium

shapefilepaths = 'C:\\Users\\saide\\Downloads\\110m_cultural\\ne_110m_admin_0_countries.shp'
world = gpd.read_file(shapefilepaths)
# Create a base Folium map
world_map = folium.Map(location=[0, 0], zoom_start=2)

# Add a choropleth overlay with population data
folium.Choropleth(
    geo_data=world.__geo_interface__,  # Converts the GeoDataFrame to GeoJSON
    data=world,  # The GeoDataFrame
    columns=['ISO_A3', 'POP_EST'],  # Mapping ISO country code and population estimation columns
    key_on='feature.properties.ISO_A3',  # Refers to the unique key in the GeoDataFrame
    fill_color='YlGnBu',  # Color scale for visualizing data
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Population Estimation'
).add_to(world_map)

# Save to HTML
world_map.save('world_pop.html')
