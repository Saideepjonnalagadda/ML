import geopandas as gpd
import matplotlib.pyplot as plt

# Load sample GeoDataFrame (world boundaries are often available as a sample dataset)
shapefilepaths = 'C:\\Users\\saide\\Downloads\\110m_cultural\\ne_110m_admin_0_countries.shp'

world = gpd.read_file(shapefilepaths)

# Inspect the data
print(world.head())  # It will display columns including 'geometry' which contains the polygons for countries
world.plot()
plt.show()
