import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colormaps


# Load sample GeoDataFrame
shapefilepaths = 'C:\\Users\\saide\\Downloads\\110m_cultural\\ne_110m_admin_0_countries.shp'

world = gpd.read_file(shapefilepaths)

# Inspect the data
print(world.head())
"""column_names = list(world.columns)
print(cols)
col_maps = list(colormaps) #use this to find all the possible color gradients.
print(col_maps)"""
world.plot(column = 'POP_EST', cmap = 'inferno_r', legend = 'True')
plt.title('Population estimate by country')
plt.show()
