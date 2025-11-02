# graph_visualization.py - CORRECTED VERSION
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import os

# Load your data with the region clusters
df = pd.read_csv('earthquakes_final_model_ready.csv', parse_dates=['time'])
print("Data loaded. Plotting...")

# 1. Create a base world map plot using the DOWNLOADED file
# Make sure you have extracted the Natural Earth ZIP file
world_path = 'ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp' # Path to the .shp file
if os.path.exists(world_path):
    world = gpd.read_file(world_path)
else:
    # Fallback: create a simple map from your data bounds
    print("World map file not found. Creating simplified map from earthquake bounds.")
    from shapely.geometry import Point
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    world = gpd.GeoDataFrame(df, geometry=geometry)
    world = world.set_geometry('geometry')

base = world.plot(color='lightgray', edgecolor='black', figsize=(20, 10))

# 2. Scatter plot of all earthquakes, colored by their region_cluster
scatter = plt.scatter(df['longitude'], df['latitude'], c=df['region_cluster'], 
                       cmap='tab20', s=1, alpha=0.5)
plt.colorbar(scatter, label='Region Cluster ID')
plt.title("Earthquakes (2010-2024) Colored by Tectonic Region Cluster")

# 3. Overlay Tectonic Plate Boundaries (The Key New Step)
plate_path = 'PB2002_plates.json' # Make sure this file is in your folder
if os.path.exists(plate_path):
    plates = gpd.read_file(plate_path)
    plates.plot(ax=plt.gca(), color='none', edgecolor='red', linewidth=1.5, label='Plate Boundary')
    plt.legend()
    print("✓ Plate boundaries plotted.")
else:
    print(f"Plate boundary file not found at {plate_path}. Please download 'PB2002_plates.json'.")

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('tectonic_region_map_with_plates.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'tectonic_region_map_with_plates.png'")
plt.show()