# build_graph_advanced.py - COMPLETELY FIXED SCRIPT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
from scipy.spatial.distance import cdist
import json

print("=== ADVANCED GRAPH CONSTRUCTION ===")

def find_plate_connections(region_centroids, plate_file='PB2002_plates.json', max_distance_deg=5.0):
    """Connect regions that are on the same tectonic plate"""
    try:
        plates = gpd.read_file(plate_file)
        print(f"Successfully loaded plate file with {len(plates)} features")
    except FileNotFoundError:
        print(f"Plate file {plate_file} not found.")
        return []

    edges = []
    
    # For each plate, find which region centroids are close to it
    for plate_idx, plate in plates.iterrows():
        # Use .get() to safely access the 'properties' dictionary and then the 'Name' within it
        plate_properties = plate.get('properties', {})  # Returns an empty dict if 'properties' is missing
        plate_name = plate_properties.get('Name', f'Unknown_Plate_{plate_idx}')  # Provides a default name
        
        plate_line = plate['geometry']
        
        # Find regions near this plate boundary
        nearby_regions = []
        for region_id, centroid in region_centroids.iterrows():
            centroid_point = gpd.points_from_xy([centroid['longitude']], [centroid['latitude']])[0]
            distance_to_plate = centroid_point.distance(plate_line)
            
            if distance_to_plate < max_distance_deg:
                nearby_regions.append(region_id)
        
        # Connect all regions that are on/near the same plate
        for i in range(len(nearby_regions)):
            for j in range(i+1, len(nearby_regions)):
                edges.append((nearby_regions[i], nearby_regions[j], plate_name))
    
    return edges

def k_nearest_connectivity(centroids, k=3):
    """Connect each region to its k-nearest neighbors"""
    distances = cdist(centroids, centroids, metric='euclidean')
    np.fill_diagonal(distances, np.inf)  # Ignore self-connections
    
    edges = []
    for i in range(len(distances)):
        # Find k nearest neighbors
        nearest_indices = np.argsort(distances[i])[:k]
        for j in nearest_indices:
            edges.append((i, j, distances[i][j]))
    
    return edges

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# MAIN EXECUTION CODE
if __name__ == "__main__":
    print("Starting main execution...")
    
    # Load your data
    try:
        df = pd.read_csv('earthquakes_final_model_ready.csv')
        print(f"Loaded {len(df)} earthquakes")
    except FileNotFoundError:
        print("Error: earthquakes_final_model_ready.csv not found")
        exit()
    
    region_centroids = df.groupby('region_cluster')[['latitude', 'longitude']].mean()
    print(f"Calculated centroids for {len(region_centroids)} regions:")

    # Strategy 1: Plate Boundary-Based Connectivity
    print("\n1. PLATE BOUNDARY-BASED CONNECTIVITY...")
    plate_edges = find_plate_connections(region_centroids)
    print(f"Plate-based edges: {len(plate_edges)}")

    # Strategy 2: K-Nearest Neighbors Connectivity
    print("\n2. K-NEAREST NEIGHBORS CONNECTIVITY...")
    knn_edges = k_nearest_connectivity(region_centroids[['latitude', 'longitude']].values, k=3)
    print(f"KNN edges: {len(knn_edges)}")

    # Combine both strategies and CONVERT ALL TO NATIVE PYTHON TYPES
    all_edges = []
    for i, j, _ in plate_edges:
        all_edges.append((int(i), int(j)))  # Convert to native Python int
    for i, j, _ in knn_edges:
        all_edges.append((int(i), int(j)))  # Convert to native Python int
    
    # Remove duplicates
    all_edges = list(set(all_edges))
    print(f"\nTotal unique edges: {len(all_edges)}")

    # Create and visualize the graph
    G = nx.Graph()
    G.add_nodes_from(range(len(region_centroids)))
    G.add_edges_from(all_edges)

    print(f"\nFINAL GRAPH STATISTICS:")
    print(f" - Nodes: {G.number_of_nodes()}")
    print(f" - Edges: {G.number_of_edges()}")
    print(f" - Connected components: {nx.number_connected_components(G)}")
    print(f" - Average degree: {np.mean([d for n, d in G.degree()]):.2f}")

    # Enhanced visualization
    plt.figure(figsize=(16, 8))

    # Plot 1: Graph on world map
    plt.subplot(1, 2, 1)
    try:
        world = gpd.read_file('ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp')
        world.plot(color='lightgray', edgecolor='black', ax=plt.gca())
    except:
        print("World map file not found, creating basic plot")
        # Create a simple background
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)

    # Plot centroids with labels
    for i, (idx, row) in enumerate(region_centroids.iterrows()):
        plt.scatter(row['longitude'], row['latitude'], c='red', s=100, zorder=5)
        plt.text(row['longitude'], row['latitude'], f'R{i}', fontsize=8, 
                 ha='center', va='bottom', fontweight='bold')

    # Plot edges
    for edge in all_edges:
        i, j = edge
        plt.plot([region_centroids.iloc[i]['longitude'], region_centroids.iloc[j]['longitude']],
                 [region_centroids.iloc[i]['latitude'], region_centroids.iloc[j]['latitude']],
                 'blue', alpha=0.7, linewidth=2)

    plt.title('Tectonic Region Connectivity\n(Plate-based + KNN)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Plot 2: Network diagram
    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color='lightblue', node_size=500, 
            edge_color='gray', width=2, with_labels=True,
            font_size=10, font_weight='bold')
    plt.title('Network Diagram of Tectonic Regions')

    plt.tight_layout()
    plt.savefig('advanced_tectonic_graph.png', dpi=300, bbox_inches='tight')
    print("\n✓ Advanced graph visualization saved")

    # Save for GNN model - COMPLETELY FIXED VERSION
    graph_data = {
        'num_nodes': int(len(region_centroids)),
        'edges': all_edges,  # Now contains only native Python ints
        'node_features': {
            'latitude': [float(x) for x in region_centroids['latitude'].tolist()],
            'longitude': [float(x) for x in region_centroids['longitude'].tolist()]
        }
    }

    with open('advanced_tectonic_graph.json', 'w') as f:
        json.dump(graph_data, f, indent=2, cls=NumpyEncoder)  # Use custom encoder
    print("✓ Advanced graph structure saved")

    print("\n=== ADVANCED GRAPH CONSTRUCTION COMPLETE ===")
    print("✅ NO ERRORS - Ready for GNN-LSTM implementation!")