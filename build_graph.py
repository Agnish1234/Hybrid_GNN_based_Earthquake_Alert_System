# build_graph.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Load your data
df = pd.read_csv('earthquakes_final_model_ready.csv')

# Calculate centroid of each region
region_centroids = df.groupby('region_cluster')[['latitude', 'longitude']].mean()
print("Region Centroids:")
print(region_centroids)

# Calculate distances between all region centroids
distances = cdist(region_centroids, region_centroids, metric='euclidean')
print(f"\nDistance matrix shape: {distances.shape}")

# Create edges between regions closer than threshold (e.g., 10 degrees)
distance_threshold = 10.0
edges = []
for i in range(len(distances)):
    for j in range(i+1, len(distances)):
        if distances[i, j] < distance_threshold:
            edges.append((i, j))

print(f"Number of edges created: {len(edges)}")
print("Edges (region_i, region_j):", edges[:10])  # Show first 10 edges

# This gives you the basic connectivity for your graph neural network