import pipmaster as pm

if not pm.is_installed("pyvis"):
    pm.install("pyvis")
if not pm.is_installed("networkx"):
    pm.install("networkx")

import networkx as nx
from pyvis.network import Network
import random

# Load the GraphML file
G = nx.read_graphml("./dickens/graph_chunk_entity_relation.graphml")

# Create a Pyvis network with modern settings
net = Network(height="100vh", notebook=True, bgcolor="#ffffff", font_color="#2c3e50")

# Set physics options for better layout
net.set_options("""
{
    "physics": {
        "forceAtlas2Based": {
            "gravitationalConstant": -100,
            "centralGravity": 0.01,
            "springLength": 200,
            "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {
            "enabled": true,
            "iterations": 1000,
            "updateInterval": 50
        }
    }
}
""")

# Convert NetworkX graph to Pyvis network
net.from_nx(G)

# Define a color palette for nodes
color_palette = [
    "#3498db", "#2ecc71", "#e74c3c", "#f1c40f", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#7f8c8d", "#16a085"
]

# Add colors and styling to nodes
for node in net.nodes:
    # Assign color based on node type or randomly
    node["color"] = random.choice(color_palette)
    # Set node size based on degree (if available)
    node["size"] = 25 + random.randint(5, 15)
    # Add border
    node["borderWidth"] = 2
    node["borderColor"] = "#2c3e50"
    # Add shadow effect
    node["shadow"] = True
    # Set font properties
    node["font"] = {
        "size": 14,
        "face": "Arial"
    }
    if "description" in node:
        node["title"] = node["description"]

# Add styling to edges
for edge in net.edges:
    # Set edge color
    edge["color"] = "#95a5a6"
    # Set edge width
    edge["width"] = 2
    # Add smooth edges
    edge["smooth"] = {
        "type": "continuous",
        "roundness": 0.5
    }
    # Add hover effect
    edge["hoverWidth"] = 3
    if "description" in edge:
        edge["title"] = edge["description"]

# Save and display the network
net.show("knowledge_graph.html")
