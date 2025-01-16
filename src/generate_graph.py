import torch
import numpy as np
from myRead import myRead

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def edge_func(a):
    return a

def generate_graph_adjacency_matrix_from_positions(positions, func=edge_func, device='cpu', threshold=None):
    """
    Generate an adjacency matrix for an undirected weighted graph based on 3D particle positions.

    Parameters:
        positions (torch.Tensor): Tensor of shape (N, 3) representing the 3D coordinates of N particles.
        func (function): Function to compute edge weights from distances. Defaults to distance.
        device (str): Device to store the tensor ('cpu' or 'cuda').
        threshold (float, optional): Maximum distance threshold. Edges with distances > threshold are removed.

    Returns:
        torch.Tensor: Adjacency matrix of the graph, where edge weights are calculated using the provided function.
    """
    distance = torch.cdist(positions, positions)
    if threshold is not None:
        distance[distance > threshold] = 0  # Remove edges with distance greater than threshold
    adjacency_matrix = func(distance)
    adjacency_matrix.fill_diagonal_(0)  # No self-loops
    return adjacency_matrix


def mst_adjacency_matrix(adjacency_matrix):
    """
    Compute the Minimum Spanning Forest (MSF) adjacency matrix from the given graph's adjacency matrix using Kruskal's algorithm on GPU.

    Parameters:
        adjacency_matrix (torch.Tensor): Adjacency matrix of the graph.

    Returns:
        torch.Tensor: Adjacency matrix of the MSF.
    """
    device = adjacency_matrix.device
    num_nodes = adjacency_matrix.size(0)

    # Get edges and weights
    edges = []
    weights = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] > 0:
                edges.append((i, j))
                weights.append(adjacency_matrix[i, j].item())

    edges = torch.tensor(edges, device=device)
    weights = torch.tensor(weights, device=device)

    # Sort edges by weight
    sorted_indices = torch.argsort(weights)
    edges = edges[sorted_indices]
    weights = weights[sorted_indices]

    # Initialize union-find structure
    parent = torch.arange(num_nodes, device=device)
    rank = torch.zeros(num_nodes, device=device, dtype=torch.int32)

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])  # Path compression
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            # Union by rank
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1

    # Kruskal's algorithm for Minimum Spanning Forest
    msf_matrix = torch.zeros_like(adjacency_matrix)
    for edge, weight in zip(edges, weights):
        u, v = edge
        if find(u) != find(v):
            union(u, v)
            msf_matrix[u, v] = weight
            msf_matrix[v, u] = weight

    return msf_matrix

def visualize_graph_3d(adjacency_matrix, positions, title="3D Graph Visualization"):
    """
    Visualize a 3D graph from its adjacency matrix.

    Parameters:
        adjacency_matrix (torch.Tensor): Adjacency matrix of the graph.
        positions (torch.Tensor): Positions of the nodes for layout (N, 3).
        title (str): Title of the graph plot.
    """
    num_nodes = adjacency_matrix.size(0)
    adjacency_matrix = adjacency_matrix.cpu().numpy()
    positions = positions.cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] > 0:
                # Get start and end positions
                x = [positions[i, 0], positions[j, 0]]
                y = [positions[i, 1], positions[j, 1]]
                z = [positions[i, 2], positions[j, 2]]
                ax.plot(x, y, z, 'k-', alpha=0.5)  # Edge line

                # Add edge weight text
                mid_x = (x[0] + x[1]) / 2
                mid_y = (y[0] + y[1]) / 2
                mid_z = (z[0] + z[1]) / 2
                ax.text(mid_x, mid_y, mid_z, f"{adjacency_matrix[i, j]:.2f}", color='red', fontsize=8)

    # Draw nodes
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               s=200, c='lightblue', edgecolors='k', depthshade=True, zorder=2)
    for i in range(num_nodes):
        ax.text(positions[i, 0], positions[i, 1], positions[i, 2], str(i), fontsize=10, zorder=3)

    ax.set_title(title)
    plt.savefig("test_graph.png")

# Example usage
if __name__ == "__main__":
    test=True
    if test==True:
        torch.manual_seed(42)

        # Generate a small example graph
        num_nodes = 30
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        positions = torch.rand((num_nodes, 3), device=device) * 10  # Random positions in a 10x10x10 cube
        threshold = 4.0
        adjacency_matrix = generate_graph_adjacency_matrix_from_positions(
            positions, device=device, threshold=threshold
        )

        print("\nAdjacency Matrix:")
        print(adjacency_matrix)

        mst_matrix = mst_adjacency_matrix(adjacency_matrix)

        # Visualize the graph in 3D
        visualize_graph_3d(mst_matrix, positions, title="3D Graph Visualization")

    else:
        f_base = '../samples/cola/f5000.properties'
        pos, mass = myRead(f_base)
        
        pos = torch.tensor(pos).to('cuda')
        mass = torch.tensor(mass).to('cuda')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        adjacency_matrix = generate_graph_adjacency_matrix_from_positions(pos, device=device,threshold=300)

        print("Original Graph Adjacency Matrix:")
        print(adjacency_matrix)

        # Compute the MST adjacency matrix
        mst_matrix = mst_adjacency_matrix(adjacency_matrix)

        print("\nMST Adjacency Matrix:")
        np.savetxt("tensor.txt", mst_matrix.cpu().numpy(), fmt="%.6f")
