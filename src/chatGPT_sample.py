import torch
from torch_geometric.utils import get_laplacian, to_dense_adj
from cuml.metrics import pairwise_distances

# ダークマターハローの位置データ
positions = torch.rand((1000, 3))  # サンプルデータ（1000個の3次元座標）

# cuML を用いた距離行列の計算
distances = pairwise_distances(positions.numpy(), metric='euclidean')

# PyTorch Geometric でエッジリストと重みを構築
edge_index = torch.combinations(torch.arange(positions.shape[0]), r=2).T
edge_weight = torch.tensor(distances[edge_index[0], edge_index[1]])

# グラフラプラシアンを計算
laplacian, _ = get_laplacian(edge_index, edge_weight)

# 固有値分解 (GFT)
laplacian_matrix = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_matrix.cuda())
