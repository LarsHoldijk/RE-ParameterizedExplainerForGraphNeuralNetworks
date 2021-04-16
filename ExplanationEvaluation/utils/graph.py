import torch

def index_edge(graph, pair):
    return torch.where((graph.T == pair).all(dim=1))[0]
