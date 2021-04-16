import torch
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.nn import global_mean_pool, global_max_pool


class NodeGCN(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """
    def __init__(self, num_features, num_classes):
        super(NodeGCN, self).__init__()
        self.embedding_size = 20 * 3
        self.conv1 = GCNConv(num_features, 20)
        self.relu1 = ReLU()
        self.bn1 = BatchNorm(20)        # BN is not used in GNNExplainer
        self.conv2 = GCNConv(20, 20)
        self.relu2 = ReLU()
        self.bn2 = BatchNorm(20)
        self.conv3 = GCNConv(20, 20)
        self.relu3 = ReLU()
        self.lin = Linear(self.embedding_size, num_classes)

    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)
        out = self.lin(input_lin)
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = self.relu1(out1)
        out1 = self.bn1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = self.relu2(out2)
        out2 = self.bn2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin

class GraphGCN(torch.nn.Module):
    """
    A graph clasification model for graphs decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    In between the GCN outputs and linear layers are pooling operations in both mean and max.
    """
    def __init__(self, num_features, num_classes):
        super(GraphGCN, self).__init__()
        self.embedding_size = 20
        self.conv1 = GCNConv(num_features, 20)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(20, 20)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(20, 20)
        self.relu3 = ReLU()
        self.lin = Linear(self.embedding_size * 2, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.embedding(x, edge_index, edge_weights)

        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        input_lin = torch.cat([out1, out2], dim=-1)

        out = self.lin(input_lin)
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)

        input_lin = out3

        return input_lin
