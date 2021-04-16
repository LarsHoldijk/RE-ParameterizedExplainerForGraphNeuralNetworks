import numpy as np
import scipy.sparse as sp
import torch
import scipy
import pickle as pkl
from scipy.sparse import coo_matrix

"""
Most of the functions in this module are copied from the PGExplainer code base. This ensures that the data is handled in the same way.

link: https://github.com/flyingdoog/PGExplainer
"""

def adj_to_edge_index(adj):
    """
    Convert an adjacency matrix to an edge index
    :param adj: Original adjacency matrix
    :return: Edge index representation of the graphs
    """
    converted = []
    for d in adj:
        edge_index = np.argwhere(d > 0.).T
        converted.append(edge_index)

    return converted


def preprocess_features(features):
    """
    Preprocess the features and transforms them into the edge index representation
    :param features: Orginal feature representation
    :return: edge index representation
    """
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features).astype(np.float32)
    try:
        return features.todense() # [coordinates, data, shape], []
    except:
        return features


def preprocess_adj(adj):
    """
    Transforms adj matrix into edge index.
    Is different to adj_to_edge_index in terms of how the final representation can be used
    :param adj: adjacency matrix
    :return: edge index
    """
    return sparse_to_tuple(sp.coo_matrix(adj))


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        values = values.astype(np.float32)
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def get_graph_data(path):
    """Obtain the mutagenicity dataset from text files.
    
    :param path: Location of the txt files.
    :returns: np.array, np.array, np.array, np.array
    """
    pri = path

    file_edges = pri+'A.txt'
    file_edge_labels = pri+'edge_labels.txt'
    file_edge_labels = pri+'edge_gt.txt'
    file_graph_indicator = pri+'graph_indicator.txt'
    file_graph_labels = pri+'graph_labels.txt'
    file_node_labels = pri+'node_labels.txt'

    edges = np.loadtxt( file_edges,delimiter=',').astype(np.int32)
    edge_labels = np.loadtxt(file_edge_labels,delimiter=',').astype(np.int32)
    graph_indicator = np.loadtxt(file_graph_indicator,delimiter=',').astype(np.int32)
    graph_labels = np.loadtxt(file_graph_labels,delimiter=',').astype(np.int32)
    node_labels = np.loadtxt(file_node_labels,delimiter=',').astype(np.int32)

    graph_id = 1
    starts = [1]
    node2graph = {}
    for i in range(len(graph_indicator)):
        if graph_indicator[i]!=graph_id:
            graph_id = graph_indicator[i]
            starts.append(i+1)
        node2graph[i+1]=len(starts)-1

    graphid  = 0
    edge_lists = []
    edge_label_lists = []
    edge_list = []
    edge_label_list = []
    for (s,t),l in list(zip(edges,edge_labels)):
        sgid = node2graph[s]
        tgid = node2graph[t]
        if sgid!=tgid:
            print('edges connecting different graphs, error here, please check.')
            print(s,t,'graph id',sgid,tgid)
            exit(1)
        gid = sgid
        if gid !=  graphid:
            edge_lists.append(edge_list)
            edge_label_lists.append(edge_label_list)
            edge_list = []
            edge_label_list = []
            graphid = gid
        start = starts[gid]
        edge_list.append((s-start,t-start))
        edge_label_list.append(l)

    edge_lists.append(edge_list)
    edge_label_lists.append(edge_label_list)

    # node labels
    node_label_lists = []
    graphid = 0
    node_label_list = []
    for i in range(len(node_labels)):
        nid = i+1
        gid = node2graph[nid]
        # start = starts[gid]
        if gid!=graphid:
            node_label_lists.append(node_label_list)
            graphid = gid
            node_label_list = []
        node_label_list.append(node_labels[i])
    node_label_lists.append(node_label_list)

    return edge_lists, graph_labels, edge_label_lists, node_label_lists


def load_real_dataset(path_pkl, path_graph):
    """Obtain the mutagenicity dataset from text files.
    
    :param path_pkl: Path to save the pickle file containing the mutagenicity dataset.
    :param path_graph: Location of the txt files.
    :returns: adjecency matrix, node features, labels.
    """
    edge_lists, graph_labels, edge_label_lists, node_label_lists = get_graph_data(path_graph)

    graph_labels[graph_labels == -1] = 0

    max_node_nmb = np.max([len(node_label) for node_label in node_label_lists]) + 1  # add nodes for each graph

    edge_label_nmb = np.max([np.max(l) for l in edge_label_lists]) + 1
    node_label_nmb = np.max([np.max(l) for l in node_label_lists]) + 1

    for gid in range(len(edge_lists)):
        node_nmb = len(node_label_lists[gid])
        for nid in range(node_nmb, max_node_nmb):
            edge_lists[gid].append((nid, nid))  # add self edges
            node_label_lists[gid].append(node_label_nmb)  # the label of added node is node_label_nmb
            edge_label_lists[gid].append(edge_label_nmb)

    adjs = []
    for edge_list in edge_lists:
        row = np.array(edge_list)[:, 0]
        col = np.array(edge_list)[:, 1]
        data = np.ones(row.shape)
        adj = coo_matrix((data, (row, col))).toarray()
        if True: # originally checked the adjacency to be normal
            degree = np.sum(adj, axis=0, dtype=float).squeeze()
            degree[degree == 0] = 1
            sqrt_deg = np.diag(1.0 / np.sqrt(degree))
            adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
        adjs.append(np.expand_dims(adj, 0))

    labels = graph_labels

    adjs = np.concatenate(adjs, 0)
    labels = np.array(labels).astype(int)
    feas = []

    for node_label in node_label_lists:
        fea = np.zeros((len(node_label), node_label_nmb + 1))
        rows = np.arange(len(node_label))
        fea[rows, node_label] = 1
        fea = fea[:, :-1]  # remove the added node feature

        if node_label_nmb < 3:
            const_features = np.ones([fea.shape[0], 10])
            fea = np.concatenate([fea, const_features], -1)
        feas.append(fea)

    feas = np.array(feas)

    b = np.zeros((labels.size, labels.max() + 1))
    b[np.arange(labels.size), labels] = 1
    labels = b
    with open(path_pkl,'wb') as fout:
        pkl.dump((adjs, feas,labels),fout)
    return adjs, feas, labels
