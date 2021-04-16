import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

""" 
The function in this file is largely copied from the orginal PGExplainer codebase. The decision was made to largely copy this file to ensure
that the graph visualization between the original and replicate results would be as similar as possible. Additional comments were added
to clarify the code. 
"""

def plot(graph, edge_weigths, labels, idx, thres_min, thres_snip, dataset, args=None, gt=None, show=False):
    """
    Function that can plot an explanation (sub)graph and store the image.

    :param graph: graph provided by explainer
    :param edge_weigths: Mask of edge weights provided by explainer
    :param labels: Label of each node required for coloring of nodes
    :param idx: Node index of interesting node
    :param thresh_min: total number of edges
    :param thres_snip: number of top edges
    :param args: Object containing arguments from configuration
    :param gt: Ground Truth
    :param show: flag to show plot made
    """
    # Set thresholds
    sorted_edge_weigths, _ = torch.sort(edge_weigths)

    thres_index = max(int(edge_weigths.shape[0]-thres_snip),0)

    thres = sorted_edge_weigths[thres_index]
    if thres_min == -1:
        filter_thres_index = 0
    else:
        filter_thres_index = min(thres_index,
                                max(int(edge_weigths.shape[0]-edge_weigths.shape[0]/2),
                                    edge_weigths.shape[0]-thres_min))
    filter_thres = sorted_edge_weigths[filter_thres_index]
    # Init edges
    filter_nodes = set()
    filter_edges = []
    pos_edges = []
    # Select all edges and nodes to plot
    for i in range(edge_weigths.shape[0]):
        # Select important edges
        if edge_weigths[i] >= thres and not graph[0][i] == graph[1][i]:
            pos_edges.append((graph[0][i].item(),graph[1][i].item()))
        # Select all edges to plot
        if edge_weigths[i] > filter_thres and not graph[0][i] == graph[1][i]:
            filter_edges.append((graph[0][i].item(),graph[1][i].item()))
            filter_nodes.add(graph[0][i].item())
            filter_nodes.add(graph[1][i].item())
    num_nodes = len(pos_edges)

    # Initialize graph object
    G = nx.Graph()

    if not (thres_min == -1):
        # Deal with plotting of node datasets
        G.add_edges_from(filter_edges)
        pos = nx.kamada_kawai_layout(G)

        label = []
        for node in filter_nodes:
            label.append(int(labels[node]))

        for cc in nx.connected_components(G):
            if idx in cc:
                G = G.subgraph(cc).copy()
                break

        pos_edges = [(u, v) for (u, v) in pos_edges if u in G.nodes() and v in G.nodes()]
        colors = ['orange', 'red', 'green', 'blue', 'maroon', 'brown', 'darkslategray', 'paleturquoise', 'darksalmon',
                  'slategray', 'mediumseagreen', 'mediumblue', 'orchid', ]
        if dataset=='syn3':
            colors = ['orange', 'blue']


        if dataset=='syn4':
            colors = ['orange', 'black','black','black','blue']

        # node coloring
        label2nodes= []
        max_label = np.max(label)+1 # amount of labels to use
        nmb_nodes = len(filter_nodes) # amount of nodes that need coloring

        # Create empty lists of possible labels
        for i in range(max_label):
            label2nodes.append([])

        # For each node add the node to it's assigned label
        for i in range(nmb_nodes):
            label2nodes[label[i]].append(list(filter_nodes)[i])

        # actually draw the nodes
        for i in range(len(label2nodes)):
            node_list = []
            # For each label that needs a color
            for j in range(len(label2nodes[i])):
                if label2nodes[i][j] in G.nodes():
                    node_list.append(label2nodes[i][j])
            # Draw all nodes of a certain color
            nx.draw_networkx_nodes(G,
                                    pos,
                                    nodelist=node_list,
                                    node_color=colors[i % len(colors)],
                                    node_size=500)

        # Draw a base node
        if idx in pos.keys():
            nx.draw_networkx_nodes(G, pos,
                                   nodelist=[idx],
                                   node_color=colors[labels[idx]],
                                   node_size=1000)

    # Deal with plotting of graph datasets
    else:
        # Format edges
        edges = [(pair[0], pair[1]) for pair in gt[0][idx].T]
        # Obtain all unique nodes
        nodes = np.unique(gt[0][idx])
        # Add all unique nodes and all edges
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        # Let the graph generate all positions
        pos = nx.kamada_kawai_layout(G)

        pos_edges = [(u, v) for (u, v) in pos_edges if u in G.nodes() and v in G.nodes()]

        nx.draw_networkx_nodes(G,
                               pos,
                               nodelist=nodes,
                               node_color='red',
                               node_size=500)


    # Draw an edge
    nx.draw_networkx_edges(G,
                           pos,
                           width=7,
                           alpha=0.5,
                           edge_color='grey')

    # Draw all pos edges
    nx.draw_networkx_edges(G,
                           pos,
                           edgelist=pos_edges,
                           width=7,
                           alpha=0.5)
    plt.axis('off')
    if show:
        plt.show()
    else:
        save_path = f'./qualitative/e_{args.explainer}/m_{args.model}/d_{args.dataset}/'

        # Generate folders if they do not exist
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Save figure
        plt.savefig(f'{save_path}{idx}.png')
        plt.clf()
