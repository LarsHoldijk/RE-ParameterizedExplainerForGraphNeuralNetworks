import numpy as np
from sklearn.metrics import roc_auc_score


def evaluation_auc(task, explanations, explanation_labels, indices):
    """Determines based on the task which auc evaluation method should be called to determine the AUC score

    :param task: str either "node" or "graph".
    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :returns: area under curve score.
    """
    if task == 'graph':
        return evaluation_auc_graph(explanations, explanation_labels, indices)
    elif task == 'node':
        return evaluation_auc_node(explanations, explanation_labels)


def evaluation_auc_graph(explanations, explanation_labels, indices):
    """Evaluate the auc score given explaination and ground truth labels.

    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :returns: area under curve score.
    """
    ground_truth = []
    predictions = []

    for idx, n in enumerate(indices): # Use idx for explanation list and indices for ground truth list

        # Select explanation
        mask = explanations[idx][1].detach().numpy()
        graph = explanations[idx][0].detach().numpy()

        # Select ground truths
        edge_list = explanation_labels[0][n]
        edge_labels = explanation_labels[1][n]

        for edge_idx in range(0, edge_labels.shape[0]): # Consider every edge in the ground truth
            edge_ = edge_list.T[edge_idx]
            if edge_[0] == edge_[1]:  # We dont consider self loops for our evaluation (Needed for Ba2Motif)
                continue
            t = np.where((graph.T == edge_.T).all(axis=1)) # Determine index of edge in graph

            # Retrieve predictions and ground truth
            predictions.append(mask[t][0])
            ground_truth.append(edge_labels[edge_idx])

    score = roc_auc_score(ground_truth, predictions)
    return score


def evaluation_auc_node(explanations, explanation_labels):
    """Evaluate the auc score given explaination and ground truth labels.

    :param explanations: predicted labels.
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate. We ignore all others.
    :returns: area under curve score.
    """
    ground_truth = []
    predictions = []
    for expl in explanations: # Loop over the explanations for each node

        ground_truth_node = []
        prediction_node = []

        for i in range(0, expl[0].size(1)): # Loop over all edges in the explanation sub-graph
            prediction_node.append(expl[1][i].item())

            # Graphs are defined bidirectional, so we need to retrieve both edges
            pair = expl[0].T[i].numpy()
            idx_edge = np.where((explanation_labels[0].T == pair).all(axis=1))[0]
            idx_edge_rev = np.where((explanation_labels[0].T == [pair[1], pair[0]]).all(axis=1))[0]

            # If any of the edges is in the ground truth set, the edge should be in the explanation
            gt = explanation_labels[1][idx_edge] + explanation_labels[1][idx_edge_rev]
            if gt == 0:
                ground_truth_node.append(0)
            else:
                ground_truth_node.append(1)

        ground_truth.extend(ground_truth_node)
        predictions.extend(prediction_node)

    score = roc_auc_score(ground_truth, predictions)
    return score