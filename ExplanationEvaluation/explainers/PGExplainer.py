import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm

from ExplanationEvaluation.explainers.BaseExplainer import BaseExplainer
from ExplanationEvaluation.utils.graph import index_edge

class PGExplainer(BaseExplainer):
    """
    A class encaptulating the PGExplainer (https://arxiv.org/abs/2011.04573).
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs.
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph".
    :param epochs: amount of epochs to train our explainer.
    :param lr: learning rate used in the training of the explainer.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.
    
    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function _loss: calculate the loss of the explainer during training.
    :function train: train the explainer
    :function explain: search for the subgraph which contributes most to the clasification decision of the model-to-be-explained.
    """
    def __init__(self, model_to_explain, graphs, features, task, epochs=30, lr=0.003, temp=(5.0, 2.0), reg_coefs=(0.05, 1.0),sample_bias=0):
        super().__init__(model_to_explain, graphs, features, task)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias

        if self.type == "graph":
            self.expl_embedding = self.model_to_explain.embedding_size * 2
        else:
            self.expl_embedding = self.model_to_explain.embedding_size * 3


    def _create_explainer_input(self, pair, embeds, node_id):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]
        if self.type == 'node':
            node_embed = embeds[node_id].repeat(rows.size(0), 1)
            input_expl = torch.cat([row_embeds, col_embeds, node_embed], 1)
        else:
            # Node id is not used in this case
            input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl


    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph


    def _loss(self, masked_pred, original_pred, mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]

        # Regularization losses
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        return cce_loss + size_loss + mask_ent_loss

    def prepare(self, indices=None):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if indices is None: # Consider all indices
            indices = range(0, self.graphs.size(0))

        self.train(indices=indices)

    def train(self, indices = None):
        """
        Main method to train the model
        :param indices: Indices that we want to use for training.
        :return:
        """
        # Make sure the explainer model can be trained
        self.explainer_model.train()

        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))

        # If we are explaining a graph, we can determine the embeddings before we run
        if self.type == 'node':
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()

        # Start training loop
        for e in tqdm(range(0, self.epochs)):
            optimizer.zero_grad()
            loss = torch.FloatTensor([0]).detach()
            t = temp_schedule(e)

            for n in indices:
                n = int(n)
                if self.type == 'node':
                    # Similar to the original paper we only consider a subgraph for explaining
                    feats = self.features
                    graph = ptgeom.utils.k_hop_subgraph(n, 3, self.graphs)[1]
                else:
                    feats = self.features[n].detach()
                    graph = self.graphs[n].detach()
                    embeds = self.model_to_explain.embedding(feats, graph).detach()

                # Sample possible explanation
                input_expl = self._create_explainer_input(graph, embeds, n).unsqueeze(0)
                sampling_weights = self.explainer_model(input_expl)
                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()

                masked_pred = self.model_to_explain(feats, graph, edge_weights=mask)
                original_pred = self.model_to_explain(feats, graph)

                if self.type == 'node': # we only care for the prediction of the node
                    masked_pred = masked_pred[n].unsqueeze(dim=0)
                    original_pred = original_pred[n]

                id_loss = self._loss(masked_pred, torch.argmax(original_pred).unsqueeze(0), mask, self.reg_coefs)
                loss += id_loss

            loss.backward()
            optimizer.step()

    def explain(self, index):
        """
        Given the index of a node/graph this method returns its explanation. This only gives sensible results if the prepare method has
        already been called.
        :param index: index of the node/graph that we wish to explain
        :return: explanaiton graph and edge weights
        """
        index = int(index)
        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()
        else:
            feats = self.features[index].clone().detach()
            graph = self.graphs[index].clone().detach()
            embeds = self.model_to_explain.embedding(feats, graph).detach()

        # Use explainer mlp to get an explanation
        input_expl = self._create_explainer_input(graph, embeds, index).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()

        expl_graph_weights = torch.zeros(graph.size(1)) # Combine with original graph
        for i in range(0, mask.size(0)):
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        return graph, expl_graph_weights
