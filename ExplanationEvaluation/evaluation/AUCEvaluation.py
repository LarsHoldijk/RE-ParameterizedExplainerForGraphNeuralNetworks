from ExplanationEvaluation.evaluation.BaseEvaluation import BaseEvaluation
from ExplanationEvaluation.evaluation.utils import evaluation_auc


class AUCEvaluation(BaseEvaluation):
    """
    A class enabling the evaluation of the AUC metric on both graphs and nodes.
    
    :param task: str either "node" or "graph".
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate.
    
    :funcion get_score: obtain the roc auc score.
    """
    def __init__(self, task, ground_truth, indices):
        self.task = task
        self.ground_truth = ground_truth
        self.indices = indices

    def get_score(self, explanations):
        """
        Determines the auc score based on the given list of explanations and the list of ground truths
        :param explanations: list of explanations
        :return: auc score
        """
        return evaluation_auc(self.task, explanations, self.ground_truth, self.indices)
