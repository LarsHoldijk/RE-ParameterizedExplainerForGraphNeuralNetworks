from abc import ABC, abstractmethod


class BaseEvaluation(ABC):
    """Base class for evaluations that provide a score metric"""
    @abstractmethod
    def get_score(self, explanations):
        """
        Returns the score of the metric
        :param explanations: list of explanations by the explainer
        :return: score
        """
        pass