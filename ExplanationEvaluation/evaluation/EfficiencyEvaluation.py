import time

from ExplanationEvaluation.evaluation.BaseEvaluation import BaseEvaluation


class EfficiencyEvluation(BaseEvaluation):
    def __init__(self):
        self.t_prep = 0
        self.t_expl = 0
        self.t_done = 0

    def reset(self):
        """Resets all times"""
        self.t_prep = 0
        self.t_expl = 0
        self.t_done = 0

    def start_prepate(self):
        """Should be called when the evaluation starts preparing the explainer"""
        self.t_prep = time.time()

    def start_explaining(self):
        """Should be called when the explainers starts explaining the samples"""
        self.t_expl = time.time()

    def done_explaining(self):
        """Should be called when the explainer is done explaining all the samples"""
        self.t_done = time.time()

    def get_score(self, explanations):
        """Returns the time it took to explain a single instance
        :param explanations: List of all explanations performed
        :return: time it took to explain a single instance
        """
        return (self.t_done - self.t_expl) / len(explanations) * 1000