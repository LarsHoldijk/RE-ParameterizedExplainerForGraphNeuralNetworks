# [Re] Parameterized Explainer for Graph Neural Networks
**Authors**: Maarten Boon, Stijn Henckens, [Lars Holdijk](https://www.larsholdijk.com/) and Lysander de Jong

This repository contains all code required to replicate our replication study of the paper _Parameterized Explainer for Graph Neural Networks_[1]. This includes both a new implementation of the PGExplainer method introduced as well as a reimplementation of the earlier introduced GNNExplainer [2], which serves as a benchmark in the evaluation. In addition to this, the repository contains the datasets and pretrained models needed for a faithfull replication.

In addition to ipython notebooks for replicating our study, the repository also contains an example that shows how our codebase can be used for your own experiments. This example also serves as a good starting point for understanding how the replication study is performed.

**bibtex citation**
```
@inproceedings{holdijk2021re,
  title={[Re] Parameterized Explainer for Graph Neural Network},
  author={Holdijk, Lars and Boon, Maarten and Henckens, Stijn and de Jong, Lysander},
  booktitle={ML Reproducibility Challenge 2020},
  year={2021}
}
```

## IPython Notebooks
Four IPython Notebooks are availabe to replicate our experiments

- **experiment_model_training**: Replicates the trained models used in the evaluations. Instead of retraining the models yourself, it is also possible to reuse the already trained models.
- **experiment_replication**: Replicates the main replication study of our paper. By default the notebook uses the pretrained models.
- **experiment_ablation**: Replicates the small ablation study found in the paper
- **example_explain_your_model**: This notebook is not part of the replication study but instead serves as a starting point for reusing our code in your own project.

## Codebase
All code required for the replication study can be found in the `ExplanationEvaluation` module. This also includes the required datasets and pretrained models.

## Configurations
In the folder `ExplanataionEvaluation` all configuration files needed to replicate the replication study can be found. A discussion of their setup can be found in the appendix of the correspondig paper.


## Installation
Install required packages using
```pip install -r requirements.txt```
additionally follow the [instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) in order to install PyTorch Geometric.


## Resources
- [1] [GNNExplainer](https://arxiv.org/pdf/1903.03894.pdf)
- [2] [PGExplainer](https://arxiv.org/pdf/2011.04573.pdf)
