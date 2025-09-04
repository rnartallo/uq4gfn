import numpy as np
import pandas as pd
import networkx as nx

from numpy.random import default_rng
from pgmpy.models import LinearGaussianBayesianNetwork, BayesianNetwork
from pgmpy.sampling import BayesianModelSampling


# def sample_from_linear_gaussian(model, num_samples, rng=default_rng()):
#     """Sample from a linear-Gaussian model using ancestral sampling."""
#     if not isinstance(model, LinearGaussianBayesianNetwork):
#         raise ValueError('The model must be an instance '
#                          'of LinearGaussianBayesianNetwork')

#     samples = pd.DataFrame(columns=list(model.nodes()))
#     for node in nx.topological_sort(model):
#         cpd = model.get_cpds(node)

#         if cpd.evidence:
#             values = np.vstack([samples[parent] for parent in cpd.evidence])
#             mean = cpd.mean[0] + np.dot(cpd.mean[1:], values)
#             samples[node] = rng.normal(mean, cpd.variance)
#         else:
#             samples[node] = rng.normal(cpd.mean[0], cpd.variance, size=(num_samples,))

#     return samples

def sample_from_linear_gaussian(model, num_samples, rng=default_rng()):
    if not isinstance(model, LinearGaussianBayesianNetwork):
        raise ValueError('The model must be an instance of LinearGaussianBayesianNetwork')

    samples = pd.DataFrame(index=range(num_samples), columns=list(model.nodes()))
    for node in nx.topological_sort(model):
        cpd = model.get_cpds(node)

        if cpd.evidence:
            # Stack parent samples: shape (num_parents, num_samples)
            parent_samples = np.vstack([samples[parent].values for parent in cpd.evidence])
            # mean shape: (num_samples,)
            mean = cpd.beta[0] + np.dot(cpd.beta[1:], parent_samples)
            samples[node] = rng.normal(mean, cpd.std)
        else:
            samples[node] = rng.normal(cpd.beta[0], cpd.std, size=num_samples)

    return samples


def sample_from_discrete(model, num_samples, rng=default_rng(), **kwargs):
    """Sample from a discrete model using ancestral sampling."""
    if not isinstance(model, BayesianNetwork):
        raise ValueError('The model must be an instance of BayesianNetwork')
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=num_samples, show_progress=False, **kwargs)

    # Convert values to pd.Categorical for faster operations
    for node in samples.columns:
        cpd = model.get_cpds(node)
        samples[node] = pd.Categorical(samples[node], categories=cpd.state_names[node])

    return samples
