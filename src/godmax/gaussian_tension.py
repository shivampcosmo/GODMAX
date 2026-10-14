"""
Copied from Marco Raveri's tensiometer github repository. We don't need the full package, just these functions.
"""

"""
This file contains the functions and utilities to compute agreement and
disagreement between two different chains using a Gaussian approximation
for the posterior.

For more details on the method implemented see
`arxiv 1806.04649 <https://arxiv.org/pdf/1806.04649.pdf>`_
and `arxiv 1912.04880 <https://arxiv.org/pdf/1912.04880.pdf>`_.
"""

###############################################################################
# initial imports:

import scipy
import numpy as np
import copy
from getdist import MCSamples
from getdist.gaussian_mixtures import GaussianND
import matplotlib.pyplot as plt

###############################################################################
# series of helpers to check input of functions:


def _check_param_names(chain, param_names):
    """
    Utility to check input param names.
    """
    if param_names is None:
        param_names = chain.getParamNames().getRunningNames()
    else:
        param_list = chain.getParamNames().list()
        if not np.all([name in param_list for name in param_names]):
            raise ValueError('Input parameter is not in the chain',
                             chain.name_tag, '\n'
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', param_list)
    #
    return param_names


def _check_common_names(param_names_1, param_names_2):
    """
    Utility to get the common param names between two chains.
    """
    param_names = [name for name in param_names_1 if name in param_names_2]
    if len(param_names) == 0:
        raise ValueError('Chains do not have shared parameters.\n',
                         'Parameters for chain_1 ', param_names_1, '\n',
                         'Parameters for chain_2 ', param_names_2, '\n')
    #
    return param_names


def _check_chain_type(chain):
    """
    Check if an object is a GetDist chain.
    """
    # test the type of the chain:
    if not isinstance(chain, MCSamples):
        raise TypeError('Input chain is not of MCSamples type.')

###############################################################################


def get_prior_covariance(chain, param_names=None):
    """
    Utility to estimate the prior covariance from the ranges of a chain.
    The flat range prior covariance
    (`link <https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)>`_)
    is given by:

    .. math:: C_{ij} = \\delta_{ij} \\frac{( \\mathrm{max}(p_i) - \\mathrm{min}(p_i) )^2}{12}

    :param chain: the input chain.
    :type chain: :class:`~getdist.mcsamples.MCSamples`
    :param param_names: choice of parameter names to
        restrict the calculation.
    :type param_names: optional
    :return: the estimated covariance of the prior.
    """
    # get the parameter names to use:
    param_names = _check_param_names(chain, param_names)
    # get the ranges:
    _prior_min = []
    _prior_max = []
    for name in param_names:
        # lower bound:
        if name in chain.ranges.lower.keys():
            _prior_min.append(chain.ranges.lower[name])
        else:
            _prior_min.append(-1.e30)
            # upper bound:
        if name in chain.ranges.upper.keys():
            _prior_max.append(chain.ranges.upper[name])
        else:
            _prior_max.append(1.e30)
    _prior_min = np.array(_prior_min)
    _prior_max = np.array(_prior_max)
    #
    return np.diag((_prior_max-_prior_min)**2/12.)


def get_localized_covariance(chain_1, chain_2, param_names,
                             localize_params=None, scale=10.):
    """
    This function calculates the localized covariance matrix of `chain_1` using `chain_2` as a reference.
    The localization is performed by adjusting the weights of the samples in `chain_1` based on their
    likelihoods with respect to `chain_2`. The resulting covariance matrix is then computed by removing
    the localization covariance from the full covariance matrix of `chain_1`.

    :param chain_1: the first chain to be localized.
    :type chain_1: :class:`~getdist.mcsamples.MCSamples`
    :param chain_2: the second chain used for localization.
    :type chain_2: :class:`~getdist.mcsamples.MCSamples`
    :param param_names: the names of the parameters.
    :param localize_params: the parameters to be localized. If not provided, all parameters will be localized.
    :type localize_params: optional
    :param scale: the scaling factor for the covariance matrix. Default is 10.
    :type scale: optional
    :return: The localized covariance matrix.
    """
    # initialize param names:
    param_names_1 = _check_param_names(chain_1, param_names)
    param_names_2 = _check_param_names(chain_2, param_names)
    param_names = _check_common_names(param_names_1, param_names_2)
    # check localized parameters:
    if localize_params is None:
        localize_params = param_names
    else:
        if not np.all([name in param_names for name in localize_params]):
            raise ValueError('Input localize_params is not in param_names')
    # get mean and covariance of the chain that we use for localization:
    mean = chain_2.getMeans(pars=[chain_2.index[name]
                                  for name in localize_params])
    cov = chain_2.cov(pars=localize_params)
    inv_cov = np.linalg.inv(scale**2*cov)
    sqrt_inv_cov = scipy.linalg.sqrtm(inv_cov)
    # get the Gaussian chi2:
    idx = [chain_1.index[name] for name in localize_params]
    X = np.dot(sqrt_inv_cov, (chain_1.samples[:, idx] - mean).T).T
    logLikes = (X*X).sum(axis=1)
    max_logLikes = np.amin(logLikes)
    # compute weights:
    new_weights = chain_1.weights * np.exp(-(logLikes - max_logLikes))
    # check that weights are reasonable:
    old_neff_samples = np.sum(chain_1.weights)**2 / np.sum(chain_1.weights**2)
    new_neff_samples = np.sum(new_weights)**2 / np.sum(new_weights**2)
    if old_neff_samples / new_neff_samples > 10.:
        print('WARNING: localization of covariance is resulting in too many '
              + 'samples being under-weighted.\n'
              + 'Neff original = ', round(old_neff_samples, 3), '\n'
              + 'Neff new      = ', round(new_neff_samples, 3), '\n'
              + 'this can result in large errors and can be improved with '
              + 'more samples in chain_1.')
    # compute covariance with all parameters:
    idx_full = [chain_1.index[name] for name in param_names]
    # compute covariance:
    cov2 = np.cov(chain_1.samples[:, idx_full].T, aweights=new_weights)
    # remove localization covariance:
    idx_rel = [param_names.index(name) for name in localize_params]
    inv_cov2 = np.linalg.inv(cov2)
    inv_cov2[np.ix_(idx_rel, idx_rel)] = inv_cov2[np.ix_(idx_rel, idx_rel)] \
        - inv_cov
    cov2 = np.linalg.inv(inv_cov2)
    #
    return cov2

###############################################################################


def get_Neff(chain, prior_chain=None, param_names=None,
             prior_factor=1.0, localize=False, **kwargs):
    """
    Function to compute the number of effective parameters constrained by a
    chain over the prior.
    The number of effective parameters is defined as in Eq. (29) of
    (`Raveri and Hu 18 <https://arxiv.org/pdf/1806.04649.pdf>`_) as:

    .. math:: N_{\\rm eff} \\equiv
        N -{\\rm tr}[ \\mathcal{C}_\\Pi^{-1}\\mathcal{C}_p ]

    where :math:`N` is the total number of nominal parameters of the chain,
    :math:`\\mathcal{C}_\\Pi` is the covariance of the prior and
    :math:`\\mathcal{C}_p` is the posterior covariance.

    :param chain: the input chain.
    :type chain: :class:`~getdist.mcsamples.MCSamples`
    :param prior_chain: the prior chain.
        If the prior is not well approximated by
        a ranged prior and is informative it is better to explicitly
        use a prior only chain.
        If this is not given the algorithm will assume ranged priors with the
        ranges computed from the input chain.
    :type prior_chain: :class:`~getdist.mcsamples.MCSamples` - optional
    :param param_names: parameter names to restrict the
        calculation of :math:`N_{\\rm eff}`.
        If none is given the default assumes that all running parameters
        should be used.
    :type param_names: optional
    :param prior_factor: factor to scale the prior covariance.
        In case of strongly non-Gaussian posteriors it might be useful to
        artificially tighten the prior to have less noise in telling apart
        parameter space directions that are constrained by data and prior.
        Default is no scaling, prior_factor=1.
    :type prior_factor: optional
    :param localize: whether to localize the covariance.
    :type localize: optional
    :return: the number of effective parameters.
    """
    # initialize param names:
    param_names = _check_param_names(chain, param_names)
    # initialize prior covariance:
    if prior_chain is not None:
        # check parameter names:
        param_names = _check_param_names(prior_chain, param_names)
        # get the prior covariance:
        if localize:
            C_Pi = get_localized_covariance(prior_chain, chain,
                                            param_names, **kwargs)
        else:
            C_Pi = prior_chain.cov(pars=param_names)
    else:
        C_Pi = get_prior_covariance(chain, param_names=param_names)
    # multiply by prior factor:
    C_Pi = prior_factor*C_Pi
    # get the posterior covariance:
    C_p = chain.cov(pars=param_names)
    # compute the number of effective parameters
    _temp = np.dot(np.linalg.inv(C_Pi), C_p)
    # compute Neff from the regularized spectrum of the eigenvalues:
    _eigv, _eigvec = np.linalg.eig(_temp)
    _eigv[_eigv > 1.] = 1.
    _eigv[_eigv < 0.] = 0.
    #
    _Ntot = len(_eigv)
    _Neff = _Ntot - np.real(np.sum(_eigv))
    #
    return _Neff

###############################################################################


def gaussian_approximation(chain, param_names=None, **kwargs):
    """
    Function that computes the Gaussian approximation of a given chain.

    :param chain: the input chain.
    :type chain: :class:`~getdist.mcsamples.MCSamples`
    :param param_names: parameter names to restrict the
        Gaussian approximation.
        If none is given the default assumes that all parameters
        should be used.
    :type param_names: optional
    :return: :class:`~getdist.gaussian_mixtures.GaussianND` object with the
        Gaussian approximation of the chain.
    """
    # initial checks:
    _check_chain_type(chain)
    if param_names is None:
        param_names = chain.getParamNames().list()
    param_names = _check_param_names(chain, param_names)
    # get the mean:
    mean = chain.getMeans(pars=[chain.index[name]
                          for name in param_names])
    # get the covariance:
    cov = chain.cov(pars=param_names)
    # get the labels:
    param_labels = [_n.label for _n
                    in chain.getParamNames().parsWithNames(param_names)]
    # get label:
    if chain.label is not None:
        label = 'Gaussian '+chain.label
    elif chain.name_tag is not None:
        label = 'Gaussian_'+chain.name_tag
    else:
        label = None
    # initialize the Gaussian distribution:
    gaussian_approx = GaussianND(mean, cov,
                                 names=param_names,
                                 labels=param_labels,
                                 label=label,
                                 **kwargs)
    #
    return gaussian_approx

