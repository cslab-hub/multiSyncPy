"""Data Generation for Synchrony Metrics

This module provides functions used to generate synthetic data for the purposes of testing and exploring multivariate synchrony metrics. It contains the following functions:

 * autoregressive_data - Produces a time series using a stochastic autoregressive function of order two.
 * kuramoto_data - Produces a time series using a Kuramoto model. 
"""

import numpy as np

def autoregressive_data(length, phi_1, phi_2, epsilon, c=0):
    """Generates synthetic data using an autoregressive function where each time point depends on the previous two.
    
    Parameters
    ----------
    length: int
        The number of time steps to generate.
    phi_1: float
        The weighting of the value two time steps ago.
    phi_2: float
        The weighting of the value one time steps ago.
    epsilon: float
        The standard deviation of Gaussian noise added at each time step.
    c: float
        An optional bias term.
    
    Returns
    -------
    autoregressive_data: array
        An array containing autoregressive synthetic data.
    """
    
    x_1 = c + np.random.normal(0) * phi_1 * phi_2
    x_2 = c + np.random.normal(0) * phi_1 * phi_2
    
    outputs = []
    
    for i in range(length + 100): ## 100 extra points just to make sure that the autoregressive process has settled down
        
        x_new = c + (phi_1 * x_1) + (phi_2 * x_2) + np.random.normal(0, epsilon)
        
        outputs.append(x_new)
        
        x_1 = x_2
        x_2 = x_new
        
    return np.array(outputs[-length:])


def kuramoto_data(phases, omegas, K, alpha, d_t, length):
    """Generates synthetic data from a Kuramoto model of coupled oscillators.
    
    Parameters
    ----------
    phases: array
        An array of initial phases in radians for each oscillator. 
    omegas: array
        The natural frequencies of each oscillator. Must be the same length as phases. 
    K: float
        The coupling strength between oscillators in the model. 
    alpha: float
        A parameter to control the amount of Gaussian noise added at each time step. 
    d_t: float
        The change in time that each time step represents.
    length: int
        The number of time points desired in the synthetic data.
    
    Returns
    -------
    kuramoto_data: array
        An array containing synthetic data generated from the Kuramoto model. 
    """
    
    def update_phases(phases, omegas, K, alpha, d_t):

        psi = np.mean(phases)

        r = np.linalg.norm([np.cos(phases), np.sin(phases)])

        return phases + d_t * (omegas + K * r * np.sin(psi - phases)) + alpha * np.random.normal(0, np.sqrt(d_t), phases.shape)
    
    phases_sequence = []

    for i in range(length):

        phases_sequence.append(phases)
        phases = update_phases(phases, omegas, K, alpha, d_t)
        
    return np.sin(np.array(phases_sequence)).T


def linear_nonstationary_transition(begin_frequency, end_frequency, sampling_rate, duration):
    
    transition_frequencies = np.linspace(begin_frequency, end_frequency, duration * sampling_rate)

    return np.exp(np.cumsum(transition_frequencies)/sampling_rate * np.pi * 2j).real