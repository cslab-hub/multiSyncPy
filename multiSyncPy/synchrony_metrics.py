"""Synchrony Metrics

This module provides functions used to compute synchrony metrics on multivariate time series. It contains the following functions:

 * recurrence_matrix - Creates a recurrence matrix from a multivariate time series. 
 * get_diagonal_lengths - Finds the lengths of diagonals in a recurrence matrix. Used by rqa_metrics. 
 * rqa_metrics - Computes the proportion of recurrence, proportion of determinism, average diagonal length and longest diagonal length for an input recurrence matrix. 
 * rho - A cluster-phase synchrony metric.
 * coherence_team - A synchrony metric based on spectral density.
 * sum_normalized_csd - A synchrony metric based on cross-spectral density, similar to coherence_team().
 * convert_to_terciles - Takes a time series and returns a time series where each value is replaced by a number indicating which tercile it belongs in. Used by pattern_entropy. 
 * symbolic_entropy - A metric based on the entropy of the combined 'state' across a multivariate time series. 
 * kuramoto_weak_null - Tests the significance of the observed Kuramoto order parameter values in a sample of multivariate time series. 
 * metric_fixed_parameters - Provides a copy of a function to calculate a synchrony metric, but with all parameters fixed except the input data. For use with apply_windowed when functions have multiple parameters. 
 * apply_windowed - A function used to apply other functions in a windowed fashion. 
 * shuffle_recordings - Creates surrogate_data by shuffling variables between time series in a sample of multivariate time series. 
 * shuffle_time_windows - Creates surrogate_data by shuffling time windows, separately for each variable of a multivariate time series. 
"""

import numpy as np
import scipy.spatial
import scipy.signal
import scipy.stats
import copy


def recurrence_matrix(data, radius, normalise=True, embedding_dimension=None, embedding_delay=None):
    """Creates a recurrence matrix from a multivariate time series. The Euclidean distance, combined with the radius parameter, is used to determine which points are close enough to count as 'recurrent'. 
    
    Parameters
    ----------
    data: ndarray
        An array containing the multivariate time series with shape (number_signals, duration).
    radius: float
        The Euclidean distance below which two points will count as recurrent.
    normalise: bool
        Whether to apply normalisation. Normalisation is applied on each variable separately, transforming the data to have mean 0 and variance 1, which is intended to help balance the relative importance of each variable when calculating Euclidean distances.
    embedding_dimension: int
        The number of copies of the multivariate time series to use. If provided, embedding_delay must also be used. 
    embedding_delay: int
        If using embedding_dimension, this is the delay in number of time steps that is applied to each new copy of the multivariate time series. 
    
    Returns
    -------
    recurrence_matrix: ndarray
        A square matrix with shape (duration, duration). Cells have value True when two time points are recurrent, and False otherwise. 
    """
    
    if normalise:
        
        data = (data - data.mean(axis=1).reshape(-1,1)) / data.std(axis=1).reshape(-1,1)
    
    if embedding_dimension and embedding_delay:
        
        copies = []
        
        copy_length = data.shape[1] - (embedding_delay * (embedding_dimension-1))
        
        for i in range(embedding_dimension):
            
            copies.append(data[:, i*embedding_delay:copy_length+i*embedding_delay])
    
        data = np.concatenate(copies)
    
    distance_matrix = scipy.spatial.distance_matrix(data.T, data.T)
    
    return distance_matrix < radius


def get_diagonal_lengths(recurrence_matrix):
    """Returns the lengths of sequences where the successive cells have a value of 1, and how many times sequences of each length were observed, looking along the diagonals of a recurrence matrix. Considers only the upper triangle of the recurrence matrix, not including the line of identity. 
    
    Parameters
    ----------
    recurrence_matrix: ndarray
        Matrix indicating which time points are recurrent with other time points. Truthy values are used to indicate a recurrence.
    
    Returns
    -------
    full_diagonal_length_counts: dict
        A dict where the keys are the length of a sequence and the values are the number of times a sequence of that length was observed. 
    """
    
    full_diagonal_length_counts = {}
    
    def get_lengths(x):
        
        diagonal_length_counts = {}
        current_length = 0
        
        for cell in x:
            
            if cell:
                
                current_length += 1 
                
            else:
                
                if current_length > 0:
                    
                    diagonal_length_counts[current_length] = diagonal_length_counts.get(current_length, 0) + 1
                
                current_length = 0
                
        if current_length > 0:
                    
            diagonal_length_counts[current_length] = diagonal_length_counts.get(current_length, 0) + 1
                
        return diagonal_length_counts
    
    for diagonal in range(1,recurrence_matrix.shape[0]):
        
        temp_length_counts = get_lengths(np.diag(recurrence_matrix, diagonal))
        
        for length, count in temp_length_counts.items():
            
            full_diagonal_length_counts[length] = full_diagonal_length_counts.get(length,0) + count
            
    return full_diagonal_length_counts


def rqa_metrics(recurrence_matrix, min_length=2):
    """Returns the proportion of recurrence, proportion of determinism, mean diagonal length, and max diagonal length, for the input recurrence matrix. 
    
    Parameters
    ----------
    recurrence_matrix: ndarray
        Matrix indicating which time points are recurrent with other time points. Truthy values are used to indicate a recurrence.
    
    Returns
    -------
    rec: float
        A value between 0 and 1 representing the proportion of recurrence observed in the recurrence matrix. Multiply by 100 to get the %REC. 
    det: float
        A value between 0 and 1 representing the proportion of determinism observed in the recurrence matrix. Multiply by 100 to get the %det. 
    mean_length: float
        The mean length of diagonal sequences in the recurrence matrix. 
    max_length: int
        The maximum length of diagonal sequences in the recurrence matrix. 
    """
    
    diagonal_length_counts = get_diagonal_lengths(recurrence_matrix)
    
    rec = 0
    det = 0
    mean_length = 0
    max_length = 0
    
    if diagonal_length_counts.keys():
    
        for length, count in diagonal_length_counts.items():

            rec += length * count

            if length >= min_length:

                det += length * count

            mean_length = rec
            
        det = det / rec ##Divide by the number of recurrent points (before rec becomes a fraction)
        rec = rec / (recurrence_matrix.shape[0]*(recurrence_matrix.shape[1]-1)/2) ##The number of off-diagonal cells in the upper triangle 
        mean_length = mean_length / sum(diagonal_length_counts.values())
        max_length = max(diagonal_length_counts.keys())
    
    return rec, det, mean_length, max_length


def rho(phases):
    """Returns the quantity defined by Richardson et al. as 'rho' in "Measuring group synchrony: a cluster-phase method foranalyzing multivariate movement time-series:, doi: 10.3389/fphys.2012.00405. 
    
    Parameters
    ----------
    phases: ndarray
        The phase time series (in radians) of the signals with the shape (number_signals, duration).
    
    Returns
    -------
    rho_group_i: ndarray
        The quantity rho, computed for each signal at each time step.
    rho_group: ndarray
        The quantity rho averaged over time.
    """

    # Group level
    q_dash = np.exp(phases * 1j).mean(axis=0)
    q = np.arctan2(q_dash.imag, q_dash.real)
    # Individual level
    phi = phases - q
    phi_bar_dash = np.exp(phi * 1j).mean(axis=1)
    phi_bar = np.arctan2(phi_bar_dash.imag, phi_bar_dash.real)
    rho = np.abs(phi_bar_dash)
    # Group level
    rho_group_i = np.abs(np.exp((phi - phi_bar[:,None]) * 1j).mean(axis=0))
    rho_group = rho_group_i.mean()
    
    return rho_group_i, rho_group


def coherence_team(data, nperseg=None):
    """Returns the quantity defined by Reinero, Dikker, and Bavel as 'coherence' in "Inter-brain synchrony in teams predicts collective performance", doi: 10.1093/scan/nsaa135, with the quantity being averaged across the team.
    
    Parameters
    ----------
    data: ndarray
        An array containing the time series of measurements with shape (number_signals, duration).
    nperseg: int
        The number of time steps used to form a 'sample' of the signal when computing coherence, see scipy.signal.coherence documentation for more details. Optional, and will default to the lesser of (data duration / 4) and 256. 
    
    Returns
    -------
    coherence: float
        The quantity coherence. 
    """
    
    ## Set nperseg to a reasonable default value for shorter input lengths
    if nperseg is None:
        if (data.shape[1] // 256) < 4:  ## Default value is 256 
            nperseg = data.shape[1] // 4
    
    coherence_scores = []
    
    for i, x in enumerate(data):
        
        for j, y in enumerate(data):
            
            if i < j:
                
                coherence_scores.append(scipy.signal.coherence(x, y, nperseg=nperseg)[1].mean()) ## Actually we should just use the scipy coherence function
                
    return np.mean(coherence_scores)


def sum_normalized_csd(data):
    """Returns a quantity, based on the cross-spectral density (CSD), similar to that of coherence_team() but which is less impacted by Gaussian noise.
    
    Parameters
    ----------
    data: ndarray
        An array containing the time series of measurements with shape (number_signals, duration).
    
    Returns
    -------
    aggregated_csd: float
        The sum-normalized CSD quantity.
    """
    
    ## Set nperseg to a reasonable value for shorter input lengths
    if data.shape[1] // 256 < 4: ## Default value is 256
    
        nperseg = data.shape[1] // 4
    
    else:
    
        nperseg = None ## Let scipy set it to default value
    
    csd_scores = []
    
    for i, x in enumerate(data):
    
        for j, y in enumerate(data):
        
            if i < j:
            
                csd_scores.append(
                    (np.abs(scipy.signal.csd(x, y, nperseg=nperseg)[1]) ** 2).sum() / (scipy.signal.csd(x, x, nperseg=nperseg)[1] * scipy.signal.csd(y, y, nperseg=nperseg)[1]).sum()
                )
    
    return np.mean(csd_scores)


def convert_to_terciles(data):
    """Maps the input time series to numbers representing 'low', 'medium' and 'high' values. The thresholds for deciding 'low', 'medium' and 'high' are terciles. 
    
    Parameters
    ----------
    data: array
        An array containing the time series of measurements for a single signal. 
    
    Returns
    -------
    data_terciles: array
        An array where 0 represents a 'low' value, 1 represents a 'medium' value and 2 represents a 'high' value
    """
    
    terciles = np.quantile(data.reshape(-1),[1/3,2/3])
    
    data_terciles = data.copy()
    data_terciles[:] = 2
    data_terciles[data<terciles[1]] = 1
    data_terciles[data<terciles[0]] = 0
    
    return data_terciles


def symbolic_entropy(data):
    """Computes entropy after mapping the signals to numbers representing 'low', 'medium' and 'high' values, and then concatenating these numbers (across the signals) to create a 'pattern' at each time step. The thresholds for deciding 'low', 'medium' and 'high' are terciles. 
    
    Parameters
    ----------
    data: ndarray
        An array containing the time series of measurements, with the shape (number_signals, duration). 
    
    Returns
    -------
    pattern_entropy: float
        The Shannon entropy of symbols found by mapping the input signals to 'low', 'medium' and 'high' and concatenating across signals. 
    """
    
    data_terciles = np.apply_along_axis(convert_to_terciles, 1, data)
    data_patterns = np.apply_along_axis(lambda x: "".join([str(int(y)) for y in x]), 0, data_terciles)
    
    pattern_probabilities = np.unique(data_patterns, return_counts=True)[1]/data_patterns.shape[0]
    
    return -np.sum(pattern_probabilities * np.log(pattern_probabilities))


def kuramoto_weak_null(phases):
    """Estimates the significance of the Kuramoto order parameter for a sample of multi-signal recordings, according to the 'weak null' test described by Frank and Richardson in "On a test statistic for the Kuramoto order parameter of synchronization: An illustration for group synchronization during rocking chairs", doi: 10.1016/j.physd.2010.07.015. 
    
    Parameters
    ----------
    phases: list
        A list containing the phase time series (in radians) for each member of the sample, with each phase time series being an array with the shape (number_signals, duration). Each multivariate time series in the sample can be a different duration, although different numbers of signals are not permissible. 
    
    Returns
    -------
    p-value: float
        The p-value of the observed Kuramoto order parameter.
    t-statistic: float
        The t-statistic.
    df: float
        The degrees of freedom.
    """
    
    ## Check that the number of signals is the same in each time series 
    assert len(np.unique(list(map(lambda x: x.shape[0], phases)))) == 1, "The number of signals in each time series must be consistent across the whole sample."
    
    def y_bar(phases):
    
        kuramoto_r = np.abs(np.exp(phases * 1j).mean(axis=0))
        
        y = kuramoto_r**2
        
        return y.mean(axis=-1)
    
    y_bars = np.fromiter(map(y_bar, phases), dtype=np.float)
    
    M = y_bars.mean() 
    
    mu = 1/phases[0].shape[0]
    
    s = y_bars.std()
    
    R_root = len(phases)**0.5
    
    t_statistic = (M - mu) / (s / R_root)
    
    df = len(phases) - 1
    
    return scipy.stats.t.sf(t_statistic, df=df), t_statistic, df ## This is a one-sided t-test


def metric_fixed_parameters(function, parameters):
    """Returns a copy of a function to compute a metric, but with all parameters fixed except the data input. For use with apply_windowed. 
    
    Parameters
    ----------
    function: function
        A function to compute a synchrony metric.
    parameters: dict
        A dictionary containing the parameters and their values for the function, except the main data input parameter.
    
    Returns
    -------
    new_function: function
        A copy of the function to compute a synchrony metric, with all parameters fixed except the data input.
    """
    
    parameters = copy.deepcopy(parameters)
    
    def new_function(data):
        
        return function(data, **parameters)
    
    return new_function


def apply_windowed(data, function, window_length, step=None):
    """Applies a function in a windowed fashion to a time series with shape (number_signals, duration). 
    
    Parameters
    ----------
    data: ndarray
        An array containing the time series of measurements, with the shape (number_signals, duration).
    function: function
        The function to apply within each window. 
    window_length: int
        The number of time steps to be included in a window.
    step: int
        The number of time steps by which to move forward in order to obtain the next window. 
    
    Returns
    -------
    windowed_results: ndarray
        A numpy array containing the results of the function when it is applied to each window. 
    """
    
    if step is None:
    
        step = window_length
    
    def windowed_view(data, window_length, step):
    
        dim_0 = 1 + (data.shape[1] - window_length) // step
        dim_1 = data.shape[0]
        dim_2 = window_length
        
        if step is None:
            step = dim_2
        
        stride_0, stride_1 = data.strides
        
        return np.lib.stride_tricks.as_strided(data, shape=(dim_0,dim_1,dim_2), strides=(stride_1 * step, stride_0, stride_1))
    
    return np.array(list(map(function, windowed_view(data, window_length, step))))


def shuffle_recordings(data):
    """Creates surrogate data by shuffling variables between time series in a sample of multivariate recordings. This assumes that all recordings in the sample are the same length. 
    
    Parameters
    ----------
    data: ndarray
        An array containing a sample of recordings with shape (number_recordings, number_signals, duration). 
    
    Returns
    -------
    surrogate_data: ndarray
        An array containing a sample of recordings where the variables have been shuffled between recordings, with shape (number_recordings, number_signals, duration).
    """
    
    surrogate_shape = data.shape
    
    surrogate_data = data.copy()
    
    surrogate_data = surrogate_data.reshape(surrogate_shape[0] * surrogate_shape[1],-1)
    
    np.random.shuffle(surrogate_data)
    
    return surrogate_data.reshape(surrogate_shape[0],surrogate_shape[1],-1)


def shuffle_time_windows(data, window_length):
    """Creates surrogate data by shuffling windows of data within each variable in a multivariate time series. 
    
    Parameters
    ----------
    data: ndarray
        An array containing a multivariate recording with shape (number_signals, duration).
    window_length: int
        The number of time steps to use as the window length. 
    
    Returns
    -------
    surrogate_data: ndarray
        An array containing a multivariate recording where windows of time have been shuffled independently for each variable, with shape (number_signals, duration).
    """
    
    def shuffle_individual(data):
    
        surrogate_shape = data.shape

        surrogate_data = data.copy()

        surrogate_data = surrogate_data.reshape(-1, window_length)

        np.random.shuffle(surrogate_data)

        return surrogate_data.reshape(surrogate_shape[0])
    
    return np.apply_along_axis(shuffle_individual, -1, data)
