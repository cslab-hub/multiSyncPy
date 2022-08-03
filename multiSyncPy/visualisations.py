"""Synchrony Metrics

This module provides functions used to calculate and visualise synchrony metrics from a multivariate time series. It contains the following functions:

 * plot_entropy - Calculates and visualises the symbolic entropy of a multivariate time series. 
 * plot_coherence - Calculates and visualises the coherence of a multivariate time series. 
 * plot_csd - Calculates and visualises the 'sum-normalized cross spectral density' of a multivariate time series. 
 * plot_rho - Calculates and visualises the cluster-phase 'rho' of a multivariate time series. 
 * plot_synchronyindex - Calculates and visualises the 'synchronization index' of a multivariate time series. 

"""

import scipy 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import multiSyncPy.synchrony_metrics as sm 


def plot_entropy(data, window_length, step=None, figsize=None, custom_cols=None, gradient=False):  
    """
    Plots the windowed group entropy over a heatmap of the individual signals' tercile patterns.  
    
    Parameters
    ----------
    data: ndarray
        An array containing the multivariate time series with shape (number_signals, duration).
    window_length: int
        The number of time steps to be included in a window.
    step: int
        The number of time steps by which to move forward in order to obtain the next window. 
    figsize: tuple
        The width and height of the figure in inches, respectively. Default is (8,4). 
    custom_cols: tuple 
        The heatmap colormap/colors and line color to use for the plots. Default is (['#D4ECFC', '#FCF49C', '#FCC4AC'], 'black'). 
    gradient: bool 
        Determines whether the heatmap is displayed as a gradient, or matches the window and step size of the group metric. Default is False. 
    
    Returns
    -------
    figure: figure
        A figure containing the windowed group entropy and the individual signals' tercile patterns, along with axes labels and a colorbar.
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

    if gradient == True: 
        windowed_signals = (windowed_view(data, window_length=window_length, step=5)) # step size of 5 for the gradient 
    else: 
        windowed_signals = (windowed_view(data, window_length=window_length, step=step))
        
    yticks = [] # obtaining y tick labels from the number of windowed_signals 
    for i in range(windowed_signals.shape[1]): 
        yticks.append(i+1)

    data_terciles = np.apply_along_axis(sm.convert_to_terciles, 1, windowed_signals).mean(axis=2).T

    windowed_entropy =  sm.apply_windowed(data, sm.symbolic_entropy, window_length=window_length, step=step)

    if figsize is None: 
        figsize = (8,4)    
    if custom_cols is None: 
        custom_cols = (['#D4ECFC', '#FCF49C', '#FCC4AC'] , 'black')
    
    # Plotting 
    figure, (ax1, cax) = plt.subplots(nrows=2, figsize=figsize,  gridspec_kw={"height_ratios":[1, 0.05]})
    figure.subplots_adjust(hspace=0.5)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05]) # adding a second grid ontop of the subplots to overlay the group metric line onto the heatmap 
    ax2 = figure.add_subplot(gs[0], frame_on=False)
    ax1.set_xlim(0, data_terciles.shape[0]) # manually setting the x axis limits to ensure ticks of both plots are aligned 
    ax2.set_xlim(0, windowed_entropy.shape[0])
    # plotting the heatmap 
    cbar_kws = {'ticks': [0, 1, 2], 'label':'Individual pattern: 0=low, 1=medium, 2=high', 
                'orientation':'horizontal'} 
    sns.heatmap(data_terciles, linewidth=0, cmap=custom_cols[0], cbar_kws=cbar_kws, cbar_ax=cax, ax=ax1, yticklabels=yticks, vmin=0, vmax=2) 
    # plotting the group metric 
    ax2.plot(windowed_entropy, color=custom_cols[1], linewidth=1)
    # editing axes labels and parameters 
    ax1.tick_params(labelrotation=0) 
    ax1.set(ylabel='Individual pattern')
    ax1.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()  
    ax2.set(ylabel='Symbolic Entropy', xlabel=f'Time (window_length={window_length}, step={step})') 
    ax2.yaxis.set_label_position('right')
    
    return figure
    

def plot_coherence(data, window_length, step=None, gradient=False, figsize=None, custom_cols=None, bounded_yaxis=True):
    """
    Plots the windowed group coherence over a heatmap of the pairwise signal coherence.  
    
    Parameters
    ----------
    data: ndarray
        An array containing the multivariate time series with shape (number_signals, duration).
    window_length: int
        The number of time steps to be included in a window.
    step: int
        The number of time steps by which to move forward in order to obtain the next window. 
    gradient: bool 
        Determines whether the heatmap is displayed as a gradient, or matches the window and step size of the group metric. Default is False. 
    figsize: tuple
        The width and height of the figure in inches, respectively. Default is (8,4). 
    custom_cols: tuple 
        The heatmap colormap/colors and line color to use for the plots. Default is ('YlOrRd', 'black'). 
    bounded_yaxis: bool
        Determines whether the y-axis limits of the group metric are bounded to their range. Default is True. 
        
    Returns
    -------
    figure: figure
        A figure containing the windowed group coherence and the pairwise coherence, along with axes labels and a colorbar.
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
    
    if gradient == True: 
        windowed_signals = (windowed_view(data, window_length=window_length, step=5)) # step size of 5 for the gradient 
    else: 
        windowed_signals = (windowed_view(data, window_length=window_length, step=step))

    nperseg = None 
    if nperseg is None:
        if (windowed_signals.shape[2] // 256) < 4:  ## Default value is 256 
            nperseg = windowed_signals.shape[2] // 4
    
    paired_coherence = []
    
    for dim in range(windowed_signals.shape[0]):
        temp_list = []
        signal_pairs = []
        for i, x in enumerate(windowed_signals[dim, :, :]):
            for j, y in enumerate(windowed_signals[dim, :, :]):
                if i < j:
                    temp_list.append(scipy.signal.coherence(x, y, nperseg=nperseg)[1].mean()) ## Actually we should just use the scipy coherence function
                    signal_pairs.append(str(i+1)+'-'+str(j+1))
        paired_coherence.append(temp_list)
    
    paired_coherence = np.array(paired_coherence)
    
    windowed_coherence = sm.apply_windowed(data, sm.coherence_team, window_length=window_length, step=step)

    if figsize is None: 
        figsize = (8,4)
    if custom_cols is None: 
        custom_cols = ('YlOrRd', 'black')
        
    # Plotting 
    figure, (ax1, cax) = plt.subplots(nrows=2, figsize=figsize,  gridspec_kw={"height_ratios":[1, 0.05]})
    figure.subplots_adjust(hspace=0.5)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05]) # adding a second grid ontop of the subplots to overlay the group metric line onto the heatmap 
    ax2 = figure.add_subplot(gs[0], frame_on=False)
    ax1.set_xlim(0, paired_coherence.shape[0]) # manually setting the x axis limits to ensure ticks of both plots are aligned 
    ax2.set_xlim(0, windowed_coherence.shape[0])
    # plotting the heatmap 
    cbar_kws = {'label':'Coherence', 'orientation':'horizontal'}
    sns.heatmap(paired_coherence.T, linewidth=0, cmap=custom_cols[0], cbar_kws=cbar_kws, cbar_ax=cax, ax=ax1, yticklabels=signal_pairs, vmin=0, vmax=1) 
    # plotting the group metric
    ax2.plot(windowed_coherence, color=custom_cols[1], linewidth=1)
    # editing axes labels and parameters 
    ax1.tick_params(labelrotation=0) 
    ax1.set(ylabel='Pairwise Coherence')
    ax1.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set(ylabel='Group Coherence', xlabel=f'Time (window_length={window_length}, step={step})') 
    ax2.yaxis.set_label_position('right')
    if bounded_yaxis is True: 
        ax2.set_ylim(0, 1)
    
    return figure    
    

def plot_csd(data, window_length, step=None, gradient=False, figsize=None, custom_cols=None, bounded_yaxis=True):
    """
    Plots the windowed group sum-normalized CSD over a heatmap of the pairwise sum-normalized CSD.  
    
    Parameters
    ----------
    data: ndarray
        An array containing the multivariate time series with shape (number_signals, duration).
    window_length: int
        The number of time steps to be included in a window.
    step: int
        The number of time steps by which to move forward in order to obtain the next window. 
    gradient: bool 
        Determines whether the heatmap is displayed as a gradient, or matches the window and step size of the group metric. Default is False. 
    figsize: tuple
        The width and height of the figure in inches, respectively. Default is (8,4). 
    custom_cols: tuple 
        The heatmap colormap/colors and line color to use for the plots. Default is ('YlOrRd', 'black').  
    bounded_yaxis: bool
        Determines whether the y-axis limits of the group metric are bounded to their range. Default is True. 
        
    Returns
    -------
    figure: figure
        A figure containing the windowed group sum-normalized CSD and the pairwise sum-normalized CSD, along with axes labels and a colorbar.
    """
    if step is None: 
        step=window_length 

    def windowed_view(data, window_length, step):
        dim_0 = 1 + (data.shape[1] - window_length) // step
        dim_1 = data.shape[0]
        dim_2 = window_length
        
        if step is None:
            step = dim_2
        stride_0, stride_1 = data.strides
        
        return np.lib.stride_tricks.as_strided(data, shape=(dim_0,dim_1,dim_2), strides=(stride_1 * step, stride_0, stride_1))
    
    if gradient == True: 
        windowed_signals = (windowed_view(data, window_length=window_length, step=5)) # step size of 5 for the gradient 
    else: 
        windowed_signals = (windowed_view(data, window_length=window_length, step=step))
        
    if windowed_signals.shape[2] // 256 < 4: ## Default value is 256
        nperseg = windowed_signals.shape[2] // 4
    else:
        nperseg = None ## Let scipy set it to default value
    
    paired_sum_normalized_csd = []
    
    for dim in range(windowed_signals.shape[0]):
        temp_list = []
        signal_pairs = []
        for i, x in enumerate(windowed_signals[dim, :, :]):
            for j, y in enumerate(windowed_signals[dim, :, :]):
                if i < j:
                    temp_list.append(
                    (np.abs(scipy.signal.csd(x, y, nperseg=nperseg)[1]) ** 2).sum() / (scipy.signal.csd(x, x, nperseg=nperseg)[1] * scipy.signal.csd(y, y, nperseg=nperseg)[1]).sum()
                    )
                    signal_pairs.append(str(i+1)+'-'+str(j+1))
        paired_sum_normalized_csd.append(temp_list)

    paired_sum_normalized_csd = np.array(paired_sum_normalized_csd)

    windowed_sum_normalized_csd = sm.apply_windowed(data, sm.sum_normalized_csd, window_length=window_length, step=step)
    
    if figsize is None: 
        figsize = (8,4)
    if custom_cols is None: 
        custom_cols = ('YlOrRd', 'black')
        
    # Plotting 
    figure, (ax1, cax) = plt.subplots(nrows=2, figsize=figsize,  gridspec_kw={"height_ratios":[1, 0.05]})
    figure.subplots_adjust(hspace=0.5)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05]) # adding a second grid ontop of the subplots to overlay the group metric line onto the heatmap 
    ax2 = figure.add_subplot(gs[0], frame_on=False)
    ax1.set_xlim(0, paired_sum_normalized_csd.shape[0]) # manually setting the x axis limits to ensure ticks of both plots are aligned 
    ax2.set_xlim(0, windowed_sum_normalized_csd.shape[0])
    # plotting the heatmap 
    cbar_kws = {'label':'Sum-normalized CSD', 'orientation':'horizontal'}
    sns.heatmap(paired_sum_normalized_csd.T, linewidth=0, cmap=custom_cols[0], cbar_kws=cbar_kws, cbar_ax=cax, ax=ax1, yticklabels=signal_pairs, vmin=0, vmax=1) 
    # plotting the group metric
    ax2.plot(windowed_sum_normalized_csd, color=custom_cols[1], linewidth=1)
    # editing axes labels and parameters 
    ax1.tick_params(labelrotation=0) 
    ax1.set(ylabel='Pairwise CSD')
    ax1.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set(ylabel='Group CSD', xlabel=f'Time (window_length={window_length}, step={step})') 
    ax2.yaxis.set_label_position('right')
    if bounded_yaxis is True: 
        ax2.set_ylim(0, 1)
    
    return figure


def plot_rho(phases, relative_phases=False, start=None, end=None, gradient=False, figsize=None, custom_cols=None, bounded_yaxis=True):
    """
    Plots the group rho over a heatmap of the relative individual rho by default, and relative phase values of individual signals by choice.  
    
    Parameters
    ----------
    data: ndarray
        An array containing the multivariate time series with shape (number_signals, duration).
    relative_phases: bool 
        Determines whether the individual relative phases (if True) or the individual rho values (if False) are plotted in the heatmap. Default is False, to plot the individual rho. 
    start: int
        The starting point of the plot. Default is the first timestep of the signals' duration. 
    end: int
        The ending point of the plot. Default is the last timestep of the signals' duration. 		
    gradient: bool 
        Determines whether the heatmap is displayed as a gradient, or matches the window and step size of the group metric. Default is False. 
    figsize: tuple
        The width and height of the figure in inches, respectively. Default is (8,4). 
    custom_cols: tuple 
        The colors and line color to use for the plots. Default is ('YlOrRd', 'black'). 
    bounded_yaxis: bool
        Determines whether the y-axis limits of the group metric are bounded to their range. Default is True. 
        
    Returns
    -------
    figure: figure
        A figure containing the time-varying group rho and individual rho, along with axes labels and a colorbar.
    """
    q_dash = np.exp(phases * 1j).mean(axis=0)
    q = np.arctan2(q_dash.imag, q_dash.real)
    phi = phases - q
    phi_bar_dash = np.exp(phi * 1j).mean(axis=1)

    phi_bar = np.arctan2(phi_bar_dash.imag, phi_bar_dash.real)
    rho_individual = np.real(np.exp((phi - phi_bar[:,None]) * 1j))

    time_varying_rho, mean_rho = sm.rho(phases)

    if figsize is None: 
        figsize = (8,4)
    if custom_cols is None: 
        custom_cols = ('YlOrRd', 'black')
    yticks = []
    for i in range(phases.shape[0]): 
        yticks.append(i+1)
    
    # Plotting 
    figure, (ax1, cax) = plt.subplots(nrows=2, figsize=figsize,  gridspec_kw={"height_ratios":[1, 0.05]})
    figure.subplots_adjust(hspace=0.5)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05]) # adding a second grid ontop of the subplots to overlay the group metric line onto the heatmap 
    ax2 = figure.add_subplot(gs[0], frame_on=False)
    if start is None: 
        start = 0
    if end is None: 
        end = phases.shape[1]
    ax1.set_xlim(start, end) # manually setting the x axis limits to ensure ticks of both plots are aligned 
    ax2.set_xlim(start, end)
    # plotting the heatmap 
    cbar_kws = {'label':'Cluster-phase Rho', 'orientation':'horizontal'}
    sns.heatmap(rho_individual, linewidth=0, cmap=custom_cols[0], cbar_kws=cbar_kws, cbar_ax=cax, ax=ax1, yticklabels=yticks, vmin=0, vmax=1) 
    # plotting the group metric
    ax2.plot(time_varying_rho, color=custom_cols[1], linewidth=1)
    
    # editing axes labels and parameters 
    ax1.tick_params(labelrotation=0) 
    ax1.set(ylabel='Individual Rho')
    ax1.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.set(ylabel='Group Rho', xlabel='Time') 
    ax2.yaxis.set_label_position('right')
    if bounded_yaxis is True: 
        ax2.set_ylim(0, 1)
    
    return figure


def plot_synchronyindex(data, window_length, step=None, driver=True, lag_length=10, gradient=False, figsize=None, custom_cols=None): 
    """
    Plots the windowed synchrony index over a heatmap of either the drivers or empaths of synchrony. 
    
    Parameters
    ----------
    data: ndarray
        An array containing the multivariate time series with shape (number_signals, duration).
    window_length: int
        The number of time steps to be included in a window.
    step: int
        The number of time steps by which to move forward in order to obtain the next window. 
    driver: bool
        Determines whether the drivers (if True) or the empaths (if False) are plotted in the heatmap. Default is True, to plot the drivers. 
    lag_length: int
        The number of samples for lag length - used by the synchrony metric, default = 10.
    gradient: bool 
        Determines whether the heatmap is displayed as a gradient, or matches the window and step size of the group metric. Default is False. 
    figsize: tuple
        The width and height of the figure in inches, respectively. Default is (8,4). 
    custom_cols: tuple 
        The colors and line color to use for the plots. Default is ('YlOrRd', 'black'). 

    Returns
    -------
    figure: figure
        A figure containing the windowed group synchrony index and the windowed driver or empath scores for each signal, along with axes labels and a colorbar.
    """    
    if step is None: 
        step = window_length 

    synchrony_index = sm.apply_windowed(
        data, 
        lambda x: sm.get_sync_index(x, lag_length=lag_length), 
        window_length=window_length, 
        step=step)
        
    if figsize is None:  
        figsize = (8,4)  
    if custom_cols is None: 
        custom_cols = ('YlOrRd', 'black')
    yticks = []
    for i in range(data.shape[0]): 
        yticks.append(i+1)
    
    # Plotting
    figure, (ax1, cax) = plt.subplots(nrows=2, figsize=figsize,  gridspec_kw={"height_ratios":[1, 0.05]})
    figure.subplots_adjust(hspace=0.5)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05]) # adding a second grid ontop of the subplots to overlay the group metric line onto the heatmap 
    ax2 = figure.add_subplot(gs[0], frame_on=False)
    # Driver visualization
    if driver == True:   
        if gradient == True: 
            w_data = sm.apply_windowed(
                data, 
                lambda x: sm.get_driver_scores(x, lag_length=lag_length), 
                window_length=window_length, 
                step=5) # step size of 5 for the gradient
        else: 
            w_data = sm.apply_windowed(
                data, 
                lambda x: sm.get_driver_scores(x, lag_length=lag_length), 
                window_length=window_length, 
                step=step)
            
        norm_w_data = []
        for window in range( len( w_data[ : , 0 ] ) ):
            norm_w_data.append((w_data[ window ] - np.amin(w_data[ window ])) / (np.max(w_data[ window ]) - np.min(w_data[ window ])))
        norm_w_data = np.array(norm_w_data).T
        
        cbar_kws = {'label':'Driver score', 'orientation':'horizontal'}
        sns.heatmap(norm_w_data, linewidth=0, cmap=custom_cols[0], cbar_kws=cbar_kws, cbar_ax=cax, ax=ax1, yticklabels=yticks, vmin=0, vmax=1) 
        ax1.set(ylabel='Individual Driver Scores')
        ax2.plot(synchrony_index, color=custom_cols[1], linewidth=1)  
        ax1.set_xlim(0, norm_w_data.shape[1]) # manually setting the x axis limits to ensure ticks of both plots are aligned 
        ax2.set_xlim(0, len(synchrony_index))
    # Empath visualization
    else:   
        if gradient == True: 
            w_data = sm.apply_windowed(
                data, 
                lambda x: sm.get_empath_scores(x, lag_length=lag_length), 
                window_length=window_length, 
                step=5) # step size of 5 for the gradient
        else: 
            w_data = sm.apply_windowed(
                data, 
                lambda x: sm.get_empath_scores(x, lag_length=lag_length), 
                window_length=window_length, 
                step=step)
            
        norm_w_data = []
        for window in range( len( w_data[ : , 0 ] ) ):
            norm_w_data.append((w_data[ window ] - np.amin(w_data[ window ])) / (np.max(w_data[ window ]) - np.min(w_data[ window ])))
        norm_w_data = np.array(norm_w_data).T
        
        cbar_kws = {'label':'Driver score', 'orientation':'horizontal'}
        sns.heatmap(norm_w_data, linewidth=0, cmap=custom_cols[0], cbar_kws=cbar_kws, cbar_ax=cax, ax=ax1, yticklabels=yticks, vmin=0, vmax=1) 
        ax1.set(ylabel='Individual Empath Scores')
        ax2.plot(synchrony_index, color=custom_cols[1], linewidth=1)  
        ax1.set_xlim(0, norm_w_data.shape[1]) # manually setting the x axis limits to ensure ticks of both plots are aligned 
        ax2.set_xlim(0, len(synchrony_index))
    # editing axes labels and parameters 
    ax1.tick_params(labelrotation=0) 
    ax1.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set(ylabel='Group Synchrony Index', xlabel=f'Time (window_length={window_length}, step={step})') 
    ax2.yaxis.set_label_position('right')

    return figure