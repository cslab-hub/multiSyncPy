"""Multivariate Synchrony

This module provides functions to compute multivariate synchrony metrics and to generate synthetic data. It contains the following subpackages:

 * synchrony_metrics - Provides functions used to compute synchrony metrics on multivariate time series.
 * data_generation - Provides functions used to generate synthetic data for the purposes of testing and exploring multivariate synchrony metrics. 
 * visualisations - Provides functions to calculate and then visualise multivariate synchrony metrics. 
"""

__all__ = ["data_generation", "synchrony_metrics", "visualisations"]

import multiSyncPy.data_generation
import multiSyncPy.synchrony_metrics
import multiSyncPy.visualisations
