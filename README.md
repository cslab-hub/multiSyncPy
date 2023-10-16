**Important announcement** - *There is a new version of multiSyncPy (0.1.0) that includes a new multivariate synchronization measures as well as some visualiation functions.*

# multiSyncPy

multiSyncPy is a Python package for quantifying multivariate synchrony. Our package supports the burgeoning field of research into synchrony, making accessible a set of methods for studying group-level rather than dyadic constructs of synchrony and/or coordination. We offer a range of metrics for estimating multivariate synchrony based on a collection of those used in recent literature.

The main methods of this package are functions to calculate:

 * symbolic entropy, 
 * multidimensional recurrence quantification, 
 * coherence (and a related 'sum-normalized CSD' metric),
 * the cluster-phase 'Rho' metric
 * the synchronization coefficient metric,
 * a statistical test based on the Kuramoto order parameter, and
 * driver-empath model with synchrony index

We also include functions for two surrogation techniques to compare the observed coordination dynamics with chance levels.

Additionally, we include a set of functions to visualize the time-varying coordination metrics as well as the individual or pair-wise contributions to the multivariate measure (depending on the particular method).

multiSyncPy is freely available under the LGPL license. The source code is maintained at <https://github.com/cslab-hub/multiSyncPy>, which also includes examples of usage of the package. Documentation can be accessed through `help()` or accessed at <https://cslab-hub.github.io/multiSyncPy/>. 

Further details of the package and case studies of its use on real-world data are described in our paper. 

Hudson, D., Wiltshire, T. J., & Atzmueller, M. (2023). multiSyncPy: A Python package for assessing multivariate coordination dynamics. Behavior Research Methods, 55(2), 932-962. <https://doi.org/10.3758/s13428-022-01855-y>. 

Please cite this paper if you use multiSyncPy in your research.
