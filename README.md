# Drone-Tracking Repository

## Implemented: 
Simple Kalman Filter 
Simulated measurements [Linear]



## References: 
  Simple Kalman Filter: <br>
  -https://www.geeksforgeeks.org/python/kalman-filter-in-python/ <br>
  -https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb <br>
  -https://stackoverflow.com/questions/66007351/kalman-filter-2d-with-pykalman <br>

  Math behind the simple kalman filter: <br>
  -https://aleksandarhaber.com/introduction-to-kalman-filter-derivation-of-the-recursive-least-squares-method-with-python-codes/ <br>
  -https://aleksandarhaber.com/time-propagation-of-state-vector-and-state-covariance-matrix-of-linear-dynamical-systems-intro-to-kalman-filtering/ <br>
  -https://aleksandarhaber.com/kalman-filter-complete-derivation-from-scratch/ <br>

  (G)OSPA distance: <br>
  -https://ieeexplore-ieee-org.tudelft.idm.oclc.org/stamp/stamp.jsp?tp=&arnumber=8009645 <br>
  -https://ieeexplore-ieee-org.tudelft.idm.oclc.org/stamp/stamp.jsp?tp=&arnumber=5744132&tag=1 <br>
  Stone soup documentation:<br>
  -https://stonesoup.readthedocs.io/en/v0.1b4/stonesoup.metricgenerator.ospametric.html <br>
  

## Changes/Notes:

Error covariance matrix P has to have a large trace! The initial certainty must be very low to adopt changes faster.

Added OSPA distance as an evaluation metric.
