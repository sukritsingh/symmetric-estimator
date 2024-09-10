# symmetric-estimator

An estimator for nonequilibrium averages for symmetric alchemical protocols in which forward and reverse processes are identical.

This code is a sandbox to try out new estimators to extract free energy differences from
non-equilibrium cycling and other symmetric alchemical protocols. The code is written in Python.

Implementing the protocol written about in [Nonequilibrium path-ensemble averages for symmetric protocols](https://doi.org/10.1063/1.5121306)

## Contents

Inside `symmetric-estimator` you will find three scripts: 
1. `utils.py` - this has all the utility functions that are used in the other two scripts
2. `estimators.py` - this has the main estimator functions that are used in `fe_functions.py`
3. `fe_functions.py` - this has the functions that are used to compute free energy differences from the estimated quantities