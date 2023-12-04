import numpy as np

LIKELIHOOD = 'likelihood'
QUADRATIC = 'quadratic'

# FRANK-WOLFE MIXTURE LEARNING ALGORITHM CONSTANTS
NUM_BFGS_INIT = 20
NUM_SUBPROB_ITERS = 50

MAX_MSS_ITERS = 1000
MAX_BS_ITERS = 10
MAX_CP_ITERS = 30

CORRECTIVE_GAP = 1e-8
MAX_CORRECTIVE_STEPS = 400
FOX_BAJARI_GAP = 1e-1
BB_TOL = 1e-4
BS_TOL = 1e-6


# NORM_TYPE = 'linf'
NORM_TYPE = 'l2'
DEMAND_SHOCKS_MAX = 1
no_purch_const = 0

xi_per_ub = 0.5
g0 = 0.0


XI_ZERO_MEAN = False
NUM_BETA_UPDATES = 1
NUM_XI_UPDATES = 1
BETA_UB = 6
num_FW_iterations = 50

PRICE_CHANGE_PERCENT_1 = 1.0
PRICE_CHANGE_PERCENT_2 = 5.0
PRICE_CHANGE_PERCENT_3 = 10.0


#  ALGO_VARIANT AltDesc refers to the nonparametric estimator in the paper "Nonparametric demand estimation in the presence of unobserved factors."
#  ALGO_VARIANT IgnoreUF refers to the nonparametric estimator ignoring the unobserved factors proposed in the paper "A Conditional Gradient Approach for Nonparametric Estimation of Mixing Distributions."
ALGO_VARIANT = 'AltDesc'
# ALGO_VARIANT = 'IgnoreUF'

# Directory that contains the input data file
file_path_data = ''
# Directory to save the results of the estimator
file_path_results = 'Results/'
# Name of the input csv file. Ensure that the column names in your input file match those in the Sample_data.csv file, and are in the same order.
file_name = 'Sample_data'
# Number of features in the utility specification of the model. Columns x_0, x_1, x_2 and so on correspond to the non-price features, and the column prices correspond to the price.
num_feats = 2
# Total number of markets
num_markets = 400
# Total number of products. The current code requires all products to be offered in all markets. We can modify the code to accommodate varying offer sets upon request.
num_prods = 25

NOPURCHSHOCKS = np.zeros(num_markets)[:, np.newaxis]
