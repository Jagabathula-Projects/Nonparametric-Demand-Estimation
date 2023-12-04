import numpy as np

LIKELIHOOD = 'likelihood'
QUADRATIC = 'quadratic'

# MIXTURE LEARNING ALGORITHM CONSTANTS
NUM_BFGS_INIT = 20
NUM_SUBPROB_ITERS = 50

CORRECTIVE_GAP = 1e-8
MAX_CORRECTIVE_STEPS = 400

# NORM_TYPE = 'linf'
NORM_TYPE = 'l2'
DEMAND_SHOCKS_MAX = 1
no_purch_const = 0

XI_ZERO_MEAN = False
NUM_BETA_UPDATES = 1
NUM_XI_UPDATES = 1

# OUT-OF-SAMPLE PRICE CHANGES
PRICE_CHANGE_PERCENT_1 = 1.0
PRICE_CHANGE_PERCENT_2 = 5.0
PRICE_CHANGE_PERCENT_3 = 10.0


# Directory that contains the input data file
file_path_data = ''
# Directory to save the results of the estimator
file_path_results = 'Results/'
# Name of the input csv file. Ensure that the column names in your input file match those in the Sample_data.csv file, and are in the same order.
file_name = 'Sample_data'
# Total number of features (including price) in the utility specification of the model. Columns x_0, x_1, x_2 and so on correspond to the non-price features, and the column prices correspond to the price.
num_feats = 2
# Total number of markets
num_markets = 400
# Total number of products. The current code requires all products to be offered in all markets. We can modify the code to accommodate varying offer sets upon request.
num_prods = 25

NOPURCHSHOCKS = np.zeros(num_markets)[:, np.newaxis]


#  ALGO_VARIANT AltDesc refers to the nonparametric estimator in the paper "Nonparametric demand estimation in the presence of unobserved factors."
#  ALGO_VARIANT IgnoreUF refers to the nonparametric estimator ignoring the unobserved factors proposed in the paper "A Conditional Gradient Approach for Nonparametric Estimation of Mixing Distributions."
ALGO_VARIANT = 'AltDesc'
# ALGO_VARIANT = 'IgnoreUF'

# Hyperparameters corresponding to ALGO_VARIANT = 'AltDesc'
# g0 = 0.0 and xi_per = 0.5 are the default values, but they must be tuned based on the application.
g0 = 0.0
xi_per = 0.5

# Number of iterations to run for the corresponding algorithms
num_iterations_IgnoreUF = 50
num_iterations_AltDesc = 10
