# Nonparametric Demand Estimation
Code implementing the nonparametric estimator proposed in the [Nonparametric demand estimation in the presence of unobserved factors](https://ssrn.com/abstract=4244086) paper.

## How to run
Modify the following variables in the `constants.py` file as required. Refer to the comments in the file for variable descriptions. 

`file_path_results`, \
`file_path_data`, \
`file_name`, \
`num_feats`, \
`num_markets`, \
`num_prods`, \
`ALGO_VARIANT`, \
`g0`, \
`xi_per`, \
`num_iterations_IgnoreUF`, \
`num_iterations_AltDesc` 

The input data file should contain the following columns, arranged in the same order (this is the same format used in the PyBLP estimator of Conlon and Gortmaker). For an example, refer to the Sample_data.csv file.

`market_ids`, \
`product_ids`, \
`shares`, \
`prices`, \
`x_0`, `x_1`, $\cdots$, \
`demand_instruments0`,`demand_instruments1`, $\cdots$ 

Run the file `nonparametric_experiments.py` that contains two functions: `run_nonparametric_estimator()` and `compute_metrics()`.

### 1. `run_nonparametric_estimator()`
This function runs a nonparametric estimator on market data. It first calls the `compute_variables_from_expt_file` function to process the input data file and generate market shares, product features, and instruments in the required format. 

It then fits a `FrankWolfeMixedLogitBLPEstimator` object for estimation and saves the output in the `file_path_results` directory. The output is a pickle file that contains the estimated random coefficients and their mixing proportions, the unobserved factors (demand shocks in BLP framework), the KL divergence loss and the GMM objective values at the estimated model parameters.

### 2. `compute_metrics()`
This function computes the following post-estimation in-sample and out-of-sample metrics:
- In-sample price elasticity. 
- In-sample own-price elasticity.
- In-sample predicted market shares based on training data.
- Out-of-sample predicted market shares at counterfactual prices. We randomly select one product in each market and increase its price by 1%, 5%, and 10%, respectively. We then compute the market shares based on the counterfactual prices using the estimates of our model.

## Dependencies
The code has been tested with the following (main) dependencies:

python                    3.8.5\
numpy                     1.20.1\
pandas                    1.2.2\
scikit-learn              0.24.1\
scipy                     1.10.

We conducted tests on this code using a High-Performance Computing (HPC) system equipped with 64GB of CPU memory. Given the memory-intensive nature of the code, we recommend running it on an HPC environment for optimal performance.
