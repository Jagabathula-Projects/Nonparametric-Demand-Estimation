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
`ALGO_VARIANT`

The input data file should contain the following columns, arranged in the same order (this is the same format used in the PyBLP estimator of Conlon and Gortmaker). For an example, refer to the Sample_data.csv file.

`market_ids`, \
`product_ids`, \
`shares`, \
`prices`, \
`x_0`, `x_1`, $\cdots$, \
`demand_instruments0`,`demand_instruments1`, $\cdots$ 

Run the file `nonparametric_experiments.py` that contains two functions: `run_nonparametric_estimator()` and `compute_metrics()`.

### 1. `run_nonparametric_estimator(file_name, num_feats)`
This function runs a nonparametric estimator on market data by taking `file_name` and `num_feats` as inputs. It first calls the `compute_variables_from_expt_file` function to process the input data file and generate market shares, product features, and instruments in the required format. 

It then fits a `FrankWolfeMixedLogitBLPEstimator` object for estimation and saves the output in the `file_path_results` directory. The output is a pickle file that contains the estimated random coefficients and their mixing proportions, the unobserved factors (aka demand shokcs), the KL divergence loss and the GMM objective value at the estimated model parameters.

### 2. `compute_metrics(file_name)`
This function computes the following post-estimation in-sample and out-of-sample metrics:
- In-sample price elasticity. 
- In-sample own-price elasticity.
- In-sample predicted market shares based on training data.
- Out-of-sample predicted market shares at counterfactual prices. We randomly select one product in each market and increase its price by 1%, 5%, and 10%, respectively. We then compute the market shares based on the counterfactual prices using the estimates of our model.


<!-- ## Utility model
The utility of customer $i$ for product $j$ in market $t$ is of the form $u_{ijt} = \bm{omega}_{i}^{\top} \bm{x}_{jt} + \xi_{jt} + \epsilon_{ijt}$ , where the coefficient vector $\bm{omega}_i \in \Real^D$ is sampled from some (unknown) distribution $Q$, the intercept $\xi_{jt}$ is a product and offer-set specific factor that is {\em not} fully observed by the firm, and $\epsilon_{ijt}$ is an error term.  -->

<!-- 
## Estimator description
The code for the estimator is in `frank_wolf_lc_mnl.py`. The class variable *mix_props* stores the mixing proportions in a 1-d array and the variable *coefs_* stores the beta parameters in a 2-d array, such that the entry in row $k$ and column $j$ corresponds to $\beta_{kj}$. Other variables are described in the file.

The main method in the estimator is `fit_to_choice_data()`, which takes as input the membership matrix (binary matrix that encodes whether a product is offered in each offerset) and the number of sales for each product in each offerset. We transform the data from the provided input file to this format; refer the documentation for more details on the input format. The following arguments to the `fit_to_choice_data()` method can be modified based on the application:

1. *num_iters* : this is the number of iterations to run the estimation for. As mentioned in the paper, this provides an upper bound for the number of latent classes in the estimated LC-MNL model.
2. *init_coefs* and *init_mix_props*: the initial betas and mixture proportions. 

## Out-of-sample choice predictions
After estimating the model, you should use the `predict_choice_proba()` function to predict choice probabilities on out-of-sample-transactions, which are also provided in the example instance *test_instance.dt*. An example of how to do this is provided in the *run_estimator.py* file. -->

<!-- ## Dependencies
The code has been tested with the following (main) dependencies:

numpy==1.20.3

scipy==1.7.1

multiprocess==0.70.12.2

ipython==7.26.0 -->
